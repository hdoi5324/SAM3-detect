from collections import defaultdict
import numpy as np

from pathlib import Path
import cv2
import torch
from PIL import Image
from sqapi import Annotator
from sqapi.media import SQMediaObject
import ast

import sam3
from .squidle_utils import load_squidle_annotations_to_df

SAM3_DIR = Path(sam3.__file__).resolve().parent
BPE_PATH = SAM3_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"

from sam3_override.model_builder import build_sam3_image_model
from sam3_override.sam3_image_processor import Sam3Processor
from sam3_override.utils import autocast_scope
from .sam_utils import clean_polygon, mask_to_polygon, check_polygon, create_annotation_polygon_with_label_id, bbox_of_vertices, polygon_is_bbox
from exemplar_seg.postprocess import (
    assign_candidates_to_points,
    build_candidate,
    points_inside_mask,
    suppress_overlapping_candidates,
)

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "none": None,
}


class SAM3Segmenter(Annotator):
    def __init__(self, device="cuda",
                 use_autocast=True,
                 autocast_dtype="bf16",
                 model_path="models/sam3.pt",
                 add_new_annotations=True,
                 add_new_threshold=0.5,
                 replace_existing_polygons=True,
                 iou_threshold=0.5,
                 n_exemplars=3,
                 n_negative=0,
                 n_images_to_process=0,
                 sam_resolution=1008,
                 embed_merge="avg_cat",
                 point_prompt_mode="combined",
                 points_direct_project=True,
                 points_pool=True,
                 points_pos_enc=True,
                 boxes_direct_project=True,
                 boxes_pool=True,
                 boxes_pos_enc=True,
                 polygon_sample_points=8,
                 **annotator_args):
        super().__init__(**annotator_args)
        self.device = device
        self.use_autocast = use_autocast
        self.autocast_dtype = _DTYPE_MAP.get(autocast_dtype.lower(), torch.bfloat16)
        self.checkpoint_path = model_path
        self.confidence_threshold = self.prob_thresh
        self.add_new_annotations = add_new_annotations
        self.add_new_threshold = add_new_threshold
        self.replace_existing_polygons = replace_existing_polygons
        self.sam_resolution = sam_resolution if sam_resolution is not None else 1008 # default used by SAM3
        self.n_exemplars = n_exemplars if n_exemplars is not None else 3
        self.n_negative = n_negative if n_negative is not None else 0
        self.n_images_to_process = n_images_to_process if n_images_to_process is not None else 0
        self.iou_threshold = iou_threshold
        self.embed_merge = embed_merge if embed_merge is not None else "avg_cat"
        self.polygon_sample_points = (
            polygon_sample_points if polygon_sample_points is not None else 8
        )
        # "combined": single run with exemplar + point prompts concatenated.
        # "point_only": point prompts only (skips exemplar encoding; for testing).
        self.point_prompt_mode = (point_prompt_mode or "combined").lower()
        if self.point_prompt_mode not in ("combined", "point_only"):
            raise ValueError(
                f"point_prompt_mode must be 'combined' or 'point_only'; "
                f"got {point_prompt_mode!r}"
            )

        # Load Sam3 model
        print("Loading SAM model...")
        # todo: how to download the sam3.pt model from hugging face to models/sam3.pt
        with autocast_scope(device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast):
            self.model = build_sam3_image_model(
                bpe_path=BPE_PATH,
                device=self.device,
                checkpoint_path=self.checkpoint_path,
                points_direct_project=points_direct_project,
                points_pool=points_pool,
                points_pos_enc=points_pos_enc,
                boxes_direct_project=boxes_direct_project,
                boxes_pool=boxes_pool,
                boxes_pos_enc=boxes_pos_enc,
            )
            self.model.to(self.device)
            self.model.eval()

    def get_exemplar_prompts(self, exemplar_data_by_label_id, target_label_id,
                                           combine_prompts="avg_cat", encode_text=True, n_exemplars=3):
        exemplar_processor = Sam3Processor(self.model, resolution=self.sam_resolution, confidence_threshold=self.confidence_threshold)
        exemplar_data_dict = exemplar_data_by_label_id[target_label_id]
        return exemplar_processor.get_exemplar_prompts(exemplar_data_dict,
                                                       combine_prompts=combine_prompts,
                                                       encode_text=encode_text,
                                                       n_exemplars=n_exemplars,
                                                       n_polygon_sample_points=self.polygon_sample_points)

    def _partition_points_by_label(self, points):
        """
        Split query points by annotation label_id.

        Labeled points drive per-label SAM3 prompts. Unlabeled points (label_id None)
        are kept for fallback mask assignment in exemplar modes only (not point_only).
        """
        points_by_label_id = defaultdict(list)
        unlabeled_points = []
        for idx, p in enumerate(points):
            annotations = p.get("annotations")
            if not annotations:
                continue
            label_id = annotations[0].get("label_id")
            entry = {"index": idx, "x": p["x"], "y": p["y"]}
            if p.get('data') is not None:
                if p.get('data').get('polygon') is not None:
                    entry['polygon'] = p.get('data').get('polygon')
            if label_id is None:
                unlabeled_points.append(entry)
            else:
                points_by_label_id[label_id].append(entry)
        return points_by_label_id, unlabeled_points

    def _label_point_pixel_coords(self, label_points, mediaobj):
        """Pixel [row, col] coords and global point indices for one label."""
        if not label_points:
            return np.empty((0, 2), dtype=int), []
        rows = np.clip(
            [int(p["y"] * mediaobj.height) for p in label_points], 0, mediaobj.height - 1
        )
        cols = np.clip(
            [int(p["x"] * mediaobj.width) for p in label_points], 0, mediaobj.width - 1
        )
        pts_coords = np.stack([rows, cols], axis=1)
        point_indices = [p["index"] for p in label_points]
        return pts_coords, point_indices

    def _iter_label_jobs(self, points_by_label_id):
        """
        Yield one SAM3 job per label_id: exemplar prompt(s) paired with that label's points.

        Each job runs in isolation — point prompts and mask hit-testing use only points
        belonging to the same label_id.
        """
        if self.point_prompt_mode == "point_only":
            for label_id, label_points in points_by_label_id.items():
                if label_points:
                    yield label_id, None, None, label_points
            return

        for label_id, [prompt, prompt_mask] in self.prompts_by_label_id.items():
            if label_id is None:
                continue
            yield label_id, prompt, prompt_mask, points_by_label_id.get(label_id) or []

    def _masks_to_candidates(
        self,
        scores,
        masks,
        label_id,
        label_pts_coords,
        point_indices,
        unlabeled_pts_coords,
        unlabeled_point_indices,
        mediaobj,
    ):
        candidates = []
        for i in range(len(scores)):
            mask_np = masks[i].squeeze(0).cpu().numpy()
            prob = scores[i].item()
            hit_local = points_inside_mask(mask_np, label_pts_coords)
            hit_indices = [point_indices[j] for j in hit_local]
            fallback_local = points_inside_mask(mask_np, unlabeled_pts_coords)
            fallback_indices = [unlabeled_point_indices[j] for j in fallback_local]
            raw_polys = mask_to_polygon(mask_np)
            if not raw_polys:
                continue
            poly = clean_polygon(raw_polys[0], mediaobj.width, mediaobj.height)
            if not check_polygon(poly, mediaobj.width, mediaobj.height):
                continue
            cand = build_candidate(
                score=prob,
                label_id=label_id,
                poly=poly,
                hit_point_indices=hit_indices,
                fallback_hit_point_indices=fallback_indices,
            )
            if cand is not None:
                candidates.append(cand)
        return candidates

    def _run_label_grounding(
        self,
        processor,
        inference_state,
        prompt,
        prompt_mask,
        box_inputs,
        box_labels,
        point_inputs,
        point_labels,
    ):
        """Run one SAM3 forward pass for a label; returns updated inference_state."""
        if self.point_prompt_mode == "combined" and (box_inputs or point_inputs):
            inference_state = processor.add_geometric_prompts_to_state(
                box_inputs, box_labels, point_inputs, point_labels, inference_state
            )
            return processor.forward_grounding_with_exemplar_and_point_prompts(
                inference_state, prompt, prompt_mask
            )
        return processor.forward_grounding_with_prompt_embeddings(
            inference_state, prompt, prompt_mask
        )

    def _collect_label_candidates(
        self,
        processor,
        inference_state,
        prompt,
        prompt_mask,
        label_id,
        label_points,
        label_pts_coords,
        point_indices,
        unlabeled_pts_coords,
        unlabeled_point_indices,
        mediaobj,
    ):
        """Run exemplar and/or point prompts for a single label_id; return mask candidates."""
        box_inputs, box_labels, point_inputs, point_labels = processor.build_query_geometry_prompts(
            label_points,
            n_polygon_sample_points=self.polygon_sample_points,
        )
        has_query_geometry = bool(box_inputs or point_inputs)
        has_exemplar_prompt = prompt is not None and prompt_mask is not None
        if not has_query_geometry and (
            self.point_prompt_mode == "point_only" or not has_exemplar_prompt
        ):
            return []
        processor.reset_all_prompts(inference_state)
        with autocast_scope(device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast):
            if self.point_prompt_mode == "point_only":
                inference_state = processor.forward_grounding_with_geometry_prompts(
                    inference_state,
                    box_inputs,
                    box_labels,
                    point_inputs,
                    point_labels,
                )
            else:
                inference_state = self._run_label_grounding(
                    processor,
                    inference_state,
                    prompt,
                    prompt_mask,
                    box_inputs,
                    box_labels,
                    point_inputs,
                    point_labels,
                )
        scores = inference_state["scores"]
        masks = inference_state["masks"]
        if scores.numel() == 0:
            return []
        return self._masks_to_candidates(
            scores, masks, label_id, label_pts_coords, point_indices,
            unlabeled_pts_coords, unlabeled_point_indices, mediaobj,
        )

    def email_annotation_set_user(self, a, counts):
        # todo: update this to provide more information about the config used.
        user_ids = [a.get('user', {}).get('id')]
        annotation_set_url = "{}/geodata/annotation_set/{}".format(self.sqapi.host, a.get("id"))
        message = f'Hi {a.get("user", {}).get("first_name")}, <br><br>\n' \
                  f'Your annotation set "<b>{a.get("media_collection", {}).get("name")} / {a.get("name")}</b>" has been ' \
                  f'processed by {self.annotator_info.get("name")}.<br><br>\n' \
                  f'Analysed: {counts["media"]} media items, {counts["points"]} points and {counts["annotations"]} annotations.<br>\n' \
                  f'Created/updated: {counts["new_points"]} points and {counts["new_annotations"]} annotations.<br><br>\n' \
                  f'Any Label suggestions will appear as "Magical Suggestions" ' \
                  f'in the annotation window and can be validated using the QA/QC tool.<br><br>\n' \
                  f'To see results, click: <a href="{annotation_set_url}">{annotation_set_url}</a>'
        self.sqapi.send_user_email("SQ+ BOT: your Annotation Set has been processed", message, user_ids=user_ids)


    def annotate_media_list(self, annotation_set_data, page=1, results_per_page=500):
        """

        :param annotation_set_data:
        :param page:
        :param results_per_page:
        :return:
        """
        annotation_set_id = annotation_set_data.get("id")
        base_annotation_set_id = annotation_set_data.get("parent_id") or annotation_set_data.get("id")
        media_collection_id = annotation_set_data.get("media_collection", {}).get("id")
        # media_list =self.get_media_collection_media(media_collection_id, page=page, results_per_page=results_per_page)
        media_list = self.sqapi.get("/api/media", page=page, results_per_page=results_per_page).filter(
            name="media_collections", op="any", val=dict(name="id", op="eq", val=media_collection_id)
        ).order_by(field="timestamp_start", direction="asc").execute().json()
        num_results = media_list.get('num_results')

        counts = dict(media=0, points=0, annotations=0, new_points=0, new_annotations=0)

        self.exemplars_by_label_id = {}
        self.prompts_by_label_id = {}
        if self.point_prompt_mode == "point_only":
            print("point_prompt_mode=point_only: skipping exemplar loading and encoding")
        else:
            # Load exemplar images and point/bbox prompts from the base annotation set
            self.exemplars_by_label_id = self.load_exemplar_annotations(base_annotation_set_id)
            with autocast_scope(device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast):
                for label_id in self.exemplars_by_label_id.keys():
                    prompt_data = self.get_exemplar_prompts(self.exemplars_by_label_id, label_id,
                                                            n_exemplars=self.n_exemplars,
                                                            combine_prompts=self.embed_merge)
                    self.prompts_by_label_id[label_id] = prompt_data

        media_no = 0
        for m in media_list.get("objects"):
            media_no = counts['media'] + (page - 1) * results_per_page
            if self.n_images_to_process > 0 and media_no > self.n_images_to_process:
                break
            counts['media'] += 1
            print(f"\nProcessing: media item {counts['media'] + (page - 1) * results_per_page} / {num_results}")
            media_url = m.get('path_best')
            media_type = m.get("media_type", {}).get("name")
            mediaobj = SQMediaObject(media_url, media_type=media_type, media_id=m.get('id'))

            # get media annotations. If this frame has not been observed, it will generat the annotations through the request
            media_annotations = self.sqapi.get(
                f"/api/media/{m.get('id')}/annotations/{base_annotation_set_id}",
            filters=[dict(name="point__has_xy", op="has", val=True)],
            ).execute().json()
            points = media_annotations.get('annotations')
            counts['points'] += len(points)

            # Process all points by mediaobj
            _counts, new_points = self.process_points(points, mediaobj, annotation_set_id, base_annotation_set_id)
            for k in counts.keys():
                counts[k] += _counts[k]

            for p in new_points:
                p['annotation_set_id'] = base_annotation_set_id
                p['media_id'] = mediaobj.id
                if isinstance(p.get('annotation_label'), dict):
                    p['annotation_label']['annotation_set_id'] = annotation_set_id
                    counts['new_annotations'] += 1
                self.sqapi.post("/api/point", json_data=p).execute()
                counts['new_points'] += 1
            print(f"counts: {counts}")

        # continue until all images are processed
        if media_list.get("page") < media_list.get("total_pages") and media_no <= self.n_images_to_process:
            _counts = self.annotate_media_list(annotation_set_data, page=page + 1, results_per_page=results_per_page)
            for k in counts.keys():
                counts[k] += _counts[k]

        return counts


    def process_points(self, points, mediaobj, annotation_set_id, base_annotation_set_id):
        counts = dict(media=0, points=0, annotations=0, new_points=0, new_annotations=0)
        all_label_candidates = []  # candidates from all labels before global NMS

        if not mediaobj.is_processed:
            mediaobj.data()

        points_by_label_id, unlabeled_points = self._partition_points_by_label(points or [])
        if self.point_prompt_mode == "point_only":
            unlabeled_pts_coords = np.empty((0, 2), dtype=int)
            unlabeled_point_indices = []
        else:
            unlabeled_pts_coords, unlabeled_point_indices = self._label_point_pixel_coords(
                unlabeled_points, mediaobj
            )

        orig_image = mediaobj.data()
        image_data = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        with autocast_scope(device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast):
            processor = Sam3Processor(self.model, resolution=self.sam_resolution,
                                      confidence_threshold=self.confidence_threshold)
            inference_state = processor.set_image(Image.fromarray(image_data))

        # --- PHASE 1: COLLECT ALL CANDIDATES ACROSS ALL LABELS ---
        for label_id, prompt, prompt_mask, label_points in self._iter_label_jobs(points_by_label_id):
            label_pts_coords, point_indices = self._label_point_pixel_coords(label_points, mediaobj)
            all_label_candidates.extend(
                self._collect_label_candidates(
                    processor, inference_state, prompt, prompt_mask,
                    label_id, label_points, label_pts_coords, point_indices,
                    unlabeled_pts_coords, unlabeled_point_indices, mediaobj,
                )
            )

        # --- PHASE 2: GLOBAL IOU FILTERING (NMS) + POINT ASSIGNMENT ---
        #todo: this is not needed at this point.  It needs to be done with all candidates from all labels as well as existing points that have polygons or boxes.
        nms_candidates = suppress_overlapping_candidates(
            all_label_candidates,
            iou_threshold=self.iou_threshold,
        )
        accepted_candidates = assign_candidates_to_points(
            nms_candidates,
            add_new_annotations=self.add_new_annotations,
            add_new_threshold=self.add_new_threshold,
        )

        # --- PHASE 3: APPLY UPDATES ---
        new_points_to_return = []
        for cand in accepted_candidates:
            if not cand.is_new:
                if self.replace_existing_polygons:
                    # UPDATE EXISTING POINT AND ANNOTATION if it has better prob
                    if cand.assigned_point_index is None:
                        continue
                    point_obj = points[cand.assigned_point_index]
                    point_ann = (point_obj.get("annotations") or [None])[0]
                    if point_ann is not None and point_ann.get("label_id") not in (None, cand.label_id):
                        continue
                    point_id = point_obj.get("id")
                    # todo: Assumes the first annotation is the relevant one.  Possibility there are multiple annotations.
                    # todo: does this work for supplementary datasets???
                    current_ann = point_obj['annotations'][0]
                    point_has_bbox = polygon_is_bbox(point_obj.get("data", {}).get("polygon"))
                    no_polygon = point_obj.get("data", {}).get("polygon") is None
                    if current_ann['label_id'] is None or current_ann['likelihood'] < cand.score or point_has_bbox or no_polygon:
                        # Update point with new polygon
                        print(f"Updating point_id {point_id}, current point is bbox: {point_has_bbox}, current prob: {current_ann['likelihood']:.2f}; new prob: {cand.score:.2f}; current_label: {current_ann['label_id']}, new label: {cand.label_id}, new polygon vertices: {len(cand.poly)}")
                        data = dict(pixels=dict(polygon=cand.poly, width=mediaobj.width, height=mediaobj.height))
                        try:
                            self.sqapi.patch(f"/api/point/{point_id}", json_data=data).execute()
                        except Exception as e:
                            print(f"Error: {e}")
                            pass

                        # Update or post annotation
                        new_ann = dict(
                            annotation_set_id=annotation_set_id,
                            label_id=cand.label_id,
                            likelihood=cand.score,
                            point_id=point_id,
                        )
                        if annotation_set_id == base_annotation_set_id:
                            # update the annotation on the point
                            a = point_obj['annotations'][0]
                            self.sqapi.patch(f"/api/annotation/{a['id']}", json_data=new_ann).execute()
                            counts['annotations'] += 1
                        else:
                            self.sqapi.post("/api/annotation", json_data=new_ann).execute()
                            counts['new_annotations'] += 1
            else:
                # CREATE NEW ANNOTATION
                p = create_annotation_polygon_with_label_id(
                    cand.label_id, cand.poly, likelihood=cand.score,
                    width=mediaobj.width, height=mediaobj.height
                )
                new_points_to_return.append(p)

        return counts, new_points_to_return

    @staticmethod
    def _append_exemplar_geometry(bboxes, points, polygons, x, y, polygon):
        """Append one exemplar; use polygon point sampling when polygon is not a bbox."""
        points.append([x, y])
        if polygon_is_bbox(polygon):
            bboxes.append(bbox_of_vertices(polygon, fmt="xywh"))
            polygons.append(None)
        else:
            bboxes.append(None)
            polygons.append(polygon)

    def load_exemplar_annotations(self, annotation_set_id, label_ids_to_ignore=[6835,636,458]):
        df = load_squidle_annotations_to_df(self.sqapi, annotation_set_id, label_ids_to_ignore)
        if df.empty:
            raise ValueError("Error: The loaded dataframe is empty. No available exemplar annotations. Does the squidle user have access to this annotation set?")

        exemplars_by_label_id = defaultdict(dict)

        for label_id, df_label in df.groupby('label.id', sort=False, dropna=False):
            # Do something once per label
            if label_id not in label_ids_to_ignore:
                print(f"\nLabel: {label_id} | Subrows: {len(df_label)}")
                for media_path, annotations in df_label.groupby('point.media.path_best', sort=False, dropna=False):
                    bboxes, points, polygons = [], [], []
                    for _, a in annotations.iterrows():
                        x = a['point.x']
                        y = a['point.y']
                        polygon = ast.literal_eval(a['point.polygon'])
                        polygon = [[(p[0] + x), (p[1] + y)] for p in polygon]
                        self._append_exemplar_geometry(bboxes, points, polygons, x, y, polygon)
                    exemplars_by_label_id[label_id][media_path] = [bboxes, points, [], [], polygons, []]

        # Add negative exemplars
        if self.n_negative > 0:
            for label_id, label_data in exemplars_by_label_id.items():
                for media_path, media_data in label_data.items():
                    neg_data = df.loc[(df["label.id"] != label_id) & (df['point.media.path_best'] == media_path)]
                    if len(neg_data) > 0:
                        max_rows = min(self.n_negative, len(neg_data))
                        neg_data = neg_data.sample(n=max_rows, random_state=42)
                        neg_bboxes, neg_points, neg_polygons = [], [], []
                        for _, a in neg_data.iterrows():
                            x = a['point.x']
                            y = a['point.y']
                            polygon = ast.literal_eval(a['point.polygon'])
                            polygon = [[(p[0] + x), (p[1] + y)] for p in polygon]
                            self._append_exemplar_geometry(
                                neg_bboxes, neg_points, neg_polygons, x, y, polygon
                            )
                        media_data[2] += neg_bboxes
                        media_data[3] += neg_points
                        if len(media_data) < 6:
                            media_data.append([])
                        media_data[5] += neg_polygons

        # Sort to favour images with negative examples and more positive examples
        for label_id, inner_dict in exemplars_by_label_id.items():
            # 1. Sort the items (key-value pairs) of the inner dictionary
            # x[1] is the value (the list of 4 lists)
            # x[1][2] is the list at position 2; x[1][0] is the list at position 0
            sorted_items = sorted(
                inner_dict.items(),
                key=lambda x: (len(x[1][2]) >= self.n_negative, len(x[1][0])),
                reverse=True
            )
            exemplars_by_label_id[label_id] = dict(sorted_items)

        return exemplars_by_label_id




