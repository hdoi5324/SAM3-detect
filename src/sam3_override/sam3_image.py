import torch
from sam3.model.sam3_image import Sam3Image as _Sam3Image


class Sam3Image(_Sam3Image):
    @torch.inference_mode()
    def get_prompt_embeddings(
            self,
            backbone_out,
            find_input,
            geometric_prompt,
            encode_text=True, #todo: set to false?
            use_img_pos_embed=True,
    ):
        # Encoding part of Sam3Image.forward_grounding
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            #prompt, prompt_mask, backbone_out = self._encode_prompt(
            #    backbone_out, find_input, geometric_prompt.clone()
            prompt, prompt_mask, backbone_out, txt_feats, txt_masks, geo_feats, geo_masks, visual_prompt_embed, visual_prompt_mask = self._encode_prompt(
                backbone_out, find_input, geometric_prompt.clone(),
                encode_text=encode_text,
                use_img_pos_embed=use_img_pos_embed,
            )
        return prompt, prompt_mask, backbone_out, txt_feats, txt_masks, geo_feats, geo_masks, visual_prompt_embed, visual_prompt_mask

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
        prev_mask_pred=None,
        use_img_pos_embed=True,
    ):
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats` is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Encode geometry
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds if use_img_pos_embed else None, 
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out, txt_feats, txt_masks, geo_feats, geo_masks, visual_prompt_embed, visual_prompt_mask

    @torch.inference_mode()
    def forward_with_prompt_encoding(self,
            backbone_out,
            find_input,
            prompt, prompt_mask,
            find_target=None,
        ):
        # Everything after encoding part of Sam3Image.forward_grounding
            # Run the encoder
            with torch.profiler.record_function("SAM3Image._run_encoder"):
                backbone_out, encoder_out, _ = self._run_encoder(
                    backbone_out, find_input, prompt, prompt_mask
                )
            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": backbone_out,
                },
            }

            # Run the decoder
            with torch.profiler.record_function("SAM3Image._run_decoder"):
                out, hs = self._run_decoder(
                    memory=out["encoder_hidden_states"],
                    pos_embed=encoder_out["pos_embed"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    encoder_out=encoder_out,
                )

            # Run segmentation heads
            with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
                self._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hs,
                )

            if self.training or self.num_interactive_steps_val > 0:
                self._compute_matching(out, self.back_convert(find_target))
            return out