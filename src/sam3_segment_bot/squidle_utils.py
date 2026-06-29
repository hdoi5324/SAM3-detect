import io

import pandas as pd
from pandas import DataFrame
from sqapi import SQMediaObject

from sam3_override.utils import numpy_bgr_to_pil_rgb

EXPORT_COLUMNS = [
    # https://squidle.org/api/help?template=api_help_page.html#annotation
    "id", "tag_names", "comment", "label.id", "label.name", "label.uuid", "annotation_set_id",
    "updated_at", "user.username", "needs_review", "likelihood", "point.is_targeted", "point.x", "point.y",
    "point.media.path_best", "annotation_set_id", "point.polygon",
]


def load_image(path_or_url):
    sq_media = SQMediaObject(path_or_url)
    return numpy_bgr_to_pil_rgb(sq_media.data())


def _export_annotations_to_df(
    sqapi,
    *,
    filters: list[dict],
    include_columns: list[str],
) -> DataFrame:
    fileops = [
        dict(module="pandas", method="json_normalize"),
        dict(method="sort_index", kwargs=dict(axis=1)),
    ]
    request = sqapi.export(
        "api/annotation/export",
        include_columns=include_columns,
        filters=filters,
        fileops=fileops,
        qsparams=dict(template="dataframe.csv", disposition="attachment"),
    )
    result = request.execute()
    decoded_content = result.content.decode("utf-8")
    return pd.read_csv(io.StringIO(decoded_content))


def load_squidle_annotations_to_df(sqapi, annotation_set_id, label_ids_to_ignore: list[int] = []) -> DataFrame:
    """Load polygon exemplar annotations for SAM3 segmentation."""
    filters = [
        dict(name="annotation_set_id", op="eq", val=annotation_set_id),
        dict(name="point__has_polygon", op="has", val=True),
        dict(name="label_id", op="not_in", val=label_ids_to_ignore),
    ]
    return _export_annotations_to_df(sqapi, filters=filters, include_columns=EXPORT_COLUMNS)
