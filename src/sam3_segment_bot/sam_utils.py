from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from shapely import Polygon
from skimage import measure


def check_polygon(list_of_xy, width, height, min_percent=0.001):
    if not(isinstance(list_of_xy, list) and len(list_of_xy) > 2):
        return False
    min_area = width * height * min_percent
    return is_area_greater_than(list_of_xy, min_area)


def clean_polygon(list_of_xy, width, height):
    checked_points = []
    for x, y in list_of_xy:
        if x <=0:
            x = 0
        if x >= width:
            x = width - 1
        if y <=0:
            y = 0
        if y >= height:
            y = height - 1
        checked_points.append((x, y))
    return checked_points


def group_by_label_id(points):
    grouped = defaultdict(list)
    for point in points:
        for ann in point['annotations']:
            label_id = ann.get("label_id")
            grouped[label_id].append(point)
    return dict(grouped)


def points_ex_label_id(points, label_id, with_xy=True):
    ex_points = []
    for point in points:
        point_has_label_id = False
        for ann in point['annotations']:
            if (ann.get("label_id") is not None) and (ann.get("label_id") == label_id):
                point_has_label_id = True
        if not point_has_label_id:
            if (with_xy and point.get("x") is not None and point.get("y") is not None) or not with_xy:
                ex_points.append(point)
    return ex_points


def point_in_polygon(point, polygon):
    """
    Return True if `point` is inside `polygon` OR on its boundary.
    polygon: list of (x, y) vertices; can be open or closed.
    """
    x, y = point
    if len(polygon) < 3:
        return False

    # Use open polygon (remove repeated last vertex if present)
    if polygon[0] == polygon[-1]:
        poly = polygon[:-1]
    else:
        poly = polygon

    # --- 1) Boundary check (point on any segment) ---
    def on_segment(p1, p2, q, eps=1e-12):
        (x1, y1), (x2, y2) = p1, p2
        xq, yq = q
        # Collinearity via cross product == 0
        if abs((x2 - x1) * (yq - y1) - (y2 - y1) * (xq - x1)) > eps:
            return False
        # Within bounding box
        return (min(x1, x2) - eps <= xq <= max(x1, x2) + eps and
                min(y1, y2) - eps <= yq <= max(y1, y2) + eps)

    for i in range(len(poly)):
        if on_segment(poly[i], poly[(i + 1) % len(poly)], (x, y)):
            return True  # on border counts as inside

    # --- 2) Ray casting (strict interior) ---
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Ensure y1 <= y2 to keep logic tidy
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        # Half-open interval to avoid double-counting vertices
        if (y > y1) and (y <= y2):
            # Avoid division by zero; horizontal edges are skipped by condition above
            xinters = x1 + (x2 - x1) * ((y - y1) / (y2 - y1)) if y2 != y1 else x1
            if xinters > x:
                inside = not inside

    return inside


def create_annotation_label(label_id, likelihood=0.0, tag_names=None, comment=None, needs_review=False):
    return dict(
        label_id=label_id,
        likelihood=float(likelihood),
        tag_names=tag_names,
        comment=comment,
        needs_review=needs_review
    )


def create_annotation_polygon_with_label_id(label_id, polygon, likelihood=0.0, tag_names=None, comment=None, needs_review=False,
                                     row=None, col=None, width=None, height=None, t=None):
    #if row is None and col is None:
    #    col, row = interior_midpoint_shapely(polygon)
    return dict(
        pixels=dict(row=row, col=col, width=width, height=height, polygon=polygon),
        annotation_label=create_annotation_label(
            label_id, likelihood=likelihood, tag_names=tag_names, comment=comment, needs_review=needs_review
        ),
        t=t
    )


def polygon_is_bbox(coordinates):
    if not isinstance(coordinates, list):
        return False
    if len(coordinates) not in [4, 5] or not all((isinstance(i, float) for i in p) for p in coordinates):
        return False  # The LineString must be closed with 4 corners

    # Check
    ind = 0 if coordinates[0][0] == coordinates[1][0] else 1
    matches = []
    for i in range(0, 4):
        matches.append(coordinates[i][ind] == coordinates[(i + 1) % 4][ind])
        ind = 0 if ind == 1 else 1

    return all(matches)


def mask_to_polygon(mask, tolerance=2.0):
    # Ensure mask is uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8)

    # findContours is highly optimized C++
    # RETR_EXTERNAL ignores holes; CHAIN_APPROX_SIMPLE removes redundant points on lines
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Initial simplification from findContours might be enough,
        # but we use approxPolyDP for specific tolerance control
        poly = cv2.approxPolyDP(contour, epsilon=tolerance, closed=True)
        points = poly.reshape(-1, 2).tolist()
        if len(points) >= 3:
            polygons.append(points)

    return polygons


def downscale_image(img, scaling):
    """
    Downscale a PIL image by a factor (e.g., 0.5 for half-resolution).

    Parameters
    ----------
    img : PIL.Image
        Input image of size (H, W, 3)
    scaling : float
        Factor to reduce resolution, e.g. 0.5

    Returns
    -------
    PIL.Image
        Downscaled image
    """
    # Get original dimensions
    w, h = img.size  # PIL uses (width, height)

    # Compute new dimensions
    new_w = int(w * scaling)
    new_h = int(h * scaling)

    # Resize with high-quality downsampling
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def downscale_ndarray(img, scaling):
    """
    Downscale a NumPy image array by a factor (e.g., 0.5).

    img: ndarray of shape (H, W, C)
    scaling: float scaling factor
    """
    h, w = img.shape[:2]
    new_w = int(w * scaling)
    new_h = int(h * scaling)

    # OpenCV expects width first
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def downscale_points(points, scaling):
    """
    points: list of (x, y)
    scaling: scaling factor (e.g., 0.5)
    """
    if scaling == 1.0:
        return points
    return [(x * scaling, y * scaling) for x, y in points]


def interior_midpoint_shapely(polygon_xy):
    """
    polygon_xy: list of (x, y) vertices; open or closed.
    Returns a point (x, y) guaranteed to lie inside the polygon.
    """
    poly = Polygon(polygon_xy)
    p = poly.representative_point()   # guaranteed interior point
    return (int(p.x), int(p.y))


def is_area_greater_than(coords, min_area):
    # Shapely accepts open or closed rings
    poly = Polygon(coords)
    return poly.area > float(min_area)

def bbox_of_vertices(vertices,
                      fmt: str = "xywh"):
    """
    Compute bbox from vertices in relative coords.
    fmt: 'xyxy' or 'xywh'
    """
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x1, y1 = float(np.min(xs)), float(np.min(ys))
    x2, y2 = float(np.max(xs)), float(np.max(ys))
    if fmt.lower() == "xyxy":
        return x1, y1, x2, y2
    elif fmt.lower() == "xywh":
        return x1, y1, (x2 - x1), (y2 - y1)
    else:
        raise ValueError("fmt must be 'xyxy' or 'xywh'")