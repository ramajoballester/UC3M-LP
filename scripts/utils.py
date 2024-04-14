from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def poly2bbox(poly_coord):
    x_coords = [coord[0] for coord in poly_coord]
    y_coords = [coord[1] for coord in poly_coord]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    return [[x_min, y_min], [x_max, y_max]]


def create_jsonlp(lp_id, lp_bbox, ocr_data):
    return {
        'lp_id': lp_id,
        'bbox': lp_bbox,
        'ocr': ocr_data
    }

def is_point_inside_polygon(points, poly):
    poly = Polygon(poly)

    # get the centroid of the rectangle points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    centroid = (sum(x_coords) / len(points), sum(y_coords) / len(points))
    inside = poly.contains(Point(centroid))
    
    return inside