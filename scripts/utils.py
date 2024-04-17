import os
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


def create_txt_file(input_directory):
    # Read files in lp_directory
    train_files = os.listdir(os.path.join(input_directory, 'images', 'train'))
    val_files = os.listdir(os.path.join(input_directory, 'images', 'val'))
    train_files.sort()
    val_files.sort()

    # Save train and val files in train.txt and val.txt
    with open(os.path.join(input_directory, 'train.txt'), 'w') as f:
        for filename in train_files:
            f.write(filename.split('.')[0] + '\n')
    
    with open(os.path.join(input_directory, 'val.txt'), 'w') as f:
        for filename in val_files:
            f.write(filename.split('.')[0] + '\n')

def is_point_inside_polygon(points, poly):
    poly = Polygon(poly)

    # get the centroid of the rectangle points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    centroid = (sum(x_coords) / len(points), sum(y_coords) / len(points))
    inside = poly.contains(Point(centroid))
    
    return inside