

def compute_center_obj(bbox):
    """
    Computes center point of bbox
    :param bbox: bbox in format [y1,x1,y2,x2]
    :return: avg_x, avg_y
    """
    y1, x1, y2, x2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def compute_center_ocr(bbox):
    """
    Computes center point of ocr reading
    :param bbox: bbox in format [x1, y1, x2, y2, x3, y3, x4, y4] (starts from top left order is clockwise)
    :return: avg_x, avg_y
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    center_x = (x1 + x3) / 2
    center_y = (y1 + y3) / 2
    return center_x, center_y


def euclidean_distance(center1, center2):
    """
    Calculates distance between [x1,y1] and [x2,y2]
    :param center1: center coordinates of first point
    :param center2: center coordinates of second point
    :return: distance
    """
    return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5


def sorting_key(entry, img_height):
    """
    Sorts dimensions according to the distance between bottom left coordinates of image and their central coordinate
    :param entry: List containing spacing info
    :param img_height: Height of image in pixels
    :return: sorted input
    """
    # Extract object detection bounding box
    y1, x1, y2, x2 = entry[2]
    avg_x = (x1 + x2) / 2
    avg_y = (y1 + y2) / 2
    # Calculate the Euclidean distance from the point (0, img_height)
    distance = euclidean_distance([avg_x, avg_y], [0, img_height])
    return distance






