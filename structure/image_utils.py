import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
def iterative_erosion(roi, max_iterations=10, threshold=1):
    """
    Erodes image until remaining pixels drops lower than threshold
    :param roi: region of interest on the image
    :param max_iterations: self-explanatory
    :param threshold: threshold limit
    :return: returns binary image
    """
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    prev_count = np.sum(binary == 255)

    for _ in range(max_iterations):
        eroded = cv.erode(binary, kernel)

        # Count the non-zero pixels in the eroded image
        current_count = np.sum(eroded == 255)

        if current_count <= threshold or abs(current_count - prev_count) < 10:  # Change threshold as per requirement
            break

        binary = eroded
        prev_count = current_count

    return binary


def pl_direction(pl_instances, img):
    """
    Detects direction of pl by eroding arrows, eroding it until some threshold, only thickest part (arrowhead) will
    remain, it will show the direction
    :param pl_instances: list of pl instances
    :param img: image
    :return: None (directions appended instances inside)
    """
    for pl in pl_instances:
        y1, x1, y2, x2 = pl.obj_det_bbox
        ocr_bbox = pl.text_bbox

        if pl.text_bbox:
            ry1 = int((ocr_bbox[1] + ocr_bbox[3]) / 2)
            ry2 = int((ocr_bbox[5] + ocr_bbox[7]) / 2)
            rx1 = int((ocr_bbox[0] + ocr_bbox[6]) / 2)
            rx2 = int((ocr_bbox[2] + ocr_bbox[4]) / 2)

            overlap_x1 = max(x1, rx1)
            overlap_y1 = max(y1, ry1)
            overlap_x2 = min(x2, rx2)
            overlap_y2 = min(y2, ry2)

            # If overlap exists, paint it white
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                img[overlap_y1:overlap_y2, overlap_x1:overlap_x2] = 255

        roi = img[y1:y2, x1:x2]
        binary_roi = iterative_erosion(roi, max_iterations=1)

        # Find the non-zero pixels' indices
        y_indices, x_indices = np.where(binary_roi == 255)

        # Calculate the average x and y coordinates
        avg_x = np.mean(x_indices)
        avg_y = np.mean(y_indices)

        # Calculate midpoints of x and y axes
        midpoint_x = binary_roi.shape[1] / 2
        midpoint_y = binary_roi.shape[0] / 2

        # Determine direction
        if avg_y > midpoint_y and abs(avg_x - midpoint_x) < midpoint_x / 2:
            direction = "downward"
        elif avg_y < midpoint_y and abs(avg_x - midpoint_x) < midpoint_x / 2:
            direction = "upward"
        elif avg_x > midpoint_x and abs(avg_y - midpoint_y) < midpoint_y / 2:
            direction = "right"
        elif avg_x < midpoint_x and abs(avg_y - midpoint_y) < midpoint_y / 2:
            direction = "left"
        else:
            direction = "unknown"

        pl.direction = direction

    return None


def iterative_erosion_dl(roi, initial_iterations=1, max_iterations=10, threshold=1):
    """
    Iterative erosion for distributed load
    :param roi: region of interest of image
    :param initial_iterations: min number of iterations completed
    :param max_iterations: max allowable iteration
    :param threshold: minimum number of remaining pixels acceptable
    :return: binary image (unpadded)
    """
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    ret, binary = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY_INV)

    # Padding to the image
    pad_size = 2  # Choose a suitable padding size depending on the kernel size
    binary = cv.copyMakeBorder(binary, pad_size, pad_size, pad_size, pad_size, cv.BORDER_CONSTANT, value=0)

    kernel = np.ones((3, 3), np.uint8)



    for i in range(initial_iterations + max_iterations):
        prev_binary = binary.copy()
        binary = cv.erode(binary, kernel)

        current_count = np.sum(binary == 255)
        if current_count <= threshold:
            binary = prev_binary  # Use the binary image from the previous iteration
            break

    return binary[pad_size:-pad_size, pad_size:-pad_size]  # Return the image without the added padding


def determine_direction(x_count, y_count):
    """
    Counts pixels in x and y directions, decides direction of arrow
    :param x_count: number of remained pixels after erosion in x direction
    :param y_count: number of remained pixels after erosion in y direction
    :return: direction
    """
    # Find max values and their indices for x_count and y_count
    max_x, idx_x = max((val, idx) for (idx, val) in enumerate(x_count))
    max_y, idx_y = max((val, idx) for (idx, val) in enumerate(y_count))

    # Determine if the arrow points horizontally or vertically
    if max_x >= max_y:
        if idx_x < len(x_count) / 2:
            return "left"
        else:
            return "right"
    else:
        if idx_y < len(y_count) / 2:
            return "upward"
        else:
            return "downward"

def dl_direction(dl_boxes, img, initial_iterations=1, max_iterations=10, threshold=1):
    """
    Main function of deciding direction of disributed load
    :param dl_boxes: bboxes obtained from object detection
    :param img: current image
    :param initial_iterations: min number of iterations required before deciding direction
    :param max_iterations: number of max allowable iterations
    :param threshold: min number of remaining pixels, below this threshold value iteration stops
    :return: list containing information of dl instances (in this function direction attribute assigned)
    """
    dl_with_direction = []

    for dl in dl_boxes:
        sensitivity = 5
        y1, x1, y2, x2 = dl[:4]
        roi = img[y1 + sensitivity:y2 - sensitivity, x1 + sensitivity:x2 - sensitivity]

        eroded_img = iterative_erosion_dl(roi, initial_iterations, max_iterations, threshold)

        x_count = np.sum(eroded_img == 255, axis=0)
        y_count = np.sum(eroded_img == 255, axis=1)

        if np.max(x_count) >= np.max(y_count):
            direction = "horizontal"
            dl.append(direction)
            pointing_direction = determine_direction(x_count, y_count)
            dl.append(pointing_direction)
        else:
            direction = "vertical"
            dl.append(direction)
            pointing_direction = determine_direction(x_count, y_count)
            dl.append(pointing_direction)

        dl_with_direction.append(dl)

    return dl_with_direction

def find_max_values(node_instances):
    """
    Max values of nodes obtained
    :param node_instances: list containing nodes
    :return: max values of nodes in each direction
    """
    max_shape_x = max(node.shape_x for node in node_instances)
    max_shape_y = max(node.shape_y for node in node_instances)
    return max_shape_x, max_shape_y

def calculate_node_pixel_values(max_shape_x, max_shape_y,img_width, img_height):
    """
    Calculates ratio of pixels to the node values
    :param max_shape_x: x directional max shape value
    :param max_shape_y: y directional max shape value
    :param img_width: width of image
    :param img_height: height of image
    :return: in x and y directions, how many pixels are corresponding to the shape values
    """
    if max_shape_x == 0:
        node_value_pixel_x = 0
    else:
        node_value_pixel_x = int(img_width / max_shape_x)
    if max_shape_y == 0:
        node_value_pixel_y = 0
    else:
        node_value_pixel_y = int(img_height / max_shape_y)
    return node_value_pixel_x, node_value_pixel_y


def arrange_drawing_area(shape_x, shape_y, padding, node_value_pixel_x, node_value_pixel_y):
    """
    Calculates resolution of created image
    :param shape_x: max shape x value
    :param shape_y: max shape y value
    :param padding: how many pixels added for blank corners
    :param node_value_pixel_x: ratio in x direction
    :param node_value_pixel_y: ratio in y direction
    :return: created image resolution
    """
    image_width = padding + node_value_pixel_x * int(shape_x)
    image_height = padding + node_value_pixel_y * int(shape_y)
    return image_width, image_height


def generate_image(width, height):
    """
    Generates blank-white image
    :param width: wanted width
    :param height: wanted height
    :return: instance of image
    """
    generated_image = Image.new('RGBA', (width, height), (255, 255, 255))
    return generated_image
