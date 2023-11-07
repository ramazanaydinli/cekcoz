import cv2 as cv
import numpy as np
from structure import calculation_utils
from structure.structure_classes import Node, Dimension, FixSupport, PinSupport, RollerSupport, Frame, PointLoad,\
    DistributedLoad, TriangularDistributedLoad
import itertools


def dimension_direction(sorted_spacings, img):
    """
    Decides direction of the dimension using pixel values
    :param sorted_spacings: list of spacings
    :param img: input image
    :return: adds direction to the input list and returns it back
    """
    sorted_spacing_with_direction = []
    for spacing_info in sorted_spacings:
        y1, x1, y2, x2 = spacing_info[2]
        roi = img[y1:y2, x1:x2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # Threshold the image
        _, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)  # Adjust the threshold value as needed

        # Sum along x and y axes
        sum_x = np.sum(thresh, axis=0)
        sum_y = np.sum(thresh, axis=1)

        # Check where the predominant sum is
        x_peak = np.max(sum_x)
        y_peak = np.max(sum_y)

        # Decide the direction based on the peaks
        if x_peak > 1.5 * y_peak:  # Adjust the factor as needed
            direction = "vertical"
        elif y_peak > 1.5 * x_peak:
            direction = "horizontal"
        else:
            direction = "angled"

        spacing_info.append(direction)

        # Append the updated spacing_info to the global list
        sorted_spacing_with_direction.append(spacing_info)

    return sorted_spacing_with_direction


def find_closest_node_of_previous_spacing(spacing, node1, node2):
    """
    Before placing nodes in a logical order, calculating distances between nodes to find which node is common between
    adjacent spacings
    :param spacing: Dimension instance
    :param node1: first node of the spacing
    :param node2: second node of the spacing
    :return: between 4 input nodes, sends back two which has the minimum distance between them
    """
    sn1 = [spacing.node_1.bbox_x, spacing.node_1.bbox_y]
    sn2 = [spacing.node_2.bbox_x, spacing.node_2.bbox_y]
    n1 = [node1.bbox_x, node1.bbox_y]
    n2 = [node2.bbox_x, node2.bbox_y]

    # Find distances between nodes
    dist_sn1_n1 = calculation_utils.euclidean_distance(sn1, n1)
    dist_sn1_n2 = calculation_utils.euclidean_distance(sn1, n2)
    dist_sn2_n1 = calculation_utils.euclidean_distance(sn2, n1)
    dist_sn2_n2 = calculation_utils.euclidean_distance(sn2, n2)

    # Determine the node (either node1 or node2) that is closest to any of the spacing nodes
    dist_list = [dist_sn1_n1, dist_sn1_n2, dist_sn2_n1, dist_sn2_n2]
    min_dist = min(dist_list)

    if min_dist == dist_sn1_n1:
        return spacing.node_1, node1
    elif min_dist == dist_sn1_n2:
        return spacing.node_1, node2
    elif min_dist == dist_sn2_n1:
        return spacing.node_2, node1
    else:
        return spacing.node_2, node2


def initialize_parameters(sorted_spacings_with_directions):
    """
    Initializes base classes for question
    :param sorted_spacings_with_directions: list containing spacing information
    :return: created instances of nodes and spacings
    """
    spacing_instances = []
    node_instances = []
    shape_x_start = 0
    shape_y_start = 0
    for spacing_info in sorted_spacings_with_directions:
        if not spacing_instances:
            if spacing_info[3] == "horizontal":
                y1, x1, y2, x2 = spacing_info[2]
                distance = spacing_info[0][0]
                unit = spacing_info[0][1]
                obj_det_bbox = spacing_info[2]
                text_bbox = spacing_info[1]
                bbox_x = x1
                bbox_y = int((y1 + y2) / 2)
                node_1 = Node(bbox_x, bbox_y, shape_x_start, shape_y_start)
                node_instances.append(node_1)
                bbox_x = x2
                bbox_y = int((y1 + y2) / 2)
                node_2 = Node(bbox_x, bbox_y, shape_x_start + distance, shape_y_start)
                node_instances.append(node_2)
                spacing = Dimension(node_1, node_2, distance, unit, obj_det_bbox, text_bbox)
                spacing_instances.append(spacing)
            elif spacing_info[3] == "vertical":
                y1, x1, y2, x2 = spacing_info[2]
                distance = spacing_info[0][0]
                unit = spacing_info[0][1]
                obj_det_bbox = spacing_info[2]
                text_bbox = spacing_info[1]
                bbox_x = int(x1 + x2) / 2
                bbox_y = y1
                node_1 = Node(bbox_x, bbox_y, shape_x_start, shape_y_start)
                node_instances.append(node_1)
                bbox_x = int(x1 + x2) / 2
                bbox_y = y2
                node_2 = Node(bbox_x, bbox_y, shape_x_start, shape_y_start + distance)
                node_instances.append(node_2)
                spacing = Dimension(node_1, node_2, distance, unit, obj_det_bbox, text_bbox)
                spacing_instances.append(spacing)
            else:
                # code the nodes for angled dimensioning
                pass
        else:
            if spacing_info[3] == "horizontal":
                y1, x1, y2, x2 = spacing_info[2]
                distance = spacing_info[0][0]
                unit = spacing_info[0][1]
                obj_det_bbox = spacing_info[2]
                text_bbox = spacing_info[1]
                bbox_x = x1
                bbox_y = int(y1 + y2) / 2
                node_1 = Node(bbox_x=bbox_x, bbox_y=bbox_y)
                bbox_x = x2
                bbox_y = int(y1 + y2) / 2
                node_2 = Node(bbox_x=bbox_x, bbox_y=bbox_y)
                # psc_node = previous spacing closest node
                # cc_node = current closest node
                psc_node, cc_node = find_closest_node_of_previous_spacing(spacing_instances[-1], node_1, node_2)
                if cc_node == node_1:
                    node_2.shape_x = psc_node.shape_x + distance
                    node_2.shape_y = psc_node.shape_y
                    node_instances.append(node_2)
                    spacing = Dimension(psc_node, node_2, distance, unit, obj_det_bbox, text_bbox)
                    spacing_instances.append(spacing)
                else:
                    node_1.shape_x = psc_node.shape_x - distance
                    node_1.shape_y = psc_node.shape_y
                    node_instances.append(node_1)
                    spacing = Dimension(node_1, psc_node, distance, unit, obj_det_bbox, text_bbox)
                    spacing_instances.append(spacing)
            elif spacing_info[3] == "vertical":
                y1, x1, y2, x2 = spacing_info[2]
                distance = spacing_info[0][0]
                unit = spacing_info[0][1]
                obj_det_bbox = spacing_info[2]
                text_bbox = spacing_info[1]
                bbox_x = int(x1 + x2) / 2
                bbox_y = y1
                node_1 = Node(bbox_x=bbox_x, bbox_y=bbox_y)
                bbox_x = int(x1 + x2) / 2
                bbox_y = y2
                node_2 = Node(bbox_x=bbox_x, bbox_y=bbox_y)
                # psc_node = previous spacing closest node
                # cc_node = current closest node
                psc_node, cc_node = find_closest_node_of_previous_spacing(spacing_instances[-1], node_1, node_2)
                if cc_node == node_1:
                    node_2.shape_x = psc_node.shape_x
                    node_2.shape_y = psc_node.shape_y - distance
                    node_instances.append(node_2)
                    spacing = Dimension(psc_node, node_2, distance, unit, obj_det_bbox, text_bbox)
                    spacing_instances.append(spacing)
                else:
                    node_1.shape_x = psc_node.shape_x
                    node_1.shape_y = psc_node.shape_y + distance
                    node_instances.append(node_1)
                    spacing = Dimension(node_1, psc_node, distance, unit, obj_det_bbox, text_bbox)
                    spacing_instances.append(spacing)
            else:
                pass
    return spacing_instances, node_instances


def correct_shapes(node_list):
    """
    Nodes with negative values are corrected in this function
    :param node_list: list of nodes
    :return: corrected list of nodes
    """
    # Find the minimum shape_x and shape_y values
    min_shape_x = min(node.shape_x for node in node_list)
    min_shape_y = min(node.shape_y for node in node_list)

    # Calculate the offsets required to make the smallest shape values zero
    offset_x = abs(min_shape_x) if min_shape_x < 0 else 0
    offset_y = abs(min_shape_y) if min_shape_y < 0 else 0

    # Apply the offsets to all node instances
    for node in node_list:
        node.shape_x += offset_x
        node.shape_y += offset_y

    return node_list


def find_and_append_missing_nodes(node_instances):
    """
    Creates missing nodes according to every definable point
    :param node_instances: Raw (negative value corrected) node points
    :return: New list (if any new node created in the process)
    """
    # Extract unique shape_x and shape_y values
    unique_shape_x = {node.shape_x for node in node_instances}
    unique_shape_y = {node.shape_y for node in node_instances}

    # Generate all possible combinations based on the unique values
    shape_combinations = set(itertools.product(unique_shape_x, unique_shape_y))

    # Identify existing combinations from node_instances
    existing_shapes = {(node.shape_x, node.shape_y) for node in node_instances}

    # Identify missing combinations
    missing_shapes = shape_combinations - existing_shapes

    # For each missing shape combination, create and append a new node to node_instances
    for shape_x, shape_y in missing_shapes:
        node_for_shape_x = next((node for node in node_instances if node.shape_x == shape_x), None)
        node_for_shape_y = next((node for node in node_instances if node.shape_y == shape_y), None)

        bbox_x = node_for_shape_x.bbox_x if node_for_shape_x else None
        bbox_y = node_for_shape_y.bbox_y if node_for_shape_y else None

        missing_node = Node(bbox_x=bbox_x, bbox_y=bbox_y, shape_x=shape_x, shape_y=shape_y)
        node_instances.append(missing_node)

    return node_instances


def create_support_instances(support_bbox, node_instances):
    """
    Creating support instances according to the obj det support instances
    :param support_bbox: bbox coordinates and class information of supports
    :param node_instances: node information
    :return: created support classes which also has their own nodes appended
    """
    fix_support_instances = []
    pin_support_instances = []
    roller_support_instances = []
    for bbox in support_bbox:
        center_x = (bbox[1] + bbox[3]) / 2
        center_y = (bbox[0] + bbox[2]) / 2
        closest_node = min(node_instances, key=lambda node: calculation_utils.
                           euclidean_distance([center_x, center_y], [node.bbox_x, node.bbox_y]))
        if bbox[-1] == 'fix_support':
            fix_support_instances.append(FixSupport(node=closest_node))
        elif bbox[-1] == 'pin_support':
            pin_support_instances.append(PinSupport(node=closest_node))
        elif bbox[-1] == 'roller_support':
            roller_support_instances.append(RollerSupport(node=closest_node))
        else:
            pass
    return fix_support_instances, pin_support_instances, roller_support_instances


def find_frame_direction(frame_boxes, img):
    """
    Finds and appends direction of the frame
    :param frame_boxes: bbox info of frame
    :param img: input image
    :return: direction appended bbox list
    """
    for frame_info in frame_boxes:
        y1,x1,y2,x2 = frame_info[:4]
        roi = img[y1:y2, x1:x2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
         # Threshold the image
        _, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)  # Adjust the threshold value as needed

        # Sum along x and y axes
        sum_x = np.sum(thresh, axis=0)
        sum_y = np.sum(thresh, axis=1)

        # Check where the predominant sum is
        x_peak = np.max(sum_x)
        y_peak = np.max(sum_y)

        # Decide the direction based on the peaks
        if x_peak > 1.5 * y_peak:  # Adjust the factor as needed
            direction = "vertical"
        elif y_peak > 1.5 * x_peak:
            direction = "horizontal"
        else:
            direction = "angled"
        frame_info.append(direction)
    return frame_boxes


def find_closest_node(x, y, nodes):
    """
    Self explained
    :param x: x coordinate
    :param y: y coordinate
    :param nodes: node instances
    :return:
    """
    def node_distance(node):
        """
        Distance between coordinates and nodes
        :param node: single node instance
        :return: distance between node bbox center and given x-y coordinates
        """
        return calculation_utils.euclidean_distance([x, y], [node.bbox_x, node.bbox_y])
    return min(nodes, key=node_distance)


def initialize_frames(frame_boxes, node_instances):
    """
    Creates instances of frames
    :param frame_boxes: bbox list of frames
    :param node_instances: list of nodes
    :return: list of frame instances
    """
    frame_instances = []
    for box in frame_boxes:
        if box[5] == "horizontal":
            x1 = box[1]
            x2 = box[3]
            y = int((box[0] + box[2]) / 2)

            # Get the closest node to x1 and x2
            c_node1 = find_closest_node(x1, y, node_instances)
            c_node2 = find_closest_node(x2, y, node_instances)

            frame_instance = Frame(node1=c_node1, node2=c_node2)
            frame_instances.append(frame_instance)
        elif box[5] == "vertical":
            y1 = box[0]
            y2 = box[2]
            x = int((box[1] + box[3]) / 2)
            # Get the closest node to x1 and x2
            c_node1 = find_closest_node(x, y1, node_instances)
            c_node2 = find_closest_node(x, y2, node_instances)

            frame_instance = Frame(node1=c_node1, node2=c_node2)
            frame_instances.append(frame_instance)
        else:
            # Need to fill this later
            pass
    return frame_instances


def divide_frames(frame, node_instances):
    """
    Checks current frame instance and if frame passes through any nodes except the ones it is defined, it divides frame
    :param frame: single frame instances
    :param node_instances: list of nodes
    :return: list of new frames
    """
    # Check if it's a horizontal frame (direction is along x-axis) or vertical (direction is along y-axis)
    is_horizontal = frame.node_1.shape_y == frame.node_2.shape_y

    new_frames = []

    # Extract the node list that lies between the nodes of this frame
    if is_horizontal:
        intermediate_nodes = sorted([node for node in node_instances
                                     if frame.node_1.shape_y == node.shape_y
                                     and frame.node_1.shape_x < node.shape_x < frame.node_2.shape_x],
                                    key=lambda n: n.shape_x)
    else:
        intermediate_nodes = sorted([node for node in node_instances
                                     if frame.node_1.shape_x == node.shape_x
                                     and frame.node_1.shape_y < node.shape_y < frame.node_2.shape_y],
                                    key=lambda n: n.shape_y)

    # If there are no intermediate nodes, just return the original frame
    if not intermediate_nodes:
        return [frame]

    # Create new frames based on intermediate nodes
    start_node = frame.node_1
    for node in intermediate_nodes:
        new_frame = Frame(start_node, node, frame.obj_det_bbox, frame.direction)
        new_frames.append(new_frame)
        start_node = node

    new_frame = Frame(start_node, frame.node_2, frame.obj_det_bbox, frame.direction)
    new_frames.append(new_frame)

    return new_frames


def split_frame_based_on_nodes(frame_instances, node_instances):
    """
    Loops through frame instances to divide them if necessary
    :param frame_instances:  list of frame instances
    :param node_instances: list of node instances
    :return: new frame instances
    """
    new_frames = []
    for frame in frame_instances:
        split_frames = divide_frames(frame, node_instances)
        new_frames.extend(split_frames)
    return new_frames


def create_nominal_frame(node_instances):
    """
    Creates nominal frame if no frame detected
    :param node_instances: list of nodes
    :return: nominal frame
    """
    new_frame_instances = []
    for i in range(len(node_instances)):
        for j in range(i + 1, len(node_instances)):
            frame = Frame(node1=node_instances[i], node2=node_instances[j])
            new_frame_instances.append(frame)
    return new_frame_instances


def  initialize_pl(pl_boxes, node_instances):
    """
    Creates point load instances
    :param pl_boxes: list of pl bbox
    :param node_instances: node list
    :return: instances of pl
    """
    point_load_instances = []
    for box in pl_boxes:
        center_x = (box[1] + box[3]) / 2
        center_y = (box[0] + box[2]) / 2

        try:
            closest_node = min(node_instances,
                               key=lambda node: calculation_utils.euclidean_distance(
                                   [center_x, center_y], [node.bbox_x, node.bbox_y]))

            # For simplicity, I'm not assigning direction, value, and unit.
            # These can be assigned based on further criteria or user input.
            pl_instance = PointLoad(node=closest_node, obj_det_bbox=box[:4])
            point_load_instances.append(pl_instance)
        except IndexError:
            print(f"Error processing box with coordinates: {box}. Skipping to next box.")
    return point_load_instances


def find_closest_nodes_dl(x, y, nodes):
    """
    Finds the closest existing nodes for two ends of distributed load
    :param x: pixel value of x
    :param y: pixel value of y
    :param nodes: list of node instances
    :return: closes node
    """
    def node_distance(node):
        """
        Calculates distance between node bbox and coordinates of distributed load
        :param node: current node (taken from the list one by one)
        :return: distance between current node and obj det coordinates of distributed load
        """
        # Here, I'm assuming that the 'bbox_x' and 'bbox_y' attributes of the node provide the center of the node's bounding box.
        # If not, adjust this calculation to use the correct x and y values.
        return calculation_utils.euclidean_distance([x, y], [node.bbox_x, node.bbox_y])

    return min(nodes, key=node_distance)

def assign_nodes_dl(dl_with_direction, node_instances):
    """
    Assigns the closest nodes both ends of distributed load
    :param dl_with_direction: dl instances with directions decided
    :param node_instances: list keeping node instances
    :return: node attributed assigned distributed load instances
    """
    distributed_load_instances = []
    for box in dl_with_direction:
        if box[5] == "horizontal":

            y1 = box[0]
            y2 = box[2]
            x = int((box[1] + box[3]) / 2)
            # Get the closest node to x1 and x2
            c_node1 = find_closest_node(x, y1, node_instances)
            c_node2 = find_closest_node(x, y2, node_instances)

            dl_instance = DistributedLoad(node1=c_node1, node2=c_node2, obj_det_bbox=box[:4], direction=box[6])
            distributed_load_instances.append(dl_instance)
        elif box[5] == "vertical":
            x1 = box[1]
            x2 = box[3]
            y = int((box[0] + box[2]) / 2)

            # Get the closest node to x1 and x2
            c_node1 = find_closest_node(x1, y, node_instances)
            c_node2 = find_closest_node(x2, y, node_instances)

            dl_instance = DistributedLoad(node1=c_node1, node2=c_node2, obj_det_bbox=box[:4], direction=box[6])
            distributed_load_instances.append(dl_instance)
        else:
            # Need to fill this later
            pass
    return distributed_load_instances

def assign_nodes_tdl(tdl_with_direction, node_instances):
    """
    Assigns the closest nodes both ends of distributed load
    :param dl_with_direction: dl instances with directions decided
    :param node_instances: list keeping node instances
    :return: node attributed assigned distributed load instances
    """
    t_distributed_load_instances = []
    for box in tdl_with_direction:
        if box[5] == "horizontal":

            y1 = box[0]
            y2 = box[2]
            x = int((box[1] + box[3]) / 2)
            # Get the closest node to x1 and x2
            c_node1 = find_closest_node(x, y1, node_instances)
            c_node2 = find_closest_node(x, y2, node_instances)

            tdl_instance = TriangularDistributedLoad(node1=c_node1, node2=c_node2, obj_det_bbox=box[:4], direction=box[6])
            t_distributed_load_instances.append(tdl_instance)
        elif box[5] == "vertical":
            x1 = box[1]
            x2 = box[3]
            y = int((box[0] + box[2]) / 2)

            # Get the closest node to x1 and x2
            c_node1 = find_closest_node(x1, y, node_instances)
            c_node2 = find_closest_node(x2, y, node_instances)

            tdl_instance = TriangularDistributedLoad(node1=c_node1, node2=c_node2, obj_det_bbox=box[:4], direction=box[6])
            t_distributed_load_instances.append(tdl_instance)
        else:
            # Need to fill this later
            pass
    return t_distributed_load_instances


def find_connected_elements(sorted_nodes, new_frame_instances, fix_support_instances, point_load_instances,
                            roller_support_instances, pin_support_instances, distributed_load_instances):
    """
    Loops in node list, for each node finds connected elements
    :param sorted_nodes: nodes sorted in left to right, bottom to top
    :param new_frame_instances: divided frame instances
    :param fix_support_instances: fix support instances
    :param point_load_instances: point load instances
    :param roller_support_instances: roller support instances
    :param pin_support_instances: pin support instances
    :param distributed_load_instances: distributed load instances
    :return: returns described list
    """
    connected_elements = []
    for node in sorted_nodes:
        dummy_list = []
        # Check frames
        for frame in new_frame_instances:
            if node == frame.node_1:
                dummy_list.append(frame)

        # Check fix_supports
        for fix_support in fix_support_instances:
            if fix_support.node == node:
                dummy_list.append(fix_support)

        # Check point_loads
        for point_load in point_load_instances:
            if point_load.node == node:
                dummy_list.append(point_load)
        connected_elements.append(dummy_list)
        for roller_support in roller_support_instances:
            if roller_support.node == node:
                dummy_list.append(roller_support)
        connected_elements.append(dummy_list)
        for pin_support in pin_support_instances:
            if pin_support.node == node:
                dummy_list.append(pin_support)
        connected_elements.append(dummy_list)
        for distributed_load in distributed_load_instances:
            if distributed_load.node1 == node:
                dummy_list.append(distributed_load)
        connected_elements.append(dummy_list)
    return  connected_elements