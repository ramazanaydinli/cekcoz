

def process_data(box, detection_class, score, label_id_offset=1):
    """
    Taking TensforFlow formatted information and formats them for easy usage
    :param box: obj det bboxes
    :param detection_class: class of the detected object
    :param score: detection score
    :param label_id_offset: Highly likely equals to 1 ( for offset purpose)
    :return: returns the list including information with some logic
    """
    return list([box[0], box[1], box[2], box[3], detection_class + label_id_offset, score])


def remove_ambiguous_detections(threshold, data_list, category_index, img_width, img_height):
    """
    Will take object detection results and remove doubtfull detections
    :param threshold: confidence score, detections has lower than this score will be removed
    :param data_list: list containing object detection results
    :param category_index: index of the classes
    :param img_width: x-pixel value of the current image
    :param img_height: y-pixel value of the current image
    :return: returns list which has cleared all ambigous detections
    """
    new_data_list = []
    for whole_values in data_list:
        dummy_list = []
        if whole_values[5] > threshold:
            label_name = (category_index.get(int(whole_values[4]))).get("name")
            ymin = int(whole_values[0] * img_height)
            xmin = int(whole_values[1] * img_width)
            ymax = int(whole_values[2] * img_height)
            xmax = int(whole_values[3] * img_width)
            dummy_list.append(ymin)
            dummy_list.append(xmin)
            dummy_list.append(ymax)
            dummy_list.append(xmax)
            dummy_list.append(label_name)
            new_data_list.append(dummy_list)
    return new_data_list


def overlap_percentage(bbox1, bbox2):
    """
    Takes two bounding boxes, calculates overlapping percentage
    :param bbox1: First Bbox
    :param bbox2: Second Bbox
    :return: Overlapping percentage
    """

    y1_1, x1_1, y2_1, x2_1 = bbox1
    y1_2, x1_2, y2_2, x2_2 = bbox2

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Compute the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Compute the percentage overlap
    percentage_overlap = intersection_area / min(bbox1_area, bbox2_area)

    return percentage_overlap


def remove_multiple_detections(detections_list, percentage_thresh):
    """
    Function simply takes all detections, remove smaller detection if overlapping area is higher than threshold
    :param detections_list: list of object detection results
    :param percentage_thresh: threshold overlapping percentage
    :return: returns cleared list
    """
    to_remove = []
    for i in range(len(detections_list)):
        for j in range(len(detections_list)):
            if i != j and detections_list[i][-1] == detections_list[j][-1]:
                if overlap_percentage(detections_list[i][:4], detections_list[j][:4]) >= percentage_thresh:
                    # Check which one is smaller and mark it for removal
                    area_i = (detections_list[i][2] - detections_list[i][0]) * (
                                detections_list[i][3] - detections_list[i][1])
                    area_j = (detections_list[j][2] - detections_list[j][0]) * (
                                detections_list[j][3] - detections_list[j][1])
                    if area_i < area_j:
                        to_remove.append(i)
                    else:
                        to_remove.append(j)
    to_remove = list(set(to_remove))
    detections_list = [detections_list[i] for i in range(len(detections_list)) if i not in to_remove]
    return detections_list
