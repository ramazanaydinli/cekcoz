import re
from structure import calculation_utils


def clean_ocr_results(ocr_results):
    """
    Removing bad reading characters
    :param ocr_results: initial reading results
    :return: cleansed results
    """
    # A list of meaningless readings. You can extend this list as required
    meaningless_readings = ["-"]

    # Filter out results with meaningless readings
    cleaned_results = [result for result in ocr_results if result[0] not in meaningless_readings]

    return cleaned_results


def is_valid_reading_distance(reading):
    """
    Checks if reading fits pattern considering distance units
    (only check if it contains units, secondary regex will be done)
    :param reading: reading results obtained from ocr
    :return: returns reading if it fits pattern
    """
    pattern = r"(\d+\s*(?:meters?|ft|feet|inch(?:es)?|m))"
    return re.search(pattern, reading)


def filter_results_by_distance(obj_detection_bbox, ocr_results, threshold):
    """
    Narrowing down possible matches for obj's and ocr's by discarding values farther away than threshold
    :param obj_detection_bbox: bbox of object
    :param ocr_results: bbox of ocr reading
    :param threshold: limit value, after calculating center of obj bbox and ocr reading bbox,
    if calculated value is more than threshold, that value will be discarded
    :return: returns readings within threshold value
    """
    filtered_results = []
    obj_center = calculation_utils.compute_center_obj(obj_detection_bbox)

    for result in ocr_results:
        reading, ocr_bbox = result
        if not is_valid_reading_distance(reading):  # check if the reading is valid
            continue
        ocr_center = calculation_utils.compute_center_ocr(ocr_bbox)
        distance = calculation_utils.euclidean_distance(obj_center, ocr_center)
        if distance < threshold:
            filtered_results.append(result)

    return filtered_results


def adjust_threshold(bbox):
    """
    Dynamically adjusts threshold for "filter_results_by_distance" function
    Logic is finding smaller axis (x or y) and adds some pixel to that axis length
    :param bbox: bbox of the object
    :return: calculated threshold value
    """
    if bbox[3]-bbox[1] > bbox[2]-bbox[0]:
        return bbox[2]-bbox[0]+10
    else:
        return bbox[3]-bbox[1] + 10


def match_spacings(spacing_bbox, readings):
    """
    Takes detection results classified as "spacing" and matching them with corresponding reading
    :param spacing_bbox: bbox of spacing instance
    :param readings: ocr readings
    :return: list [[ocr reading], [reading bbox], [obj det bbox]]
    """
    final_results = []
    for bbox in spacing_bbox:
        threshold = adjust_threshold(bbox)
        filtered_ocr_results = filter_results_by_distance(bbox, readings, threshold)
        if filtered_ocr_results:
            for item in filtered_ocr_results:
                result = [[item[0]], item[1], bbox]
                final_results.append(result)
    return final_results


def ocr_secondary_filter_distance(matching_results):
    """
    Filters readings to remove absurd characters
    :param matching_results: first filter results
    :return: input with bad readings removed
    """
    pattern = r"(\d+([.]\d+)?|(\d+\s*/\s*\d+))\s*(m|meters?)"
    for item in matching_results:
        reading = item[0][0]
        match = re.search(pattern, reading)
        if match:
            extracted_value = match.group()
            item[0][0] = extracted_value  # Update the original reading with the extracted value
    return matching_results


def extract_value_unit(sec_filtered_results):
    """
    Extracts precise value and units from ocr text
    :param sec_filtered_results: text results after two times filtering
    :return: replacing input's text part with a list containing unit and value
    """
    # regex pattern to match value and unit (supports meter, m, feet, ft, inch)
    pattern = r'(\d+)\s*(meters?|ft|feet|inch(?:es)?|m)'
    for element in sec_filtered_results:
        match = re.search(pattern, element[0][0], re.IGNORECASE)
        if match:
            value, unit = match.groups()
            element[0] = [int(value), unit]
    return sec_filtered_results


def append_names_to_nodes(node_instances, ocr_results, distance_threshold=100):
    """
    Appends name to the nodes, node names are generally symbolized with one character (A,B,C, etc.)
    Closest reading mathching to this pattern is appended if reading is in the threshold value
    :param node_instances: node list
    :param ocr_results: ocr text
    :param distance_threshold: pixel value
    :return: name appended node list
    """
    pattern = r'^[A-Za-z]$'
    for text, bbox in ocr_results:
        if re.match(pattern, text):  # Check if the text is a single letter (either uppercase or lowercase)
            center_x = sum(bbox[i] for i in [0, 2, 4, 6]) / 4
            center_y = sum(bbox[i] for i in [1, 3, 5, 7]) / 4

            distances = [(node, calculation_utils.euclidean_distance([center_x, center_y], [node.bbox_x, node.bbox_y]))
                         for node in node_instances]
            closest_node, min_distance = min(distances, key=lambda x: x[1])

            if min_distance <= distance_threshold:
                closest_node.name = text
    return None


def extract_pl_text(pl_boxes, ocr_results, threshold = 200):
    """
    detects texts which are closet than threshold pixel
    :param pl_boxes: list of pl boxes
    :param ocr_results: ocr readins results
    :return: possible readings of pl
    """
    texts_within_range = []

    for pl_box in pl_boxes:
        pl_center_x = (pl_box[1] + pl_box[3]) / 2
        pl_center_y = (pl_box[0] + pl_box[2]) / 2

        for text_entry in ocr_results:
            text, bbox = text_entry[0], text_entry[1]
            center_x = sum(bbox[i] for i in [0, 2, 4, 6]) / 4
            center_y = sum(bbox[i] for i in [1, 3, 5, 7]) / 4

            if calculation_utils.euclidean_distance([pl_center_x, pl_center_y], [center_x, center_y]) <= threshold:
                texts_within_range.append(text_entry)
    return texts_within_range


def extract_precise_text(point_load_instances, possible_pl_text):
    """
    Extracts precise magnitude and value for point loads and assigns related attributes
    :param point_load_instances: list containing pl instances
    :param possible_pl_text: close text to bbox of object detection
    :return: related attributes assigned version of point load instances
    """
    pattern = r"(?:[A-Za-z]*\s*=\s*)?(?P<value>\d+)\s*(?P<unit>kN|k|t)(?P<type>/ft)?"
    for pl in point_load_instances:
        closest_text_entry = None
        min_distance = float('inf')

        for text_entry in possible_pl_text:
            text = text_entry[0]
            bbox = text_entry[1]
            center_x = sum(bbox[i] for i in [0, 2, 4, 6]) / 4
            center_y = sum(bbox[i] for i in [1, 3, 5, 7]) / 4

            match = re.search(pattern, text)
            if match:
                distance = calculation_utils.euclidean_distance([center_x, center_y], [pl.node.bbox_x, pl.node.bbox_y])
                if distance < min_distance:
                    closest_text_entry = text_entry
                    closest_text_bbox = bbox
                    min_distance = distance

        # Assign the closest text entry (if found) to the PointLoad instance
        if closest_text_entry:
            text = closest_text_entry[0]
            match = re.search(pattern, text)
            value = match.group("value")
            unit = match.group("unit")

            pl.value = value
            pl.unit = unit
            pl.text_bbox = closest_text_bbox
    return point_load_instances

def find_closest_text_dl(ocr_list, bbox):
    """
    Detects the closest text to current distributed load instance
    :param ocr_list: list keeping ocr text
    :param bbox: object detection bbox of distributed load instance
    :return: the closest text
    """
    y1, x1, y2, x2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Find the closest text entry to the center of bbox
    closest_text = min(
        ocr_list,
        key=lambda entry: (center_x - (entry[1][0] + entry[1][2] + entry[1][4] + entry[1][6]) / 4)**2 +
                         (center_y - (entry[1][1] + entry[1][3] + entry[1][5] + entry[1][7]) / 4)**2
    )

    return closest_text[0]

def extract_precise_text_dl(dl_instances, ocr_text):
    """
    Decides proper text belongs to the distributed load
    :param dl_instances: list keeping instances of distributed load
    :param ocr_text: list keeping ocr reading result text
    :return: instances with related attributes assigned
    """
    pattern = r"(?i)(?P<value>\d+(\.\d+)?)\s*(?P<unit>kN/m|N/m|k/ft)"
    for dl in dl_instances:
        bbox = dl.obj_det_bbox  # Using the obj_det_bbox attribute

        text = find_closest_text_dl(ocr_text, bbox)

        match = re.search(pattern, text)
        if match:
            dl.value = float(match.group("value"))  # Convert to float instead of int
            dl.unit = match.group("unit")
    return dl_instances