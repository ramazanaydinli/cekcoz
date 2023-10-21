import os
import sys
from object_detection.utils import config_util
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import label_map_util
import numpy as np
from ocr_related.ocr_readings import read_text_on_image
from cekcoz_object_detection import object_detection_test
from cekcoz_object_detection import data_postprocess
from ocr_related import ocr_postprocess
from structure import calculation_utils, structure_utils


def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def object_detection(image, path_of_image):


    user_path = os.path.expanduser("~")
    training_demo_path = os.path.join(user_path, "Desktop", "cekcoz_v3", "workspace", "training_demo")
    config_path = os.path.join(training_demo_path, "models", "cekcoz_resnet", "pipeline.config")
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    checkpoint_path = os.path.join(training_demo_path, "models", "cekcoz_resnet")
    ckpt.restore(os.path.join(checkpoint_path, 'ckpt-98')).expect_partial()


    label_path = os.path.join(training_demo_path, "annotations", "label_map.pbtxt")
    category_index = label_map_util.create_category_index_from_labelmap(label_path,use_display_name=True)

    image_np = np.array(image)
    img_height, img_width = image_np.shape[0], image_np.shape[1]

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor, detection_model)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    label_id_offset = 1

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


    # ----------For testing object detection results, code below could be used---------------
    # object_detection_test.show_object_detection_results(image_np, detections, category_index)

    # Packs all complex data into usable data with some format
    aggregated_list = list(map(data_postprocess.process_data, detections['detection_boxes'],
                               detections['detection_classes'],detections['detection_scores']))


    # Detections which has confidence score less than threshold value described below will be discarded
    score_threshold = 0.5


    # After discarding ambiguous detections, rest will be accumulated in the list below
    applicable_list = data_postprocess.remove_ambiguous_detections(score_threshold, aggregated_list, category_index,
                                                                   img_width, img_height)

    # Models trained with small dataset may detect same object more than once, function below will remove multiple
    # detections, if dataset is large enough, this function could be removed
    applicable_list = data_postprocess.remove_multiple_detections(applicable_list, percentage_thresh=0.7)

    # Function below reads the text on the image and send back results, more information could be obtained from azure
    reading_results = read_text_on_image(path_of_image)

    # ----------For testing OCR reading results, code below could be used---------------
    # print(reading_results)

    # Reading results are the most problematic part of this project use functions below to correct mistakes
    ocr_results = ocr_postprocess.clean_ocr_results(reading_results)

    # Test cleansed result below
    #print(ocr_results)

    # Construction of question starts with dimension related operations
    # Below, obj det resultant classes filtered and "spacing" classes are separated
    spacing_bbox = [item[:4] for item in applicable_list if item[4] == 'spacing']
    # Below, obtained results above will be assigned with ocr text
    spacing_matching_results = ocr_postprocess.match_spacings(spacing_bbox, ocr_results)
    # Below, ocr text will be filtered for removing misread bad characters
    spacing_sec_filtered_results = ocr_postprocess.ocr_secondary_filter_distance(spacing_matching_results)
    # After these two filtering process, below extraction will be done
    spacing_extracted_results = ocr_postprocess.extract_value_unit(spacing_sec_filtered_results)

    # Before construction sorting is required
    sorted_spacings = sorted(spacing_extracted_results,
                             key=lambda entry: calculation_utils.sorting_key(entry, img_height))
    # Below, direction of the dimension is decided ( for now only horizontal and vertical implemented)
    sorted_spacings_with_directions = structure_utils.dimension_direction(sorted_spacings, image)

    # Initialization of construction is below
    spacing_instances, node_instances = structure_utils.initialize_parameters(sorted_spacings_with_directions)
    # Sometimes geometry of the shape creates nodes with negative values which is corrected below
    node_instances = structure_utils.correct_shapes(node_instances)
    # For every break point there should be a node which are ensured below
    node_instances = structure_utils.find_and_append_missing_nodes(node_instances)
    # Nodes may have a name, which is found and appended below
    ocr_postprocess.append_names_to_nodes(node_instances, ocr_results, distance_threshold=100)

    # Supports are extracted below
    support_bbox = [box for box in applicable_list if 'support' in box[-1]]
    fix_support_instances, pin_support_instances, roller_support_instances = structure_utils.\
        create_support_instances(support_bbox, node_instances)

    # Frames are created below
    frame_boxes = [box for box in applicable_list if box[-1] == 'frame']
    # Below, direction is appended to frame info list
    frame_boxes = structure_utils.find_frame_direction(frame_boxes, image)
    # Instances of frame created below
    frame_instances = structure_utils.initialize_frames(frame_boxes, node_instances)
    # Below, frames divided parts if there are any nodes between the ones frame defined
    new_frame_instances = structure_utils.split_frame_based_on_nodes(frame_instances, node_instances)
    # Since model trained with small dataset, sometimes it may not recognize frame
    # If a condition like above occurs, we will define a frame accross the nodes (this function will be deleted future)
    if not new_frame_instances:
        new_frame_instances = structure_utils.create_nominal_frame(node_instances)

    # Point load instances created below
    pl_boxes = [box for box in applicable_list if box[-1] == 'point_load']
    # Creates instances of point loads
    point_load_instances = structure_utils.initialize_pl(pl_boxes, node_instances)
    # Extracts possible texts related with pl
    possible_pl_text = ocr_postprocess.extract_pl_text(pl_boxes, ocr_results)
    # Extracting precise text
    point_load_instances = ocr_postprocess.extract_precise_text(point_load_instances, possible_pl_text)














