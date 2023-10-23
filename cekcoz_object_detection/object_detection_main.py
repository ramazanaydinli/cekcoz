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
from structure import calculation_utils, structure_utils, image_utils, drawing_utils
import math
from structure.structure_classes import Node, Dimension, FixSupport, PinSupport, RollerSupport, Frame, PointLoad,\
    DistributedLoad
import cv2 as cv

def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def object_detection(image, path_of_image, incoming_filename):
    """
    Main part of the software
    :param image: input image obtained from telegram chat
    :param path_of_image: image is saved after taken from group, this variable keeps that saving path
    :param incoming_filename: filename created according to a pattern (user_name + datetime info)
    :return: returns solution (for now it is an image, later it will be a pdf file)
    """

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
    sorted_spacings = sorted(spacing_extracted_results, key=calculation_utils.sorting_key)

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
    # Checking if any pl exists, if it exists finds direction and appends below
    if point_load_instances:
        image_utils.pl_direction(point_load_instances, image)
    # Sorting nodes below for logical attribute assignment
    sorted_nodes = sorted(node_instances, key=lambda node: (node.shape_y, node.shape_x))
    # Distributed load instances created below
    dl_boxes = [box for box in applicable_list if box[-1] == 'distributed_load']
    # Deciding direction with image operations
    dl_with_direction = image_utils.dl_direction(dl_boxes, image, initial_iterations=1, max_iterations=10, threshold=1)
    # Creating instances
    distributed_load_instances = structure_utils.assign_nodes_dl(dl_with_direction, node_instances)
    # Function below assigns magnitude and value of distributed load
    distributed_load_instances = ocr_postprocess.extract_precise_text_dl(distributed_load_instances, ocr_results)

    # Function below finds connected elements to the nodes (needs to be optimized in future)
    connected_elements = structure_utils.find_connected_elements(
        sorted_nodes, new_frame_instances, fix_support_instances, point_load_instances,
        roller_support_instances, pin_support_instances, distributed_load_instances)

    # Taken images from telegram group will be recreated below
    # Saving path of recreated images
    created_image_saving_path = os.path.join(os.getcwd(), "created_images", incoming_filename)
    # Blank area around the created image
    padding_of_created_images = 300
    # For creating a rationally scaled image, find max values below
    max_shape_x, max_shape_y = image_utils.find_max_values(node_instances)
    # Scale of the created image obtained below (for each unit length corresponding pixels are calculated)
    node_value_pixel_x, node_value_pixel_y = image_utils.calculate_node_pixel_values(max_shape_x, max_shape_y,
                                                                                     img_width, img_height)
    # Width and height of created image calculated below
    ci_width, ci_height = image_utils.arrange_drawing_area(max_shape_x, max_shape_y, padding_of_created_images,
                                                           node_value_pixel_x, node_value_pixel_y)
    # Detected image will be drawn on the blank image initialized below
    new_image = image_utils.generate_image(ci_width, ci_height)
    # Drawing is completed below ( better algorithm is possible but for now no need to waste time on it)
    for element in connected_elements:
        for sub_element in element:
            has_frame = isinstance(sub_element, Frame)
            if has_frame:
                new_image = drawing_utils.draw_frame(sub_element, new_image, padding_of_created_images,
                                                     ci_height, node_value_pixel_x, node_value_pixel_y)
                # Similarly, you can check for other classes too if needed
            has_point_load = isinstance(sub_element, PointLoad)
            if has_point_load:
                new_image = drawing_utils.draw_pl(sub_element, new_image, padding_of_created_images,
                                                  ci_height, node_value_pixel_x, node_value_pixel_y)
            has_fix_support = isinstance(sub_element, FixSupport)
            if has_fix_support:
                # Do something if the sub_element is an instance of FixSupport
                new_image = drawing_utils.draw_fix_support(sub_element, new_image, frame_instances,
                                                           padding_of_created_images, ci_height,
                                                           node_value_pixel_x, node_value_pixel_y)
            has_roller_support = isinstance(sub_element, RollerSupport)
            if has_roller_support:
                # Do something if the sub_element is an instance of FixSupport
                new_image = drawing_utils.draw_roller_support(sub_element, new_image, padding_of_created_images,
                                                              ci_height, node_value_pixel_x, node_value_pixel_y)
            has_pin_support = isinstance(sub_element, PinSupport)
            if has_pin_support:
                # Do something if the sub_element is an instance of FixSupport
                new_image = drawing_utils.draw_pin_support(sub_element, new_image, padding_of_created_images,
                                                           ci_height, node_value_pixel_x, node_value_pixel_y)
            has_distributed_load = isinstance(sub_element, DistributedLoad)
            if has_distributed_load:
                # Do something if the sub_element is an instance of FixSupport
                new_image = drawing_utils.draw_distributed_load(sub_element, new_image, padding_of_created_images,
                                                                ci_height, node_value_pixel_x, node_value_pixel_y)
    new_image = drawing_utils.draw_dimension(spacing_instances, new_image, padding_of_created_images,
                                             ci_height, node_value_pixel_x, node_value_pixel_y)
    new_image = drawing_utils.label_nodes(node_instances, new_image, padding_of_created_images,
                                          ci_height, node_value_pixel_x, node_value_pixel_y)
    new_image_np = np.array(new_image)
    path_without_extension, extension = os.path.splitext(created_image_saving_path)
    saving_path_with_png = path_without_extension + ".png"
    cv.imwrite(saving_path_with_png, new_image_np)
    return saving_path_with_png
















