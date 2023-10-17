import cv2 as cv
import os
from object_detection.utils import config_util
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import label_map_util
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from ocr_readings import read_text_on_image
from property_matching import create_components



# Alttaki satırı sakın açayım deme hem sapıtıyor hem yavaşlıyor
# @tf.function
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

    # cv.imshow("image",image)
    # cv.waitKey()

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor, detection_model)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.3,
        agnostic_mode=False)

    def process_data(box, detection_class, score):
        return list([box[0], box[1], box[2], box[3], detection_class + label_id_offset, score])

    aggregated_list = list(map(process_data, detections['detection_boxes'], detections['detection_classes'],
                          detections['detection_scores']))

    score_threshold = 0.5
    applicable_list = []
    for whole_values in aggregated_list:
        dummy_list = []
        if whole_values[5] > score_threshold:
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
            applicable_list.append(dummy_list)

    def area(bbox):
        """Calculate area of a bounding box."""
        ymin, xmin, ymax, xmax, _ = bbox
        return (ymax - ymin) * (xmax - xmin)

    def overlap_area(bbox1, bbox2):
        """Calculate overlap area between two bounding boxes."""
        ymin1, xmin1, ymax1, xmax1 = bbox1[:4]
        ymin2, xmin2, ymax2, xmax2 = bbox2[:4]

        overlap_ymin = max(ymin1, ymin2)
        overlap_xmin = max(xmin1, xmin2)
        overlap_ymax = min(ymax1, ymax2)
        overlap_xmax = min(xmax1, xmax2)

        # Check if the boxes overlap at all
        if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
            return area([overlap_ymin, overlap_xmin, overlap_ymax, overlap_xmax, ''])
        return 0

    filtered_boxes = []
    for i in range(len(applicable_list)):
        for j in range(i + 1, len(applicable_list)):
            if applicable_list[i][4] == applicable_list[j][4]:  # If the labels are the same
                overlap = overlap_area(applicable_list[i], applicable_list[j])
                if overlap >= 0.8 * area(applicable_list[i]):  # If bbox j contains 80% or more of bbox i
                    break
                elif overlap >= 0.8 * area(applicable_list[j]):  # If bbox i contains 80% or more of bbox j
                    continue
        else:  # If the loop didn't break, add the box to the filtered list
            filtered_boxes.append(applicable_list[i])

    print(filtered_boxes)


    reading_results = read_text_on_image(path_of_image)
    print(reading_results)
    cv.imshow("image with detections", image_np_with_detections)
    cv.waitKey()

    calculation_ready_components = create_components(filtered_boxes, reading_results)





