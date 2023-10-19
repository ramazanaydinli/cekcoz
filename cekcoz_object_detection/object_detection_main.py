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












