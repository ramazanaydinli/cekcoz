import cv2 as cv
import os
from object_detection.utils import config_util
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import label_map_util
import numpy as np
from object_detection.utils import visualization_utils as viz_utils







def object_detection(image):
    user_path = os.path.expanduser("~")
    training_demo_path = os.path.join(user_path, "Desktop", "cekcoz_v3", "workspace", "training_demo")
    config_path = os.path.join(training_demo_path, "models", "cekcoz_resnet", "pipeline.config")
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    checkpoint_path = os.path.join(training_demo_path, "models", "cekcoz_resnet")
    ckpt.restore(os.path.join(checkpoint_path, 'ckpt-98')).expect_partial()

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections
    label_path = os.path.join(training_demo_path, "annotations", "label_map.pbtxt")
    category_index = label_map_util.create_category_index_from_labelmap(label_path,use_display_name=True)

    image_np = np.array(image)
    img_height, img_width = image_np.shape[0], image_np.shape[1]

    cv.imshow("image",image)
    cv.waitKey()

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

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


    cv.imshow("image with detections", image_np_with_detections)
    cv.waitKey()






