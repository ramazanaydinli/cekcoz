import  cv2 as cv
def show_object_detection_results(image_np, detections, category_index, label_id_offset=1):
    """
    Displays object detection results labeled on input image
    :param image_np: numpy formatted image
    :param detections: detection results
    :param category_index: detection class index
    :param label_id_offset: class index - label name dict
    :return: None (shows detection result)
    """
    image_np_with_detections = image_np.copy()
    from object_detection.utils import visualization_utils as viz_utils
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