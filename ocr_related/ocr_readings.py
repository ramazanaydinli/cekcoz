from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time


def read_text_on_image(path):
    """
    Reads text on image
    :param path: input image path
    :return: ocr text , each element contains [text, [topleft_x, topleft_y, topright_x, topright_y,bottomright_x," \
    "bottomright_y,bottomleft_x,bottomleft_y]]"
    """

    subscription_key = ""
    endpoint = ""

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    read_image_path = path
    # Open the image
    read_image = open(read_image_path, "rb")

    # Call API with image and raw response (allows you to get the operation location)
    read_response = computervision_client.read_in_stream(read_image, raw=True)
    # Get the operation location (URL with ID as last appendage)
    read_operation_location = read_response.headers["Operation-Location"]
    # Take the ID off and use to get results
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for the retrieval of the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        print('Waiting for result...')
        time.sleep(3)

    reading_results = []
    # Print results, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                dummy_list = [line.text, line.bounding_box]
                reading_results.append(dummy_list)
    return reading_results
