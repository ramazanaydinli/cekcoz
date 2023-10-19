
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