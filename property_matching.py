


def create_components(bbox, text):
    "bbox: list, each element contains [ymin,xmin,ymax,xmax,label_name]"
    "text: list, each element contains [text, [topleft_x, topleft_y, topright_x, topright_y,bottomright_x," \
    "bottomright_y,bottomleft_x,bottomleft_y"


    spacing_info = []
    for element in bbox:
        if element[4] == "spacing":
            spacing_info.append(element)
    print(spacing_info)
    calculation_ready_components = []






    return calculation_ready_components