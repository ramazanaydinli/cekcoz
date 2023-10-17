
from PIL import ImageDraw



extra_pixels_x = 300
extra_pixels_y = 300
def draw_frame(frame_instance, img):
    draw = ImageDraw.Draw(img)

    # Extract the two nodes
    node_1 = frame_instance.node_1
    node_2 = frame_instance.node_2

    # Determine the min and max nodes based on shape_x and shape_y
    min_node = min(node_1, node_2, key=lambda node: (node.shape_x, node.shape_y))
    max_node = max(node_1, node_2, key=lambda node: (node.shape_x, node.shape_y))

    min_x = extra_pixels_x /2 + node_value_pixel_x * min_node.shape_x
    min_y = height - (extra_pixels_y /2 + node_value_pixel_y * min_node.shape_y)
    max_x = extra_pixels_x /2 + node_value_pixel_x * max_node.shape_x
    max_y = height - (extra_pixels_y /2 + node_value_pixel_y * max_node.shape_y)

    # Now draw the line or frame or whatever representation you have
    # For example, if you're drawing a line:
    draw.line([(min_x, min_y), (max_x, max_y)], fill="black", width=3)
    return img