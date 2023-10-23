from PIL import Image, ImageDraw, ImageFont
import math
import numpy as np

def draw_frame(frame_instance, img, padding, height,
               node_value_pixel_x, node_value_pixel_y):
    """
    Draws frame on image
    :param frame_instance:
    :param img:
    :param padding:
    :param height:
    :param node_value_pixel_x:
    :param node_value_pixel_y:
    :return:
    """
    draw = ImageDraw.Draw(img)

    # Extract the two nodes
    node_1 = frame_instance.node_1
    node_2 = frame_instance.node_2

    # Determine the min and max nodes based on shape_x and shape_y
    min_node = min(node_1, node_2, key=lambda node: (node.shape_x, node.shape_y))
    max_node = max(node_1, node_2, key=lambda node: (node.shape_x, node.shape_y))

    min_x = padding /2 + node_value_pixel_x * min_node.shape_x
    min_y = height - (padding /2 + node_value_pixel_y * min_node.shape_y)
    max_x = padding /2 + node_value_pixel_x * max_node.shape_x
    max_y = height - (padding /2 + node_value_pixel_y * max_node.shape_y)

    # Now draw the line or frame or whatever representation you have
    # For example, if you're drawing a line:
    draw.line([(min_x, min_y), (max_x, max_y)], fill="black", width=3)
    return img


def frame_direction_from_node(current_node, frame_instances):
    """
    Decides the direction of frame
    :param current_node:
    :param frame_instances:
    :return:
    """
    direction = None
    for frame in frame_instances:
        # Check if current_node is one of the two nodes in the frame
        if current_node == frame.node_1 or current_node == frame.node_2:
            # Extracting nodes' shape_x and shape_y values
            x_values = [frame.node_1.shape_x, frame.node_2.shape_x]
            y_values = [frame.node_1.shape_y, frame.node_2.shape_y]

            if min(x_values) != max(x_values):  # x values are different
                if current_node.shape_x == min(x_values):
                    direction = "right"
                else:
                    direction = "left"
            else:  # y values are different
                if current_node.shape_y == min(y_values):
                    direction = "down"
                else:
                    direction = "up"
            break

    return direction


def draw_fix_support(fs_instance, img, frame_instances, padding, height,
                     node_value_pixel_x, node_value_pixel_y):
    """
    Draws fix support
    :param fs_instance:
    :param img:
    :param frame_instances:
    :param padding:
    :param height:
    :param node_value_pixel_x:
    :param node_value_pixel_y:
    :return:
    """
    draw = ImageDraw.Draw(img)
    node = fs_instance.node
    frame_direction = frame_direction_from_node(node, frame_instances)
    if frame_direction == "right":
        # End point (arrow's head)
        center_x = padding / 2 + node.shape_x * node_value_pixel_x
        center_y = height - (padding / 2 - node.shape_y * node_value_pixel_y)
        start_x, start_y = center_x - 12, center_y - 50
        for i in range(10):
            end_x, end_y = start_x + 12, start_y + 5
            draw.line((start_x, start_y, end_x, end_y), fill='black', width=2)
            start_y += 10
        draw.line((center_x, center_y - 50, center_x, center_y + 50), fill='black', width=2)
    # Fill this later
    elif frame_direction == "left":
        # End point (arrow's head)
        center_x = padding / 2 + node.shape_x * node_value_pixel_x
        center_y = height - (padding / 2 - node.shape_y * node_value_pixel_y)
        start_x, start_y = center_x, center_y - 50
        for i in range(10):
            end_x, end_y = start_x + 12, start_y + 10
            draw.line((start_x, start_y, end_x, end_y), fill='black', width=2)
            start_y += 10
        draw.line((center_x, center_y - 50, center_x, center_y + 50), fill='black', width=2)
        pass

    return img


def draw_roller_support(rs_instance, img, padding, height,
                        node_value_pixel_x, node_value_pixel_y):
    """
    Draws roller support
    :param rs_instance:
    :param img:
    :param padding:
    :param height:
    :param node_value_pixel_x:
    :param node_value_pixel_y:
    :return:
    """
    draw = ImageDraw.Draw(img)
    node = rs_instance.node
    color = "black"

    # Calculate node's coordinates based on shape_x and shape_y
    x = padding / 2 + node.shape_x * node_value_pixel_x
    y = height - (padding / 2 - node.shape_y * node_value_pixel_y)

    # Arrowhead coordinates
    arrow_tip = (x, y)
    arrow_left = (x - 15, y + 20)
    arrow_right = (x + 15, y + 20)

    # Draw the arrowhead in red for visibility
    draw.polygon([arrow_tip, arrow_left, arrow_right], fill=color)

    # Circle coordinates
    circle_radius = 3
    circle_1_center = (x - 10, arrow_left[1] + circle_radius + 2)
    circle_2_center = (x + 10, arrow_right[1] + circle_radius + 2)

    # Draw the circles in red for visibility
    draw.ellipse((circle_1_center[0] - circle_radius, circle_1_center[1] - circle_radius,
                  circle_1_center[0] + circle_radius, circle_1_center[1] + circle_radius), fill=color)
    draw.ellipse((circle_2_center[0] - circle_radius, circle_2_center[1] - circle_radius,
                  circle_2_center[0] + circle_radius, circle_2_center[1] + circle_radius), fill=color)

    return img


def draw_pin_support(ps_instance, img, padding, height,
                     node_value_pixel_x, node_value_pixel_y):
    """
    Draws pin support
    :param ps_instance:
    :param img:
    :param padding:
    :param height:
    :param node_value_pixel_x:
    :param node_value_pixel_y:
    :return:
    """
    draw = ImageDraw.Draw(img)
    node = ps_instance.node
    color = "black"
    # Calculate node's coordinates based on shape_x and shape_y
    x = padding / 2 + node.shape_x * node_value_pixel_x
    y = height - (padding / 2 + node.shape_y * node_value_pixel_y)

    # Arrowhead coordinates
    arrow_tip = (x, y)
    arrow_left = (x - 15, y + 20)
    arrow_right = (x + 15, y + 20)

    # Draw the arrowhead in blue
    draw.polygon([arrow_tip, arrow_left, arrow_right], fill=color)

    # Angled line coordinates (60-degree angles)
    line_length = 10
    dx = line_length * math.cos(math.radians(60))
    dy = line_length * math.sin(math.radians(60))

    spacing = 5

    # Starting point for the angled lines (bottom-left corner of the arrowhead)
    start_x = arrow_left[0] - 1
    start_y = arrow_left[1]

    # Draw six angled lines starting from the bottom-left corner of the arrowhead
    for i in range(7):
        line_start = (start_x + i * spacing, start_y)
        line_end = (line_start[0] + dx, line_start[1] + dy)
        draw.line([line_start, line_end], fill=color, width=2)

    return img


def write_point_load_value_directly(img, start_x, start_y, direction, unit, value):
    """
    Writing value and unit of point load on image
    :param img:
    :param start_x:
    :param start_y:
    :param direction:
    :param unit:
    :param value:
    :return:
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 25)
    text = f"{value} {unit}"

    if direction == "downward":
        draw.text((int(start_x) - 10, start_y - 30), text, fill="black",
                  font=font)  # Text above the arrow for downward direction

    elif direction == "right":
        text_width, text_height = draw.textsize(text, font=font)
        draw.text((start_x - text_width - 15, start_y - text_height / 2), text, fill="black",
                  font=font)  # Text to the left of the arrow for right direction

    return img


def draw_pl(pl_instance, img, padding, height,
            node_value_pixel_x, node_value_pixel_y):
    """
    Drawing point load on image
    :param pl_instance:
    :param img:
    :param padding:
    :param height:
    :param node_value_pixel_x:
    :param node_value_pixel_y:
    :return:
    """
    draw = ImageDraw.Draw(img)
    arrow_length = 75
    node_of_action = pl_instance.node
    if pl_instance.direction == "downward":
        # End point (arrow's head)
        end_x = padding / 2 + node_of_action.shape_x * node_value_pixel_x
        end_y = height - (padding / 2 + node_of_action.shape_y * node_value_pixel_y + 2)

        # Arrow's tail
        start_x = end_x
        start_y = end_y - arrow_length

        draw.line((start_x, start_y, end_x, end_y - 5), fill='green', width=3)

        # Arrowhead points, no need for trigonometry as it's vertical
        x1 = end_x - 5  # 5 is half of the arrowhead width
        y1 = end_y - 10  # 10 is the arrowhead length

        x2 = end_x + 5
        y2 = y1

        draw.polygon([(end_x, end_y), (x1, y1), (x2, y2)], fill="green")
        img = write_point_load_value_directly(img, start_x, start_y, "downward", pl_instance.unit, pl_instance.value)
    elif pl_instance.direction == "right":
        # End point (arrow's head)
        end_x = padding / 2 + node_of_action.shape_x * node_value_pixel_x - 3
        end_y = height - (padding / 2 + node_of_action.shape_y * node_value_pixel_y)

        # Arrow's tail
        start_x = end_x - arrow_length
        start_y = end_y

        draw.line((start_x, start_y, end_x, end_y), fill='green', width=3)

        # Arrowhead points, no need for trigonometry as it's vertical
        x1 = end_x - 10  # 10 is the arrowhead length
        y1 = end_y - 5  # 5 is half of the arrowhead width

        x2 = x1
        y2 = end_y + 5

        draw.polygon([(end_x, end_y), (x1, y1), (x2, y2)], fill="green")
        img = write_point_load_value_directly(img, start_x, start_y, "right", pl_instance.unit, pl_instance.value)
    return img


def write_distributed_load_value_directly(img, start_x, start_y, direction, unit, value):
    """
    Writing distributed load value and unit on image
    :param img:
    :param start_x:
    :param start_y:
    :param direction:
    :param unit:
    :param value:
    :return:
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 25)
    text = f"{value} {unit}"

    if direction == "downward":
        draw.text((int(start_x) - 10, start_y - 30), text, fill="black", font=font)

    elif direction == "right":

        # Create a new image for the text
        text_width, text_height = font.getsize(text)
        text_image = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))

        # Draw the text onto the new image
        draw_text = ImageDraw.Draw(text_image)
        draw_text.text((0, 0), text, fill="black", font=font)

        # Rotate the new image
        rotated_text = text_image.rotate(90, expand=1)

        # Update the start_x by -25 pixels for the skew effect in -x direction
        start_x -= 45

        # Position for the rotated text to be to the left of the arrow
        img.paste(rotated_text, (int(start_x + rotated_text.width - 15), int(start_y - text_height / 2)), rotated_text)

    elif direction == "left":
        # Create a new image for the text
        text_width, text_height = font.getsize(text)
        text_image = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))

        # Draw the text onto the new image
        draw_text = ImageDraw.Draw(text_image)
        draw_text.text((0, 0), text, fill="black", font=font)

        # Rotate the new image
        rotated_text = text_image.rotate(270, expand=1)

        # Update the start_x by 25 pixels for the skew effect in +x direction
        start_x += 45

        # Position for the rotated text to be to the left of the arrow
        img.paste(rotated_text, (int(start_x - rotated_text.width - 15), int(start_y - rotated_text.height / 2)),
                  rotated_text)

    return img


def draw_distributed_load(dl_instance, img, padding, height,
                          node_value_pixel_x, node_value_pixel_y):
    """
    Drawing distributed load on image
    :param dl_instance:
    :param img:
    :param extra_pixels_x:
    :param extra_pixels_y:
    :param height:
    :param node_value_pixel_x:
    :param node_value_pixel_y:
    :return:
    """
    draw = ImageDraw.Draw(img)
    arrow_tail_length = 40
    text_skew_y = 10
    text_skew_x = 10
    if dl_instance.direction == "downward":
        start_x = padding / 2 + dl_instance.node1.shape_x * node_value_pixel_x
        end_x = padding / 2 + dl_instance.node2.shape_x * node_value_pixel_x
        y = height - (padding / 2 + dl_instance.node1.shape_y * node_value_pixel_y + arrow_tail_length)

        # Draw horizontal line
        draw.line((start_x, y, end_x, y), fill='green', width=3)

        # Calculate number of segments based on width and draw arrows
        line_length = np.abs(end_x - start_x)
        segments = line_length / 50
        num_arrows = int(segments) + 1  # Plus 1 for the starting arrow
        spacing = line_length / (num_arrows - 1)  # Adjust spacing so there's an arrow at the end

        arrowhead_length = 10  # Length of the arrowhead
        for i in range(num_arrows):
            arrow_x = start_x + i * spacing

            # Vertical line of the arrow
            draw.line((arrow_x, y, arrow_x, y + arrow_tail_length), fill='green', width=3)

            # Arrowhead: Two lines meeting at the arrow's tip
            draw.line((arrow_x, y + arrow_tail_length, arrow_x - arrowhead_length / 2,
                       y + arrow_tail_length - arrowhead_length), fill='green', width=3)
            draw.line((arrow_x, y + arrow_tail_length, arrow_x + arrowhead_length / 2,
                       y + arrow_tail_length - arrowhead_length), fill='green', width=3)
        img = write_distributed_load_value_directly(img, ((start_x + end_x) / 2), y - text_skew_y, "downward",
                                                    dl_instance.unit, dl_instance.value)
    elif dl_instance.direction == "left":
        x = padding / 2 + dl_instance.node1.shape_x * node_value_pixel_x + arrow_tail_length
        start_y = height - (padding / 2 + dl_instance.node1.shape_y * node_value_pixel_y)
        end_y = height - (padding / 2 + dl_instance.node2.shape_y * node_value_pixel_y)

        draw.line((x, start_y, x, end_y), fill='green', width=3)
        # Calculate number of segments based on width and draw arrows
        line_length = np.abs(end_y - start_y)
        segments = line_length / 50
        num_arrows = int(segments) + 1  # Plus 1 for the starting arrow
        spacing = line_length / (num_arrows - 1)  # Adjust spacing so there's an arrow at the end

        arrowhead_length = 10  # Length of the arrowhead
        for i in range(num_arrows):
            arrow_y = min(start_y, end_y) + i * spacing

            # Vertical line of the arrow
            draw.line((x, arrow_y, x - arrow_tail_length, arrow_y), fill='green', width=3)

            # Arrowhead: Two lines meeting at the arrow's tip
            draw.line(
                (x - arrow_tail_length, arrow_y, x + arrowhead_length - arrow_tail_length, arrow_y + arrowhead_length),
                fill='green', width=3)
            draw.line(
                (x - arrow_tail_length, arrow_y, x + arrowhead_length - arrow_tail_length, arrow_y - arrowhead_length),
                fill='green', width=3)

        img = write_distributed_load_value_directly(img, x, ((start_y + end_y) / 2) - text_skew_y, "left",
                                                    dl_instance.unit, dl_instance.value)

    elif dl_instance.direction == "right":
        x = padding / 2 + dl_instance.node1.shape_x * node_value_pixel_x - arrow_tail_length
        start_y = height - (padding / 2 + dl_instance.node1.shape_y * node_value_pixel_y)
        end_y = height - (padding / 2 + dl_instance.node2.shape_y * node_value_pixel_y)

        draw.line((x, start_y, x, end_y), fill='green', width=3)
        # Calculate number of segments based on width and draw arrows
        line_length = np.abs(end_y - start_y)
        segments = line_length / 50
        num_arrows = int(segments) + 1  # Plus 1 for the starting arrow
        spacing = line_length / (num_arrows - 1)  # Adjust spacing so there's an arrow at the end

        arrowhead_length = 10  # Length of the arrowhead
        for i in range(num_arrows):
            arrow_y = start_y + i * spacing

            # Vertical line of the arrow
            draw.line((x, arrow_y, x + arrow_tail_length, arrow_y), fill='green', width=3)

            # Arrowhead: Two lines meeting at the arrow's tip
            draw.line(
                (x + arrow_tail_length, arrow_y, x - arrowhead_length + arrow_tail_length, arrow_y + arrowhead_length),
                fill='green', width=3)
            draw.line(
                (x + arrow_tail_length, arrow_y, x - arrowhead_length + arrow_tail_length, arrow_y - arrowhead_length),
                fill='green', width=3)

        img = write_distributed_load_value_directly(img, x, ((start_y + end_y) / 2) - text_skew_y, "right",
                                                    dl_instance.unit, dl_instance.value)
    return img


def draw_dimension(dimensions, img, padding, height,
                   node_value_pixel_x, node_value_pixel_y):
    """
    Draws dimensions on a given image based on the provided Dimension instances.

    Args:
    - dimensions (list): List of Dimension objects.
    - img (Image): PIL Image object where the dimensions should be drawn.

    Returns:
    - Modified Image object with lines drawn and texts added.
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 25)  # Adjust as necessary
    line_color = 'black'
    pixel_difference = 75

    for dimension in dimensions:
        if dimension.node_1.shape_y == dimension.node_2.shape_y:
            # Use shape_x of node_1 and node_2 for determining x-coordinates and offset it by extra_pixels_x/2.
            start_x = dimension.node_1.shape_x * node_value_pixel_x + padding / 2
            start_y = height - (padding / 2 - pixel_difference)
            end_x = dimension.node_2.shape_x * node_value_pixel_x + padding / 2
            end_y = start_y

            # Draw the horizontal line
            draw.line((start_x, start_y, end_x, end_y), fill=line_color, width=2)

            # Draw the vertical lines at the start and end of each horizontal line
            draw.line((start_x, start_y - 8, start_x, start_y + 8), fill=line_color, width=2)
            draw.line((end_x, end_y - 8, end_x, end_y + 8), fill=line_color, width=2)

            # Write the value and unit just below the horizontal line
            text_x = (start_x + end_x) / 2 - 10  # A small offset to center the text
            text_y = end_y + 5  # A small offset to place the text below the line
            draw.text((text_x, text_y), f"{dimension.value} {dimension.unit}", fill=line_color, font=font)
        elif dimension.node_1.shape_x == dimension.node_2.shape_x:
            start_x = dimension.node_1.shape_x * node_value_pixel_x + padding / 2 + pixel_difference
            start_y = height - (dimension.node_1.shape_y * node_value_pixel_y + padding / 2)
            end_x = start_x
            end_y = height - (dimension.node_2.shape_y * node_value_pixel_y + padding / 2)

            # Draw the horizontal line
            draw.line((start_x, start_y, end_x, end_y), fill=line_color, width=2)

            # Draw the horizontal lines at the start and end of each vertical line
            draw.line((start_x - 8, start_y, start_x + 8, start_y), fill=line_color, width=2)
            draw.line((end_x - 8, end_y, end_x + 8, end_y), fill=line_color, width=2)

            # Write the value and unit just below the horizontal line
            text_x = start_x + 6  # A small offset to center the text
            text_y = (start_y + end_y) / 2 + -10  # A small offset to place the text below the line
            draw.text((text_x, text_y), f"{dimension.value} {dimension.unit}", fill=line_color, font=font)
    return img


def label_nodes(node_instances, img, padding, height, node_value_pixel_x, node_value_pixel_y):
    """
    Adds labels (node names) to the nodes on the given image.

    Args:
    - node_instances (list): List of Node objects.
    - img (Image): PIL Image object where the node names should be written.

    Returns:
    - Modified Image object with node names written.
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 25)
    pixel_difference = 15

    for node in node_instances:
        if node.name:  # Only label nodes that have a name
            x = node.shape_x * node_value_pixel_x + padding / 2 - 35
            y = height - (node.shape_y * node_value_pixel_y + padding / 2 - pixel_difference)

            draw.text((x, y), node.name, font=font, fill="black")

    return img


