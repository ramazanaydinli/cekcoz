texts_within_range = [['10 kN', [115.0, 25.0, 157.0, 24.0, 157.0, 38.0, 115.0, 38.0]], ['20 kN', [244.0, 23.0, 286.0, 22.0, 287.0, 36.0, 244.0, 37.0]], ['A', [22.0, 84.0, 33.0, 84.0, 34.0, 95.0, 23.0, 95.0]], ['B.', [86.0, 82.0, 102.0, 83.0, 103.0, 96.0, 86.0, 94.0]], ['C', [215.0, 81.0, 225.0, 81.0, 226.0, 94.0, 215.0, 94.0]], ['D', [297.0, 81.0, 307.0, 81.0, 307.0, 93.0, 297.0, 93.0]], ['1 m', [61.0, 141.0, 88.0, 142.0, 88.0, 153.0, 61.0, 153.0]], ['2 m', [159.0, 139.0, 186.0, 140.0, 186.0, 153.0, 158.0, 152.0]], ['1 m', [257.0, 139.0, 284.0, 140.0, 284.0, 151.0, 256.0, 152.0]], ['X', [245.0, 158.0, 237.0, 172.0, 225.0, 162.0, 233.0, 148.0]], ['20 kN', [244.0, 23.0, 286.0, 22.0, 287.0, 36.0, 244.0, 37.0]], ['5 kN', [375.0, 22.0, 410.0, 22.0, 409.0, 35.0, 375.0, 35.0]], ['C', [215.0, 81.0, 225.0, 81.0, 226.0, 94.0, 215.0, 94.0]], ['D', [297.0, 81.0, 307.0, 81.0, 307.0, 93.0, 297.0, 93.0]], ['E', [347.0, 79.0, 358.0, 80.0, 357.0, 93.0, 347.0, 92.0]], ['1 m', [257.0, 139.0, 284.0, 140.0, 284.0, 151.0, 256.0, 152.0]], ['1 m', [326.0, 139.0, 351.0, 139.0, 352.0, 152.0, 326.0, 151.0]], ['X', [245.0, 158.0, 237.0, 172.0, 225.0, 162.0, 233.0, 148.0]], ['10 kN', [115.0, 25.0, 157.0, 24.0, 157.0, 38.0, 115.0, 38.0]], ['20 kN', [244.0, 23.0, 286.0, 22.0, 287.0, 36.0, 244.0, 37.0]], ['5 kN', [375.0, 22.0, 410.0, 22.0, 409.0, 35.0, 375.0, 35.0]], ['B.', [86.0, 82.0, 102.0, 83.0, 103.0, 96.0, 86.0, 94.0]], ['C', [215.0, 81.0, 225.0, 81.0, 226.0, 94.0, 215.0, 94.0]], ['D', [297.0, 81.0, 307.0, 81.0, 307.0, 93.0, 297.0, 93.0]], ['E', [347.0, 79.0, 358.0, 80.0, 357.0, 93.0, 347.0, 92.0]], ['1 m', [61.0, 141.0, 88.0, 142.0, 88.0, 153.0, 61.0, 153.0]], ['2 m', [159.0, 139.0, 186.0, 140.0, 186.0, 153.0, 158.0, 152.0]], ['1 m', [257.0, 139.0, 284.0, 140.0, 284.0, 151.0, 256.0, 152.0]], ['1 m', [326.0, 139.0, 351.0, 139.0, 352.0, 152.0, 326.0, 151.0]], ['X', [245.0, 158.0, 237.0, 172.0, 225.0, 162.0, 233.0, 148.0]]]



import re

pattern = r"(?:[A-Za-z]*\s*=\s*)?(?P<value>\d+)\s*(?P<unit>kN|k)(?P<type>/ft)?"

assigned_pointloads = set()  # To keep track of already assigned PointLoad instances
def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

class PointLoad:
    def __init__(self, node=None, direction=None, value=None, unit=None):
        self.node = node
        self.direction = direction
        self.value = value
        self.unit = unit

    def __repr__(self):
        return f"PointLoad(node={self.node}, direction={self.direction}, value={self.value}, unit={self.unit})"
class Node:
    def __init__(self, bbox_x=None, bbox_y=None, shape_x=None, shape_y=None, name=None):
        self.bbox_x = bbox_x
        self.bbox_y = bbox_y
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.bbox_x == other.bbox_x and self.bbox_y == other.bbox_y
                    and self.shape_x == other.shape_x and self.shape_y == other.shape_y
                    and self.name == other.name)
        return False

    def __repr__(self):
        return f"Node(name={self.name}, bbox_x={self.bbox_x}, bbox_y={self.bbox_y}, shape_x={self.shape_x}, shape_y={self.shape_y})"
point_load_instances = [PointLoad(node=Node(name=None, bbox_x=109, bbox_y=154, shape_x=1, shape_y=0), direction="downward", value=None, unit=None), PointLoad(node=Node(name="E", bbox_x=369, bbox_y=151.5, shape_x=5, shape_y=0), direction="downward", value=None, unit=None), PointLoad(node=Node(name="X", bbox_x=236, bbox_y=154.0, shape_x=3, shape_y=0), direction="downward", value=None, unit=None)]




for text_entry in texts_within_range:
    text = text_entry[0]
    bbox = text_entry[1]
    center_x = sum(bbox[i] for i in [0, 2, 4, 6]) / 4
    center_y = sum(bbox[i] for i in [1, 3, 5, 7]) / 4

    match = re.search(pattern, text)
    if match:
        value = match.group("value")
        unit = match.group("unit")
        load_type = match.group("type")

        # If load_type exists, then it's a distributed load, handle it accordingly
        if load_type:
            # Handle distributed load assignment here if needed
            continue

        # Find the closest PointLoad instance that hasn't been assigned yet
        closest_pl = min(
            (pl for pl in point_load_instances if pl not in assigned_pointloads),
            key=lambda pl: calculate_distance(center_x, center_y, pl.node.bbox_x, pl.node.bbox_y),
            default=None  # If all point_load_instances are already assigned
        )

        if closest_pl:
            # Assign the value and unit to the closest PointLoad instance
            closest_pl.value = value
            closest_pl.unit = unit
            assigned_pointloads.add(closest_pl)
            print(closest_pl)

