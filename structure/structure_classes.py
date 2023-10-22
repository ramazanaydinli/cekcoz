
class Dimension:
    def __init__(self, node_1=None, node_2=None, value=None, unit=None, obj_det_bbox=None, text_bbox=None):
        self.node_1 = node_1
        self.node_2 = node_2
        self.value = value
        self.unit = unit
        self.obj_det_bbox = obj_det_bbox
        self.text_bbox = text_bbox

    def __repr__(self):
        return (f"Dimension(node_1={self.node_1}, node_2={self.node_2}, value={self.value}, "
                f"unit={self.unit}, obj_det_bbox={self.obj_det_bbox}, text_bbox={self.text_bbox})")


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
        return f"Node(name={self.name}, bbox_x={self.bbox_x}, bbox_y={self.bbox_y}, shape_x={self.shape_x}," \
               f" shape_y={self.shape_y})"


class FixSupport:
    def __init__(self, node=None, fx=0, fy=0, moment=0):
        self.node = node
        self.fx = fx
        self.fy = fy
        self.moment = moment

    def __repr__(self):
        return f"FixSupport(node={self.node}, Fx={self.fx}, Fy={self.fy}, Moment={self.moment})"


class PinSupport:
    def __init__(self, node=None, fx=0, fy=0):
        self.fx = fx
        self.fy = fy
        self.node = node

    def __repr__(self):
        return f"PinSupport(node={self.node}, Fx={self.fx}, Fy={self.fy})"


class RollerSupport:
    def __init__(self, node=None, fy=0):
        self.fy = fy
        self.node = node

    def __repr__(self):
        return f"RollerSupport(node={self.node}, Fy={self.fy})"


class Frame:
    def __init__(self, node1=None, node2=None, obj_det_bbox=None, direction=None):
        self.node_1 = node1
        self.node_2 = node2
        self.obj_det_bbox = obj_det_bbox
        self.direction = direction

    def __repr__(self):
        return f"Frame(node_1={repr(self.node_1)}, node_2={repr(self.node_2)}, obj_det_bbox={self.obj_det_bbox}," \
               f" direction={self.direction})"


class PointLoad:
    def __init__(self, node=None, direction=None, value=None, unit=None, text_bbox=None, obj_det_bbox=None):
        self.node = node
        self.direction = direction
        self.value = value
        self.unit = unit
        self.text_bbox = text_bbox
        self.obj_det_bbox = obj_det_bbox

    def __repr__(self):
        return f"PointLoad(node={self.node}, direction={self.direction}, value={self.value}, unit={self.unit}," \
               f" text_bbox = {self.text_bbox}, obj_det_bbox = {self.obj_det_bbox})"

class DistributedLoad:
    def __init__(self, node1=None, node2=None,obj_det_bbox = None, direction=None, value=None, unit=None):
        self.node1 = node1
        self.node2 = node2
        self.obj_det_bbox = obj_det_bbox
        self.direction = direction
        self.value = value
        self.unit = unit

    def __repr__(self):
        return f"DistributedLoad(node1={self.node1}, node2={self.node2}, obj_det_bbox={self.obj_det_bbox}, direction={self.direction}, value={self.value}, unit={self.unit})"

