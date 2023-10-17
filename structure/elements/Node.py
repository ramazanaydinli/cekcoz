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