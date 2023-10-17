



class Frame:
    def __init__(self, node1=None, node2=None, obj_det_bbox=None, direction=None):
        self.node_1 = node1
        self.node_2 = node2
        self.obj_det_bbox = obj_det_bbox
        self.direction = direction

    def __repr__(self):
        return f"Frame(node_1={repr(self.node_1)}, node_2={repr(self.node_2)}, obj_det_bbox={self.obj_det_bbox}, direction={self.direction})"