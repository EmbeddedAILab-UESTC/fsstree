
#树的结点结构
class Node:
    def __init__(self, split_dim=0, split_data=None, points=None , node_data = None,children_points_index=[]):
        self.split_dim = split_dim  # 0是按特征1分割，1是按特征2分割
        self.split_data = split_data #分割特征的值
        self.node_data = node_data #分割的点
        self.points = points #如果是leaf则放points
        self.children_points_index = children_points_index #子树中所有点的序号

    def get_points(self):
        return self.points
    def get_split_dim(self):
        return self.split_dim
    def get_split_data(self):
        return self.split_data
    def get_node_data(self):
        return self.node_data
    def set_points(self,points):
        self.points = points