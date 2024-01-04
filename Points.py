import random


#结点里面存放点的数据结构
class Points:
    def __init__(self,points_index =[]):
        self.points_index = points_index #点云的序号
    def get_points_index(self):
        return self.points_index
    def add_point(self,point_index):
        self.points_index.append(point_index)
    def add_point_list(self,point_index_list):
        self.points_index.extend(point_index_list)
    def get_points_num(self):
        return len(self.points_index)
    def shuffle_points(self):
        randnum = random.randint(0, 10)
        random.seed(randnum)
        random.shuffle(self.points_index)