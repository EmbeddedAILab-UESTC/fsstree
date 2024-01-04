import numpy as np
import math
import os
import sys
import time
from concurrent import futures
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import random
import torch
from Node import Node
from Points import Points
from FSSTree import FSSTree





##########################   some fuction ######################################
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points














class PC_FSSTree:
    # data是点云数据(n,3),且输入要保证乱序，np.random.shuffle(data)，不然采样会不理想
    def __init__(self, data=None, level=None,dim=None,slip_mode="kde",worker_num=64,band_width=1,k=32,ek=64):
        self.data = data # 点云数据
        self.dim = dim  # 选择的分割维度[0,1,0,1,0,1,0,1]
        self.index = np.arange(0, len(data), 1) #点的编号
        self.index_map = {x:index for index,x in enumerate(self.index)}  # 点的序号{全局:局部}
        self.slip_mode = slip_mode #分割模式
        self.level = level  # 树的高度
        self.kde_core = []  # kde的核 # [(s[e_mi],s[e_mi] ,...,...] tuple()
        self.points_num = data.shape[0]  # 树中点的数量
        self.s_points_index = []  # 采样的点的索引list()
        self.s_points_parent_node_points_list_data = [] #(m,n)m是采样点数量，n是使node>ek的node的points数量
        self.s_points_parent_node_points_list_index = []
        self.s_points_parent_node_points_list_max_length = 0 #knn需要构造的大小
        self.sum_knn_search_points = []
        self.worker_num = worker_num
        self.band_width = band_width
        self.k = k
        self.ek = ek
        self.fsstree = self.createTree(data[:,:3], self.index, self.level)


    ##################################通过点云创建一颗初始树##########################################
    def createTree(self, data, index, level):
        t = FSSTree()
        self.points_num = data.shape[0]  # 点数目
        pre_sp = self.dim[0]  # 初始维度
        pre_data_list = [data]  # 上一层的数据，按层次遍历
        pre_index_list = [index]
        pre_sp_list = [pre_sp]
        for level_index in range(level):  # 一层一层创建
            current_level_node = []  # 当前层的结点集合
            current_data_list = []  # 下层的数据：记录本层划分出来的数据
            current_index_list = []  # 下层的索引：记录本层划分出来的索引
            current_sp = self.dim[level_index + 1]  # 下层的dim
            # 不是最后一层，中间结点
            if level_index < level - 1:
                # start = time.perf_counter()
                with futures.ThreadPoolExecutor(self.worker_num) as executor:  # 实例化线程池
                    # left_pc_sorted, left_index_sorted, right_pc_sorted, right_index_sorted, sp_data
                    for res in executor.map(self.slip_data, pre_data_list, pre_index_list, pre_sp_list,
                                            [level_index] * len(pre_data_list)):  # 返回的是一个生成器，需要调用next
                        # res是tuple
                        current_data_list.append(res[0])  # 左数据
                        current_data_list.append(res[2])  # 右数据
                        current_index_list.append(res[1])
                        current_index_list.append(res[3])
                        split_data = res[4]
                        split_node_data = res[5]

                        children_points_index = res[6]
                        current_level_node.append(
                            Node(split_dim=pre_sp, split_data=split_data, node_data=split_node_data,
                                 children_points_index=children_points_index))  # 把划分依据加入当前层的结点集合
                # end = time.perf_counter()
                # print(end-start)
                # 迭代数据
                pre_sp_list = [current_sp] * len(current_index_list)
                pre_index_list = current_index_list
                pre_data_list = current_data_list
                pre_sp = current_sp
                # 将本层node添加
                t.add_one_layer_node(current_level_node,self.worker_num)

            else:  # 到叶子结点
                with futures.ThreadPoolExecutor(self.worker_num) as executor:  # 实例化线程池
                    # random.shuffle(pre_index_list)
                    points_list = list(executor.map(Points, pre_index_list))  # [p,p,p,p,]
                    node_list = executor.map(Node, pre_sp_list, [None] * len(pre_sp_list), points_list,
                                             [None] * len(pre_sp_list), points_list)
                t.add_one_layer_node(node_list,self.worker_num)
        return t



    ###############################一些kde的准备工作############################################
    def kde_prepare(self, data):
        ma = max(data) + 1
        mi = min(data) - 1
        s = np.linspace(int(mi), int(ma),10)

        #自己调整带宽精度会上升,自动带宽也有影响
        #data_sp = data.reshape(-1, 1)
        #kde = KernelDensity(kernel='gaussian', bandwidth= self.band_width).fit(data_sp)
        #e = kde.score_samples(s.reshape(-1, 1))
        try:
            kde = gaussian_kde(data)
            # 进行其他操作
        except np.linalg.LinAlgError as e:
            #print(1)
            return None
        density_estimation = kde.evaluate(s)
        e_mi = argrelextrema(density_estimation, np.less)[0]
        #初始化最小面积差异和对应的波谷点索引
        min_area_diff = np.inf
        best_valley_index = None
        # 遍历波谷点，寻找面积尽量相等的波谷位置
        for valley_index in e_mi:
            # 将横坐标和密度估计值分成两部分
            left_x = s[:valley_index]
            left_density = density_estimation[:valley_index]
            right_x = s[valley_index:]
            right_density = density_estimation[valley_index:]
            # 计算两边的面积
            left_area = np.trapz(left_density, left_x)
            right_area = np.trapz(right_density, right_x)
            # 计算面积差异
            area_diff = np.abs(left_area - right_area)

            # 更新最小面积差异和对应的波谷点索引
            if area_diff < min_area_diff:
                min_area_diff = area_diff
                best_valley_index = valley_index

        # 最佳波谷点位置
        best_valley_point = s[best_valley_index]
        if best_valley_index is None:
            best_valley_point = None
        return best_valley_point



    ##############################划分数据的模式##################################################################
    def slip_data(self, data, index, sp, level_index):
        if data is None:
            return None, [-1], None, [-1], None, None, []
        if self.slip_mode == 'mid':
            return self.slip_by_mid(data, index, sp)
        elif self.slip_mode == 'kde':
            if self.level-level_index<3:
                return self.slip_by_mid(data, index, sp)
            elif len(data)<200:
                return self.slip_by_mid(data, index, sp)
            else:
                return self.slip_by_kde(data, index, sp)
        else:
            pass


    # data是待分割的集合,sp是待分割的维度
    def slip_by_kde(self, data, index, sp):
        sp_data = self.kde_prepare(data[:,sp].reshape(-1))
        # data[0,3],[0,]可能为空，或者只有一个
        if len(index) == 0:
            return None, [-1], None, [-1], None, None, []
        elif index[0] == -1:
            return None, [-1], None, [-1], None, None, []
        elif sp_data is None:
            return self.slip_by_mid(data, index, sp)
        else:
            left_flag = data[:, sp] <= sp_data  # 左边的
            right_flag = left_flag == False
            left_pc_sorted = data[left_flag]
            left_index_sorted = index[left_flag]
            right_pc_sorted = data[right_flag]
            right_index_sorted = index[right_flag]

            return left_pc_sorted, left_index_sorted, right_pc_sorted, right_index_sorted, sp_data, None, index

    def slip_by_mid(self, data, index, sp):
        if len(index) == 0:
            return None, [-1], None, [-1], None, None,  []
        elif index[0] == -1:
            return None, [-1], None, [-1], None, None,  []
        else:
            _pc_index_sorted = np.argsort(data[:, sp], kind='quicksort')
            pc_sorted = data[_pc_index_sorted]  # 排序好的点云数据
            index_sorted = index[_pc_index_sorted]  # 排序好的点云坐标
            mid = len(data) // 2
            left_pc_sorted = pc_sorted[0:mid + 1, :]
            left_index_sorted = index_sorted[0:mid + 1]
            right_pc_sorted = pc_sorted[mid + 1:, :]
            right_index_sorted = index_sorted[mid + 1:]
            sp_data = pc_sorted[mid, sp]  # <=都在左边
            sp_node_data = pc_sorted[mid]
            points_num = data.shape[0]
            return left_pc_sorted, left_index_sorted, right_pc_sorted, right_index_sorted, sp_data, sp_node_data, index


    #############################给定一个点，判断落在哪个叶子结点[xyz]###############################################
    def get_node_index_by_data(self, data):
        if self.slip_mode == 'mid':
            return self.get_node_index_by_data_type_mid(data)
        elif self.slip_mode == 'kde':
            return self.get_node_index_by_data_type_kde(data)
        else:
            pass

    def get_node_index_by_data_type_mid(self, data):
        node_num = self.fsstree.node_num  # 1000个点则，0-999
        node_list = self.fsstree.node_list
        node_index = 0
        while (2 * node_index + 1) < (node_num - 1):  # 没到叶子结点
            current_node = node_list[node_index]
            split_dim = current_node.split_dim
            split_data = current_node.split_data
            if split_data is None:
                node_index = 2 * node_index + 1
            else:
                if data[split_dim] <= split_data:
                    node_index = 2 * node_index + 1
                else:
                    node_index = 2 * node_index + 2
        return node_index

    def get_node_index_by_data_type_kde(self, data):
        node_num = self.fsstree.node_num  # 1000个点则，0-999
        node_list = self.fsstree.node_list
        node_index = 0
        while (2 * node_index + 1) < (node_num - 1):
            current_node = node_list[node_index]
            split_dim = current_node.split_dim
            split_data = current_node.split_data
            if split_data is None:
                node_index = 2 * node_index + 1
            # s = time.perf_counter()
            else:
                d = data[split_dim]
                if d <= split_data:
                    node_index = 2 * node_index + 1
                else:
                    node_index = 2 * node_index + 2
            # e = time.perf_counter()
            # print("vq time%f"%(e-s))
        return node_index



    #######################################knn的模式########################################
    def fsstree_knn(self,sampled_data,k=None,device="cpu"):  # sample_points #(2048,1,3)
        if k:
            k = k
        else:
            k = self.k
        fs_data = torch.from_numpy(sampled_data).reshape(len(self.s_points_index), 1, 3).float().to(device)
        fs_all_data = torch.cat(self.s_points_parent_node_points_list_data,dim=0).float().to(device)
        group_idx = knn_point(self.k, fs_all_data, fs_data)
        group_idx = group_idx.reshape(-1,self.k) #(s,k)
        idx = torch.gather(torch.tensor(np.array(self.s_points_parent_node_points_list_index)).to(device),dim=1,index = group_idx)
        return idx #(s,1,k)



    ################################################some function############################################
    def get_parent_node_index(self, index):
        return ((index + 1) // 2) - 1

    def get_brother_node_index(self, index):
        if index % 2 == 1:
            return index + 1
        else:
            return index - 1

    ############################################得到一个点的所有叶子结点############################################
    def get_all_leaf_node_by_node_index(self, index):
        if self.slip_mode == 'mid':
            return self.get_all_leaf_node_by_node_index_mid(index)
        elif self.slip_mode == 'kde':
            return self.get_all_leaf_node_by_node_index_kde(index)
        else:
            pass

    def get_all_leaf_node_by_node_index_mid(self, index):
        node_num = self.fsstree.node_num
        leaf_node_list = []
        # 如果已经是laefnode
        if (2 * index) + 1 > node_num - 1:
            leaf_node_list.append(index)
            return leaf_node_list
        children_node_list = [index]
        while len(children_node_list) != 0:
            x = int(children_node_list.pop())
            if (2 * x) + 1 > node_num - 1:
                leaf_node_list.append(x)
            else:
                if self.fsstree.node_list[x].split_data is None:
                    children_node_list.append(int(2 * x + 1))
                else:
                    children_node_list.append(int(2 * x + 1))
                    children_node_list.append(int(2 * x + 2))
        return leaf_node_list

    def get_all_leaf_node_by_node_index_kde(self, index):
        node_num = self.fsstree.node_num
        leaf_node_list = []
        if (2 * index) + 1 > node_num - 1:
            leaf_node_list.append(index)
            return leaf_node_list
        children_node_list = [index]
        while len(children_node_list) != 0:
            x = int(children_node_list.pop())
            if (2 * x) + 1 > node_num - 1:
                leaf_node_list.append(x)
            else:
                if self.fsstree.node_list[x].split_data is None:
                    children_node_list.append(int(2 * x + 1))
                else:
                    children_node_list.append(int(2 * x + 1))
                    children_node_list.append(int(2 * x + 2))
        return leaf_node_list


    ####################################修剪最后一层，往上浮动###########################################
    def delete_layer(self):
        self.s_points_parent_node_points_list_index = []
        self.s_points_parent_node_points_list_data = []
        for index, points_index in enumerate(self.s_points_index):
            points_data = self.data[points_index]
            node_index = self.get_node_index_by_data_type_kde(points_data)
            parent_index = self.get_parent_node_index(node_index)
            if self.fsstree.node_list[parent_index].points is None:
                self.fsstree.node_list[parent_index].points = Points([points_index])
            else:
                self.fsstree.node_list[parent_index].points.add_point(points_index)
            self.fsstree.node_list[parent_index].children_points_index = []
        self.level = self.level - 1
        self.points_num = len(self.s_points_index)
        self.fsstree.node_num = (2 ** self.level) - 1
        self.fsstree.node_list = self.fsstree.node_list[:self.fsstree.node_num]
        self.fsstree.layer = self.fsstree.layer - 1
        self.update_children_points_index()
        self.index = self.s_points_index
        self.index_map = {x:index for index,x in enumerate(self.index)}  # 点的序号

    # 修剪后需要更新分支结点的children_points_index，含有的子树的点序号列表
    def update_children_points_index(self):
        last_2level_first_index = (2 ** (self.level - 2)) - 1
        last_2level_last_index = (2 ** (self.level - 1)) - 2
        for i in range(last_2level_last_index, last_2level_first_index - 1, -1):
            result = []
            if self.fsstree.node_list[2 * i + 1].points is None:
                pass
            else:
                result.extend(self.fsstree.node_list[2 * i + 1].points.points_index)
            if self.fsstree.node_list[2 * i + 2].points is None:
                pass
            else:
                result.extend(self.fsstree.node_list[2 * i + 2].points.points_index)
            self.fsstree.node_list[i].children_points_index = result
        for j in range(last_2level_first_index - 1, 0, -1):
            result1 = []
            result1.extend(self.fsstree.node_list[2 * j + 1].children_points_index)
            result1.extend(self.fsstree.node_list[2 * j + 2].children_points_index)
            self.fsstree.node_list[j].children_points_index = result1
        one = self.fsstree.node_list[1].children_points_index
        two = self.fsstree.node_list[2].children_points_index
        one.extend(two)
        self.fsstree.node_list[0].children_points_index = one

    ########################################采样操作###################################################
    # 采样，对每个叶子结点采样
    def sampled_points(self, nsample):

        self.s_points_parent_node_points_list_index = []
        self.s_points_parent_node_points_list_data = []
        self.s_points_index = []
        self.sum_knn_search_points =[]
        fsstree_level = self.level
        fsstree_points_num = self.points_num
        # per_points = fsstree_points_num // nsample
        last_level_first_index = (2 ** (fsstree_level - 1)) - 1
        last_level_last_index = (2 ** fsstree_level) - 2
        index_list = list(range(last_level_first_index, last_level_last_index + 1))
        # l_index = -1
        # print(index_list)
        while (len(self.s_points_index) < nsample):
            # l_index = l_index + 1
            for i in index_list: #i是node编号
                p = self.fsstree.node_list[i].points
                if (len(self.s_points_index)) < nsample:
                    if p is None:
                        # print("null")
                        pass
                    else:
                        if len(p.points_index) == 0 or p.points_index[0] == -1:
                            # print("n")
                            pass
                        else:
                            p_index = p.points_index
                            s_index = p_index[
                                random.randint(0, len(p.points_index) - 1) if (len(p.points_index) - 1) > 0 else 0]
                            self.s_points_index.append(s_index)
                            node_index = i #当前node编号
                            parent_index = self.get_parent_node_index(node_index)
                            while len(self.fsstree.node_list[parent_index].children_points_index) <= self.ek:
                                if parent_index==0: break #所有点都不够了，就穷尽
                                parent_index = self.get_parent_node_index(parent_index)
                            redundant_index = self.fsstree.node_list[parent_index].children_points_index
                            self.sum_knn_search_points.append(len(redundant_index))

                            #####

                            redundant_index = np.random.choice(np.array(redundant_index), size=self.ek, replace=True)
                                #redundant_index = random.sample(list(redundant_index), self.ek)
                                #redundant_index = redundant_index[:self.ek] #保证乱序的情况下就行
                            redundant_data = torch.from_numpy(self.data[redundant_index, :3].reshape(1, -1, 3))
                            self.s_points_parent_node_points_list_index.append(redundant_index)
                            self.s_points_parent_node_points_list_data.append(redundant_data)



    #####################################################工具类#############################################
    def get_empty_node_num(self, sample_index):
        fsstree_level = self.level
        last_level_first_index = (2 ** (fsstree_level - 1)) - 1
        last_level_last_index = (2 ** fsstree_level) - 2
        index_list = list(range(last_level_first_index, last_level_last_index + 1))
        l = 0
        if sample_index == 0:
            for i in index_list:
                p = self.fsstree.node_list[i].points
                if len(p.points_index) < 2:
                    l = l + 1
        else:
            for i in index_list:
                p = self.fsstree.node_list[i].points
                if p is None:
                    l = l + 1
        return l








if __name__ == '__main__':


    n = 80000
    data = np.random((n,3))
    e = 3
    k = 32
    # 高度也可以自己调节，超参数
    h = math.floor(math.log((data.shape[0] / (math.log(data.shape[0],10)* self.k)), 2)) + 1
    dim = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,1,2,0,1,2]
    t = PC_FSSTree(data=data, slip_mode="kde", level=h , dim = dim,k=k,ek=e*k)
    sampled_nums = [2048,1024]
    for samp in sampled_nums:
        t.sampled_points(samp)
        fsst_sample_data = np.array(t.data[t.s_points_index]).astype(np.float32)  # (2048,3)
        sampled_index = t.s_points_index #list
        sampled_data_numpy = t.data[t.s_points_index]
        knn_reesult = t.fsstree_knn(sampled_data_numpy,k=k)
        t.delete_layer()

   