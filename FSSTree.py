from concurrent import futures

#树的结构
class FSSTree:
    def __init__(self,node_num=0,layer=0):
        self.node_num = node_num #所有结点数量，也指代最后要插入结点索引
        self.node_list = []
        self.layer = layer #有几层
    #添加一个结点在后面
    def add_one_node(self,node):
        self.node_list.append(node)
        self.node_num = self.node_num+1
    #保证满二叉树，每次添加一层,有序
    def add_one_layer_node(self,layer_node_list,worker_num):
        with futures.ThreadPoolExecutor(worker_num) as executor:  # 实例化线程池
            _ = executor.map(self.add_one_node, layer_node_list)
        self.layer = self.layer +1
    def get_node_by_index(self,index):
        return self.node_list[index]