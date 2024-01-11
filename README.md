# Accelerate Point Cloud Structuring for Deep Neural Networks via Fast Spatial-Searching Tree

Created by Jinyu Zhan, Shiyu Zou, Wei Jiang, Youyuan Zhang, Suidi Peng, Ying Wang

![](pics/pipeline.png)

# Introduction

We design a novel Fast Spatial-Searching Tree (FSSTree) to accelerate point cloud structuring for DNNs on embedded devices. The FSSTree is constructed based on density distribution of point clouds,which can guarantee that points with similar spatial positions are stored in adjacent storage sets. Based on FSSTree, we propose a point-sparsity-aware sampling method and a leafwise k-nearest neighbor query method to reduce the computation overhead of structuring point clouds. Meanwhile, the point sparsity-aware sampling method achieves fair sampling on both dense and sparse parts, which can overcome the nonuniform distribution of point clouds caused by occlusion, lighting and other factors. The leafwise k-nearest neighbor query method skips a large number of dissimilar points to quickly obtain the neighbor points, which can significantly reduce the search scope. We also present a layerwise self-pruning algorithm to automatically adjust the FSSTree after each layer operation of DNNs. Finally, we conduct extensive experiments on KITTI, S3DIS and ModelNet40 datasets and three devices (including a RTX 3090 server, a Jetson AGX Xavier and an Apple M2). The experimental results demonstrate the efficiency of our approach, which can reduce the time overhead by up to 97.46% compared with the traditional combination of Farthest Point Sampling (FPS) and k-Nearest Neighbor (KNN).


# Usage

The main functions are in PC_FSSTree.py. 

### To create a Fsstree

```
data = np.random((80000,3))           # A numpy array

e = 3                                 # The expansion coefficient

k = 256                               # The query size

# The height of the tree. Of course this is a hyperparameter and the formula is just the size we recommend.
# You can also adjust it manually to make the task work best.
h = math.floor(math.log((data.shape[0] / (math.log(data.shape[0],10)* self.k)), 2)) + 1

# The partition dimension selected at each level of the Fsstree.
dim = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,1,2,0,1,2]

# To create a Fsstree.
t = PC_FSSTree(data=data, slip_mode="kde", level=h , dim = dim,k=k,ek=e*k)
```

### Sampling

```
samp = int(data.shape[0]/4)              # The number of sampled points.

t.sampled_points(samp)                   # Sampling

fsst_sample_data_numpy = np.array(t.data[t.s_points_index]).astype(np.float32)  

```


### Neighbor Query

```
k = 256

neighbor_query_reesult = t.fsstree_knn(fsst_sample_data_numpy,k=k)

```


### After each layer , Remember to Self-Pruning the tree

```
t.delete_layer()

```
