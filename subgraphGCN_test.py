import networkx as nx
import numpy as np
import synthetic_structsim
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors
import pdb
import random
import abc


# 对给定的图列表进行扰动，即通过添加边来改变图的结构
def perturb(graph_list, p):

    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


# 图生成部分
def generate_graph(basis_type="ba",
                   shape="house",
                   nb_shapes=80,  # 指定在图中有多少此类形状。
                   width_basis=300,  # 定义基础图的大小。
                   feature_generator=None,
                   m=5,
                   random_edges=0.0):
    if shape == "house":
        list_shapes = [["house"]] * nb_shapes
    elif shape == "cycle":
        list_shapes = [["cycle", 6]] * nb_shapes
    elif shape == "diamond":
        list_shapes = [["diamond"]] * nb_shapes
    elif shape == "grid":
        list_shapes = [["grid"]] * nb_shapes
    else:
        assert False

    # 生成合成图
    G, role_id, _ = synthetic_structsim.build_graph(width_basis,
                                                    basis_type,
                                                    list_shapes,
                                                    rdm_basis_plugins=True,
                                                    start=0,
                                                    m=m)

    if random_edges != 0:
        G = perturb([G], random_edges)[0]
    # feature_generator.gen_node_features(G)
    return G, role_id


G, role_id = generate_graph()

# Compute features for each node
features_dict = {
    'node_index': [],
    'degree': [],
    'clustering': [],
    'betweenness': [],
    'closeness': [],
    # 'role': [],
    'generated_features': [],
}
betweenness_dict = nx.betweenness_centrality(G)
closeness_dict = nx.closeness_centrality(G)
np.random.seed(0)  # Seed for reproducibility
generated_features = np.random.rand(G.number_of_nodes())

# Populate the features for each node
for i, node in enumerate(G.nodes()):
    features_dict['node_index'].append(i)
    features_dict['degree'].append(G.degree[node])
    features_dict['clustering'].append(nx.clustering(G, node))
    features_dict['betweenness'].append(betweenness_dict[node])
    features_dict['closeness'].append(closeness_dict[node])
    # features_dict['role'].append(role_id[i])
    features_dict['generated_features'].append(generated_features[i])

features_matrix = np.column_stack(list(features_dict.values()))
print(features_matrix.shape)  # the shape of the matrix U (700 x n)

# print feature names and the first few rows of the matrix U to see the detailed information
feature_names = list(features_dict.keys())
feature_matrix_sample = features_matrix[:5]
print(feature_names, feature_matrix_sample)

# k-hop采样生成子图
k = 2
num_subgraphs = 700
subgraphs_info = []

np.random.seed(0)
center_nodes = np.random.choice(G.nodes(), num_subgraphs, replace=False)

# 为每个中心节点生成子图
for center_node in center_nodes:
    # 获取k-hop邻居节点
    subgraph_nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=k).keys()
    subgraph = G.subgraph(subgraph_nodes)
    subgraphs_info.append({
        'center_node': center_node,
        'nodes': list(subgraph.nodes()),
        'node_count': subgraph.number_of_nodes(),
    })


# 打印前10个子图的节点个数
for i, subgraph in enumerate(subgraphs_info[:10]):
    print(f"Subgraph {i+1} (Center Node: {subgraph['center_node']}): {subgraph['node_count']} nodes")

# 打印任意一个子图的全部节点编号以及对应的特征编码向量
sample_subgraph = subgraphs_info[10]
sample_subgraph_nodes = sample_subgraph['nodes']
sample_node_features = features_matrix[sample_subgraph_nodes]

print(sample_subgraph, sample_node_features)


# test
from sklearn.preprocessing import StandardScaler
features_to_normalize = features_matrix[:, 1:]
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features_to_normalize)

# 重新组合归一化后的特征和node_index
features_matrix = np.column_stack((features_matrix[:, 0], normalized_features))

def extract_subgraph_features(subgraph_info, features_matrix):
    indices = subgraph_info['nodes']
    return features_matrix[indices]

# 示例：提取第一个子图的特征
sample_subgraph_features = extract_subgraph_features(subgraphs_info[0], features_matrix)
print("Sample Subgraph Features:\n", sample_subgraph_features)


from sklearn.model_selection import train_test_split

# 所有子图的索引
all_subgraphs_indices = list(range(len(subgraphs_info)))

train_indices, test_indices = train_test_split(all_subgraphs_indices, test_size=0.2, random_state=0)
test_indices, val_indices = train_test_split(test_indices, test_size=0.5, random_state=0)

train_subgraphs = [subgraphs_info[i] for i in train_indices]
val_subgraphs = [subgraphs_info[i] for i in val_indices]
test_subgraphs = [subgraphs_info[i] for i in test_indices]

print("训练集子图数量:", len(train_subgraphs))
print("验证集子图数量:", len(val_indices))
print("测试集子图数量:", len(test_subgraphs))

import torch
from torch_geometric.data import Data

# features_matrix 和 G
edge_index = torch.tensor(list(map(list, G.edges())), dtype=torch.long).t().contiguous()
x = torch.tensor(features_matrix, dtype=torch.float)

# 节点标签
y = torch.tensor(role_id, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)


num_nodes = data.num_nodes
train_size = int(0.7 * num_nodes)
val_size = int(0.15 * num_nodes)
test_size = num_nodes - (train_size + val_size)

indices = torch.randperm(num_nodes)
train_mask = indices[:train_size]
val_mask = indices[train_size:train_size+val_size]
test_mask = indices[train_size+val_size:]

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


from torch_geometric.loader import DataLoader

batch_size = 32
loader = DataLoader([data], batch_size=batch_size, shuffle=True)


from torch_geometric.nn import GCNConv
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc1(x))


class SubgraphGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, input_dim, hidden_dim, output_dim, mask_output_dim=None):
        super(SubgraphGCN, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.mlp = MLP(input_dim , mask_output_dim)

    def forward(self, batch_data, center_node_indices):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch

        x_core_agg = []
        x_redundant_agg = []

        for i in range(batch.max().item() + 1):  # 遍历批次中的每个子图
            mask = batch == i
            subgraph_nodes = mask.nonzero(as_tuple=False).squeeze()
            center_node_idx = center_node_indices[i]  # 获取中心节点索引

            # 确保边索引只包含当前子图的节点
            subgraph_edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            subgraph_edge_index = edge_index[:, subgraph_edge_mask]

            subgraph_data = Data(x=x[subgraph_nodes], edge_index=subgraph_edge_index)

            center_embedding = self.conv1(subgraph_data.x, subgraph_data.edge_index)[center_node_idx]

            if center_embedding.dim() != 1:
                center_embedding = center_embedding.squeeze()

            mask_vectors = []
            for j in range(subgraph_data.num_nodes):
                if j != center_node_idx:
                    node_features = subgraph_data.x[j]

                    combined_features = torch.cat([center_embedding, node_features])
                    mask_vector = self.mlp(combined_features)
                    mask_vectors.append(mask_vector)

            if mask_vectors:
                mask_matrix = torch.stack(mask_vectors).squeeze()
            else:
                mask_matrix = torch.zeros(subgraph_data.num_nodes, 1, device=subgraph_data.x.device)

            # 应用掩码，并排除中心节点
            non_center_mask = torch.tensor([k != center_node_idx for k in range(subgraph_data.num_nodes)], device=subgraph_data.x.device)
            x_core_masked = mask_matrix * subgraph_data.x[non_center_mask]
            x_redundant_masked = (1 - mask_matrix) * subgraph_data.x[non_center_mask]

            x_core_agg.append(self.conv2(x_core_masked, subgraph_data.edge_index))
            x_redundant_agg.append(self.conv3(x_redundant_masked, subgraph_data.edge_index))

        x_core_agg = torch.mean(torch.stack(x_core_agg), dim=0)
        x_redundant_agg = torch.mean(torch.stack(x_redundant_agg), dim=0)

        return x_core_agg, x_redundant_agg


import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

num_features = data.num_features
num_classes = 4
input_dim = 6
hidden_dim = 64
output_dim = 4
mask_output_dim = 6
model = SubgraphGCN(num_features, num_classes,
                    input_dim, hidden_dim, output_dim, mask_output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 准备中心节点索引
center_node_indices = []
for subgraph_info in subgraphs_info:
    # 创建映射：原始节点ID到子图索引
    node_to_index = {node_id: idx for idx, node_id in enumerate(subgraph_info['nodes'])}
    # 获取中心节点ID
    center_node_id = subgraph_info['center_node']
    # 将中心节点ID映射到其在子图中的索引
    center_node_idx = node_to_index[center_node_id]
    center_node_indices.append(center_node_idx)


import torch
from torch_geometric.data import Data

subgraph_data_list = []  # 存储所有子图的列表

# 对每个子图进行处理
for subgraph_info in subgraphs_info:
    # 提取子图信息
    center_node = subgraph_info['center_node']
    subgraph_nodes = subgraph_info['nodes']
    subgraph = G.subgraph(subgraph_nodes)

    # 创建从原始图到子图的节点索引映射（仅用于边）
    node_index_map = {node: i for i, node in enumerate(subgraph.nodes())}

    # 更新edge_index
    subgraph_edge_index = []
    for u, v in subgraph.edges():
        subgraph_edge_index.append([node_index_map[u], node_index_map[v]])
    subgraph_edge_index = torch.tensor(subgraph_edge_index, dtype=torch.long).t().contiguous()

    # 获取子图的特征矩阵
    subgraph_features = torch.tensor(features_matrix[subgraph_nodes], dtype=torch.float)

    # 添加中心节点的索引
    center_node_idx = subgraph_nodes.index(center_node)  # 获取中心节点在子图中的局部索引
    subgraph_data = Data(x=subgraph_features, edge_index=subgraph_edge_index, center_idx=center_node_idx)
    subgraph_data_list.append(subgraph_data)


subgraph_loader = DataLoader(subgraph_data_list, batch_size=batch_size, shuffle=True)


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data in loader:
            center_idx = [data.center_idx for data in batch_data.to_data_list()]
            x_core_agg, x_redundant_agg = model(batch_data, center_idx)
            y = torch.tensor([role_id[data.center_idx] for data in batch_data.to_data_list()], dtype=torch.long)
            loss = criterion(x_core_agg, y) + criterion(x_redundant_agg, y)
            total_loss += loss.item()
            predictions = x_core_agg.max(1)[1]
            total_correct += predictions.eq(y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

val_loader = DataLoader([subgraphs_info[i] for i in val_indices], batch_size=batch_size)
test_loader = DataLoader([subgraphs_info[i] for i in test_indices], batch_size=batch_size)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data in subgraph_loader:
        center_idx = [data.center_idx for data in batch_data.to_data_list()]
        #center_idx = batch_data.center_idx

        optimizer.zero_grad()
        x_core_agg, x_redundant_agg = model(batch_data, center_idx)

        # 假设 labels 是节点标签
        y = torch.tensor(role_id, dtype=torch.long)
        loss = criterion(x_core_agg, y) + criterion(x_redundant_agg, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
    print(f"Epoch {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")