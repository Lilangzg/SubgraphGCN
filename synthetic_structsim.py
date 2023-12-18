import math
import networkx as nx
import numpy as np
import random
import pdb

# 创建一个clique图（完全图）,然后在需要时从中随机移除一些边。
def clique(start, nb_nodes, nb_to_remove=0, role_start=0):
    a = np.ones((nb_nodes, nb_nodes))
    np.fill_diagonal(a, 0)
    graph = nx.from_numpy_matrix(a)
    edge_list = graph.edges().keys()
    roles = [role_start] * nb_nodes
    if nb_to_remove > 0:
        lst = np.random.choice(len(edge_list), nb_to_remove, replace=False)
        print(edge_list, lst)
        to_delete = [edge_list[e] for e in lst]
        graph.remove_edges_from(to_delete)
        for e in lst:
            print(edge_list[e][0])
            print(len(roles))
            roles[edge_list[e][0]] += 1
            roles[edge_list[e][1]] += 1
    mapping_graph = {k: (k + start) for k in range(nb_nodes)}
    graph = nx.relabel_nodes(graph, mapping_graph)
    return graph, roles


# 创建一个循环图（cycle graph）
def cycle(start, len_cycle, role_start=0):

    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + len_cycle))
    for i in range(len_cycle - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    graph.add_edges_from([(start + len_cycle - 1, start)])
    roles = [role_start] * len_cycle
    return graph, roles


def tree(start, height, r=10, role_start=0):  # r: 每个节点的分支数，即子节点的数量。

    graph = nx.balanced_tree(r, height)
    roles = [0] * graph.number_of_nodes()
    return graph, roles

# 创建一个Barabási–Albert(BA)模型的图。
# Barabási–Albert模型是一个随机图生成模型，用于产生尺度无关网络（scale-free networks）。
# 这种网络的一个特点是它们的某些节点（称为"枢纽"节点）比其他节点有更多的连接。
def ba(start, width, role_start=0, m=5):

    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    return graph, roles

# 创建一个形状类似于钻石的图。
def diamond(start, role_start=0):

    len_cycle = 6
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + len_cycle))
    for i in range(len_cycle - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    graph.add_edges_from([(start + len_cycle - 1, start)])
    graph.add_edges_from([(start + len_cycle - 1, start + 1)])
    graph.add_edges_from([(start + len_cycle - 2, start + 2)])
    roles = [role_start] * len_cycle
    return graph, roles


# 生成一个类似于房屋的图结构
def house(start, role_start=0):

    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    return graph, roles


# 创建了一个2x3的网格图（grid graph）
def grid(start, dim, r=10, role_start=0):

    grid_G = nx.grid_graph([3, 2])
    grid_G = nx.convert_node_labels_to_integers(grid_G, first_label=start)
    roles = [role_start for i in grid_G.nodes()]
    return grid_G, roles


# 从一个基本图（basis graph）开始，再附加上一些预定义的图形结构（shapes）。创建一个具有特定拓扑结构和角色的复杂图。
def build_graph(
        width_basis,  # 基本图的宽度（节点的数量）
        basis_type,  # 基图的类型，可以是 torus（环面）、string（字符串）或 cycle（循环）
        list_shapes,
        start=0,
        rdm_basis_plugins=False,
        add_random_edges=0,
        m=5,
):

    if basis_type == "ba":
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis, r=m)

    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motifs
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(n_basis / n_shapes)  # n_shapes: 80
        plugins = [int(k * spacing) for k in range(n_shapes)]
    seen_shapes = {"basis": [0, n_basis]}  # n_basis=300

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])  # attach
        # if shape_type == "cycle":
        #     if np.random.random() > 0.5:
        #         a = np.random.randint(1, 4)
        #         b = np.random.randint(1, 4)
        #         basis.add_edges_from([(a + start, b + plugins[shape_id])])
        temp_labels = [r + col_start for r in roles_graph_s]
        # temp_labels[0] += 100 * seen_shapes[shape_type][0]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
            print(src, dest)
            basis.add_edges_from([(src, dest)])

    return basis, role_id, plugins

