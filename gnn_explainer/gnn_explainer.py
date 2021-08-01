import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
"""
GNN-Explainer
Ying et al. NeurIPS 2019

Implemented by
Yonsei App.Stat.
Sunwoo Kim
"""


class gnn_explainer(torch.nn.Module):

    def __init__(self, data, GNN, device):
        super(gnn_explainer, self).__init__()
        """
        data should be given as pytorch geometric data shape
        GNN_model should be given as pytorch geometric model
        This model assumes a single layer
        """
        self.A = torch.zeros((data.x.shape[0], data.x.shape[0])).to(device)
        self.feat = data.x.reshape(1, data.x.shape[0], data.x.shape[1]).to(device)
        self.GNN = GNN
        self.GNN.convs[0].lin_rel.weight.requires_grad = False
        self.GNN.convs[0].lin_root.weight.requires_grad = False
        self.GNN.convs[0].lin_root.bias.requires_grad = False
        self.GNN.read_out.weight.requires_grad = False
        self.GNN.read_out.bias.requires_grad = False

        ## Defining adjacency matrix that will not be touched
        self.A[(data.edge_index[0], data.edge_index[1])] = 1

        # Adjacency matrix masking function
        self.adj_masking = torch.normal(mean=0, std=0.01, size=(self.A.shape[0], self.A.shape[1]),
                                        requires_grad=True).to(device)
        self.adj_masking = (self.adj_masking + torch.transpose(self.adj_masking, 1, 0)) / 2
        # Feature matrix masking fucntion
        self.feat_masking = torch.normal(mean=0, std=0.01, size=(1, self.feat.shape[1]),
                                         requires_grad=True).to(device)

    def reset_parameters(self):  # If we want to reset masking function's value
        # Adjacency matrix masking function
        self.adj_masking = torch.normal(mean=0, std=0.1, size=(self.A.shape[0], self.A.shape[1]),
                                        requires_grad=True)

    def forward(self):
        self.adj_masking.retain_grad()
        adj = self.A * torch.sigmoid(self.adj_masking)  # Ac * sigma(M)
        y = self.GNN(data=(self.feat, adj),
                     training_with_batch=False,
                     in_shape="adj")
        return y


def result_generator(data, feature_mask, adj_mask, threshold):
    sigmoid = torch.nn.Sigmoid()
    A = np.zeros((data.x.shape[0], data.x.shape[0]))
    A[(data.edge_index[0], data.edge_index[1])] = 1
    A_mask = sigmoid(adj_mask.to("cpu").detach()).numpy()
    # feat_mask = sigmoid(feature_mask.to("cpu").detach()).numpy()

    new_A = A * A_mask
    # new_x = data.x.numpy()*feat_mask
    new_A = (new_A + new_A.transpose()) / 2
    edge_list = []
    for i in range(data.x.shape[0]):
        for s in range(data.x.shape[0]):
            if new_A[i, s] > threshold:
                edge_list.append((i, s))

    G = nx.Graph()
    G.add_nodes_from([i for i in range(data.x.shape[0])])
    G.add_edges_from(edge_list)
    color_map = []
    for i in range(int(data.x.shape[0] - 5)):
        color_map.append("orange")
    color_map.append("red")
    color_map.append("green");
    color_map.append("green")
    color_map.append("green");
    color_map.append("green")
    nx.draw(G, node_color=color_map, with_labels=True,
            pos=nx.kamada_kawai_layout(G))  # , pos=nx.kamada_kawai_layout(G))
    plt.show()