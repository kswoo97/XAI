import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GCNConv, DenseSAGEConv
"""
Implementing House Shape Classifier dataset Classifier
Yonsei App.Stat. Sunwoo Kim
"""

class house_classifier(torch.nn.Module) :

    def __init__(self, dataset, gconv = DenseSAGEConv, device = "cuda", latent_dim = [16]) :
        super(house_classifier, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.device = device
        self.latent = latent_dim

        self.convs.append(
            gconv(in_channels=dataset.num_features,
                  out_channels=latent_dim[0]))
        self.read_out = torch.nn.Linear(latent_dim[-1], 1)

    def reset_parameters(self) :
        for conv_layer in self.convs :
            conv_layer.reset_parameters()
        self.read_out.reset_parameters()

    def data_reassignment(self, data, batchwise):
        if batchwise:
            full_x, full_edge, batch_info = data.x, data.edge_index, data.batch
            A = torch.zeros((full_x.shape[0], full_x.shape[0]))
            A[(full_edge[0], full_edge[1])] = 1
            batch_num, batch_count = torch.unique(batch_info, return_counts=True)  # 유니크 개수, 각 배치 별 노드 수
            max_n = torch.max(batch_count).to("cpu").item()
            full_A = torch.zeros((batch_num.shape[0], max_n, max_n)).to(self.device)
            X = torch.zeros((batch_num.shape[0], torch.max(batch_count).to("cpu").item(), data.x.shape[1])).to(
                self.device)
            cummul = 0
            for i in range(X.shape[0]):
                current_n = batch_count[i]
                X[i, :current_n, :] = full_x[cummul:(cummul + current_n), :]  # Stacking X
                full_A[i, :current_n, :current_n] = A[cummul:(cummul + current_n),
                                                    cummul:(cummul + current_n)]  # Stacking Adj
                cummul += current_n
        else:
            X = data.x.reshape(1, data.x.shape[0], data.x.shape[1])
            full_A = torch.zeros((1, data.x.shape[0], data.x.shape[0])).to(self.device)
            full_A[(0, data.edge_index[0], data.edge_index[1])] = 1
        return X, full_A

    def forward(self, data, training_with_batch, in_shape):

        if in_shape == "adj" :
            new_x, new_A = data[0], data[1].reshape(1, data[1].shape[0], data[1].shape[0])
            x = new_x
            for conv in self.convs:
                x = torch.relu(conv(x=x, adj=new_A))
            x = torch.sum(x, dim=1)
            x = self.read_out(x)
            return torch.sigmoid(x)

        else :
            if training_with_batch:
                new_x, new_A = self.data_reassignment(data, batchwise=True)
                x = new_x
                for conv in self.convs:
                    x = torch.relu(conv(x=x, adj=new_A))
                x = torch.sum(x, dim = 1)
                x = self.read_out(x)
                return torch.sigmoid(x)
            else:
                new_x, new_A = self.data_reassignment(data, batchwise=False)
                x = new_x
                for conv in self.convs:
                    x = torch.relu(conv(x=x, adj=new_A))
                x = torch.sum(x, dim=1)
                x = self.read_out(x)
                return torch.sigmoid(x)