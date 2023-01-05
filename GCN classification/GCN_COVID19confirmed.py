# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch_geometric.transforms as T
import pandas as pd
from torch_geometric.datasets import Planetoid, NELL

neigh_edge = pd.read_csv(r"D:\learning\暑研\Covid-19_dataset_GCN\geographic\neighbor.csv", encoding="UTF8", header=None)
neigh_edge_list = neigh_edge.values.tolist()
edge_index = torch.tensor(neigh_edge_list, dtype=torch.long)
features = pd.read_csv(r"D:\learning\暑研\Covid-19_dataset_GCN\factors.csv", encoding="UTF8")
features_list = features.iloc[:, 1:27].values.tolist()  # 26 features，column27 is label, column 1 is itemid
cases_list = features.iloc[:,27].values.tolist()  # Categories of the number of confirmed cases，0 is the least and 4 is the most

x = torch.tensor(features_list, dtype=torch.float)
y1 = torch.tensor(cases_list, dtype=torch.float)
# y1 = torch.LongTensor(cases_list)
y = F.one_hot(y1.to(torch.int64), num_classes=5).float()  # y onehot


class MyOwnDataset(InMemoryDataset):  # Create my own GCN dataset
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['covid19confirmed.dataset']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []  # one graph with about 2000 nodes

        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
        data = T.RandomNodeSplit(num_train_per_class=200)(data)  # Set up the training and test sets
        print(data.train_mask)
        print(data.train_mask.sum().item())

        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = MyOwnDataset("D:\learning\暑研\Covid-19_dataset_GCN\covid19confirmed")


# print(dataset[0].train_mask)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(26, 26)
        self.conv3 = GCNConv(26, 26)
        self.conv4 = GCNConv(26, 16)
        self.conv5 = GCNConv(16, 16)
        self.conv2 = GCNConv(16, 5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.softmax(x, dim=1)
        print(x)
        # print(x.shape)
        return x


def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    print()
    for epoch in range(5000):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        # print(out[data.train_mask])
        # print(data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))


def test(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)

    datay = torch.argmax(data.y[data.test_mask], -1)
    correct = int(pred[data.test_mask].eq(datay).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))
    # predlist=pred.numpy()
    # np.savetxt("D:\learning\暑研\Covid-19_dataset_GCN\mytest1206.csv",predlist)


def load_data(name):
    if name == 'NELL':
        print('./' + name + '/')
        dataset = NELL(root='./' + name + '/')
        # CUDA out of memory
        _device = torch.device('cpu')
    else:
        dataset = Planetoid(root='./' + name + '/', name=name)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(_device)
    if name == 'NELL':
        data.x = data.x.to_dense()
        num_node_features = data.x.shape[1]
    else:
        num_node_features = dataset.num_node_features
    return data, num_node_features, dataset.num_classes


device = torch.device('cpu')
model = Net().to(device)

data = dataset[0].to(device)

train(model, data, device)
test(model, data)
