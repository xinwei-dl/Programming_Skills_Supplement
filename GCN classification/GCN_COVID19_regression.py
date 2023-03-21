# -*- coding:utf-8 -*-
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn
import torch_geometric.transforms as T
import pandas as pd
from torch_geometric.datasets import Planetoid, NELL
import matplotlib.pyplot as plt

# Read the neighbor.csv and factors_nom.csv files
neigh_edge = pd.read_csv(r"D:\learning\暑研\Covid-19_dataset_GCN\geographic\neighbor.csv", encoding="UTF8", header=None)
neigh_edge_list = neigh_edge.values.tolist()
edge_index = torch.tensor(neigh_edge_list, dtype=torch.long)
feathers = pd.read_csv(r"D:\learning\暑研\Covid-19_dataset_GCN\factors_nom.csv", encoding="UTF8")
feathers_list = feathers.iloc[:,
                1:27].values.tolist()  # The first 27 columns are features, and the 28th column is label
cases_list = feathers.iloc[:, 27].values.tolist()  # Confirmed cases

# Convert data into PyTorch tensors
x = torch.tensor(feathers_list, dtype=torch.float)
y = torch.LongTensor(cases_list)
# Define a custom dataset class to store data
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['Covid_regression.dataset']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list
        data_list = []  # Only one graph with about 2000 nodes

        # Store data into a `Data` object
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
        data = T.RandomNodeSplit(num_test=400)(data) # Split the data into train/test/val masks
        print("============")
        print(data.train_mask)
        print(data.train_mask.sum().item())
        print(data.test_mask.sum().item())
        print(data.val_mask.sum().item())

        data_list.append(data)

        # Collate the list of `Data` objects into a single `Data` object and save it
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Instantiate the custom dataset class
dataset = MyOwnDataset("D:\learning\暑研\Covid-19_dataset_GCN\Covid_regression")

# Define a custom dataset class to store data
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(26, 64) # Define the first graph convolution layer
        self.conv3 = GCNConv(64, 32) # Define the third graph convolution layer
        self.line1 = torch.nn.Linear(32, 1) # Define a linear layer with output size 1

    def forward(self, data):
        x, edge_index = data.x, data.edge_index # Retrieve node features (x) and edge indices
        x = self.conv1(x, edge_index) # Apply the first convolution layer to the input features
        x = F.relu(x) # Apply ReLU activation function
        x = F.dropout(x, training=self.training) # Apply dropout regularization
        x = self.conv3(x, edge_index) # Apply the third convolution layer
        x = F.relu(x) # Apply ReLU activation function
        x = F.dropout(x, training=self.training) # Apply dropout regularization
        x = self.line1(x) # Apply the linear layer
        print(x)
        return x


def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3) # Define the optimizer
    loss_function = torch.nn.MSELoss().to(device)
    model.train()
    data.y=data.y.to(torch.float32).reshape(2352,1) # Convert the label tensor to float32 and reshape it to (2352, 1)
    loss_list=[] # Create an empty list to store training losses
    val_loss_list=[] # Create an empty list to store validation losses
    for epoch in range(3000):
        out = model(data) # Forward pass
        out= out.to(torch.float32)
        optimizer.zero_grad()
        loss_train = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss_list.append(loss_train) # Append the training loss to the list
        loss_train.backward() # Backpropagate the gradients
        with torch.no_grad():
            model.eval() # Set the model to evaluation mode
            out_val = model(data) # Forward pass on the validation set
            loss_val = loss_function(out_val[data.val_mask], data.y[data.val_mask])
            val_loss_list.append(loss_val.item())
            model.train() # Set the model back to training mode
        optimizer.step() # Update the model parameters
        print('Epoch {:03d} loss {:.4f}'.format(epoch, loss_train.item()))
    losses = [loss.detach().numpy() for loss in loss_list]
    losses1 = val_loss_list
    # Draw a line chart of loss versus epoch to find the most appropriate number of epochs
    x1 = range(1, 3001)
    y1 = losses
    y2=losses1
    plt.cla()
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '-',linewidth=0.3)
    plt.plot(x1, y2, '-',linewidth=0.3)
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig("D:\learning\暑研\Covid-19_dataset_GCN\Train_loss.png")
    plt.show()


def test(model, data):
    model.eval()
    pred = model(data)


def load_data(name):
    # Load the dataset named 'name' using PyTorch Geometric library
    if name == 'NELL':
        print('./' + name + '/')
        dataset = NELL(root='./' + name + '/')
        # If out of memory error occurs, use CPU for computations
        _device = torch.device('cpu')
    else:
        dataset = Planetoid(root='./' + name + '/', name=name)
        # Use GPU if available, otherwise use CPU for computations
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move the dataset to the selected device for computations
    data = dataset[0].to(_device)
    # For the NELL dataset, convert the feature matrix to dense format and get the number of node features
    if name == 'NELL':
        data.x = data.x.to_dense()
        num_node_features = data.x.shape[1]
    # For the Planetoid dataset, get the number of node features
    else:
        num_node_features = dataset.num_node_features
    # Return the loaded dataset, number of node features, and number of classes in the dataset
    return data, num_node_features, dataset.num_classes


# Define the device to use for computations
device = torch.device('cpu')
# Instantiate the neural network model and move it to the selected device for computations
model = Net().to(device)
data = dataset[0].to(device)
train(model, data, device)
test(model, data)

