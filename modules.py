from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# define a module you like here, and set trianmodel for each  in train.py
class network(nn.Module):
    def __init__(self, model_name: str = "CNN", p: float = 0.2, output_layer: int = 10):
        super(network, self).__init__()
        # Define passed parameters
        self.model_name = model_name
        self.p = p
        self.output_layer = output_layer

        # Define topology
        self.set_network_topology()

    # Now some behemoth functions to house all the network types
    def set_network_topology(self):

        if self.model_name == "CNN":
            # Standard CNN
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

            self.dropout = nn.Dropout(self.p)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)  
        
        elif self.model_name == "CNNWeak":
            self.dropout1 = nn.Dropout(self.p)
            self.conv1 = nn.Conv2d(1, 1, kernel_size=11)
            self.dropout2 = nn.Dropout(self.p)
            self.fc1 = nn.Linear(1*9*9, 25)
            self.dropout3 = nn.Dropout(self.p)
            self.fc2 = nn.Linear(25, self.output_layer)    

        elif self.model_name == "CNNDumb":
            self.conv1 = nn.Conv2d(1, 1, kernel_size=11)
            self.dropout = nn.Dropout(self.p)
            self.fc1 = nn.Linear(1*9*9, 10)

        elif self.model_name == "ReLu_Linear":
            self.fc1 = nn.Linear(28*28, 48)
            self.fc2 = nn.Linear(48, 24)
            self.fc3 = nn.Linear(24, 10)
            self.dropout = nn.Dropout(self.p)
        
        elif self.model_name == "sigmoid":
            self.dropout = nn.Dropout(self.p)
            self.fc1 = nn.Linear(28*28, 100)
            self.fc2 = nn.Linear(100, 10) 
        
        elif self.model_name == "Logistic":
            self.dropout = nn.Dropout(self.p)
            self.fc1 = nn.Linear(28*28, 10)

        elif self.model_name == "RNN":
            self.dropout = nn.Dropout(self.p)
            self.rnn = nn.RNN(28, 200, 2, batch_first=True, nonlinearity='relu')
            self.fc1 = nn.Linear(200, 10)    

        elif self.model_name == "BNReLu":
            self.classifier = nn.Sequential(
                nn.Linear(28*28,196),
                nn.Dropout(self.p),
                nn.BatchNorm1d(196), 
                nn.ReLU(),
                nn.Linear(196, 48),
                nn.Dropout(self.p),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Linear(48,24),
                nn.Dropout(self.p),
                nn.BatchNorm1d(24),
                nn.ReLU(),
                nn.Linear(24,10))
        else:
            raise NotImplementedError(f"Network type {self.model_name} is not defined")

    # Set forward passes based on model name
    def forward(self, x):
        if self.model_name == "CNN":
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)

        elif self.model_name == "CNNWeak":
            x = self.dropout1(x)
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = torch.flatten(x, 1)
            x = self.dropout2(x)
            x = F.relu(self.fc1(x))
            x = self.dropout3(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)

        elif self.model_name == "CNNDumb":
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            output = F.log_softmax(x, dim=1)

        elif self.model_name == "ReLu_Linear":
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.dropout(self.fc2(x)))
            output = F.log_softmax(F.relu(self.fc3(x)), dim=1)

        elif self.model_name == "sigmoid":
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = torch.sigmoid(x)
            x = self.dropout(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)    

        elif self.model_name == "Logistic":
            x = self.dropout(x.view(x.size(0), -1))
            x = self.fc1(x)
            output = F.log_softmax(x, dim=1)

        elif self.model_name == "RNN":
            x = x.view(-1, 28, 28)
            h0 = torch.zeros(2, x.size(0), 200).requires_grad_()
            x, _ = self.rnn(x, h0)
            x = self.dropout(x)
            x = self.fc1(x[:, -1, :])
            output = F.log_softmax(x, dim=1)

        elif self.model_name == "BNReLu":
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            output = F.log_softmax(x, dim=1)
        else:
            raise NotImplementedError(f"Network type {self.model_name} is not defined, \
                and frankly, I'm not sure how you made it here.")
        
        return output
