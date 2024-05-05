from torch import nn
import torch.nn.init as init


class OptPathModel(nn.Module):
    def __init__(self):
        super(OptPathModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128,64)
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(64,32)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu') 
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x




class CostToGoModel(nn.Module):
    def __init__(self):
        super(CostToGoModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, padding=1)
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, padding=1)
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128,64)
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(64,32)
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(32, 1)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu') 
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        
        return x


class FFNN(nn.Module):
    # Structure of SaIL model
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(32*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x