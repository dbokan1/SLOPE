import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import OptPathModel, CostToGoModel
from torch.utils.tensorboard.writer import SummaryWriter
from load_data import load_weighted_data, load_ctg_data
import os
from collections import Counter
from utils import get_root_path
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model():
    # model= CostToGoModel()
    model= OptPathModel()
    model.to(device)

    crit = nn.SmoothL1Loss(reduction='none')
    # crit = nn.MSELoss(reduction='none')
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    return model, crit, optimizer


def prep_dataloader(X, y, batch_size):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader



def start_training(exp_name, dataset_name, num_epochs, name, balance, batch_size):
    writer = SummaryWriter('logs/'+dataset_name+'/'+exp_name)
    # X, y = load_ctg_data(get_root_path() / 'data' / dataset_name, 400)
    X, y, w = load_weighted_data(get_root_path() / 'data' / dataset_name, 400, balance)
    class_counts = Counter(y)
    print(class_counts) # info for balancing data in load_weighted_data function

    model, crit, optimizer =load_model()
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    train_loader, val_loader = prep_dataloader(X, y, batch_size)
    previous_loss=[]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels= inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels=labels.unsqueeze(1)
            loss = crit(outputs, labels).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                labels=labels.unsqueeze(1)
                val_loss += crit(outputs, labels).mean().item()
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)

        scheduler.step()

        print(f"Epoch {epoch+1}, Train: {running_loss / len(train_loader)}, Validation: {val_loss / len(val_loader)}")
        
        # simple early stopping
        if len(previous_loss)<5:
            previous_loss.append(running_loss)
        else:
            previous_loss.pop(0)
            previous_loss.append(running_loss)
            if abs(previous_loss[0]-previous_loss[-1])/previous_loss[0]<0.001:
                print('Early stopping')
                if epoch<20:
                    return True

    print("Training finished")
    os.makedirs('../weights/'+dataset_name+'/'+exp_name, exist_ok=True)
    

    torch.save(model.state_dict(), '../weights/'+dataset_name+'/'+exp_name+'/'+name)
    writer.close()
    return False

start_training('OptPathModel', 'shifting_gaps', 45, 'demo_model.pth', False, 256)