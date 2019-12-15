import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import time
import os
import copy
from model import MultiLabelModel
from torch.utils.data import Dataset, DataLoader
from loss import FocalLoss


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


model_path='./model/'
best_path=model_path+'model.pth'
data_root='./data/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

class MyDataset(Dataset):
    def __init__(self, data_rt,data,data_transform):
        self.imgs = [os.path.join(data_rt,"images",i) for i in data['filename']]
        self.labels=[(torch.tensor(a),torch.tensor(b),torch.tensor(c)) for a,b,c in zip(data['pattern'],data['neck'],data['sleeve_length'])]
        self.data_transform=data_transform
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.data_transform(Image.open(self.imgs[idx]).convert('RGB')),torch.tensor(self.labels[idx])



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss=999999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # print('Sailabh')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            running_corrects = 0
            running_corrects1 = 0
            running_corrects2 = 0
            running_corrects3 = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
#                 print('1st batch')
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        output1,output2,output3 = model(inputs)
#                         print(labels[:,0])
                        loss1 = criterion(output1, labels[:,0].long())
#                         print(loss1)
                        loss2 = criterion(output2, labels[:,1].long())
                        loss3 = criterion(output3, labels[:,2].long())
                        loss = loss1 + loss2 + loss3
    #                         print(loss)
                            
                            
                        _, preds1 = torch.max(output1, 1)
                        _, preds2 = torch.max(output2, 1)
                        _, preds3 = torch.max(output3, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss1 += loss1.item() * inputs.size(0)
                    running_loss2 += loss2.item() * inputs.size(0)
                    running_loss3 += loss3.item() * inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects1 += torch.sum(preds1 == labels[:,0].data)
                    running_corrects2 += torch.sum(preds2 == labels[:,1].data)
                    running_corrects3 += torch.sum(preds3 == labels[:,2].data)
    #                 running_corrects += torch.sum(preds == labels.data)

                epoch_loss1 = running_loss1 / len(dataloaders[phase].dataset)
                epoch_loss2 = running_loss2 / len(dataloaders[phase].dataset)
            epoch_loss3 = running_loss3 / len(dataloaders[phase].dataset)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc1 = running_corrects1.double() / len(dataloaders[phase].dataset)
            epoch_acc2 = running_corrects2.double() / len(dataloaders[phase].dataset)
            epoch_acc3 = running_corrects3.double() / len(dataloaders[phase].dataset)
            

            print('{} Pattern Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss1, epoch_acc1))
            print('{} Neck Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss2, epoch_acc2))
            print('{} Sleeve Length Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss3, epoch_acc3))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('Saving at {} Epoch'.format(epoch))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), best_path)
            if phase == 'val':
                val_acc_history.append((epoch_acc1,epoch_acc2,epoch_acc3))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history




def main(m,criterion,num_epochs,lr_rate,optimizer,data_transforms,batch_size):

    

    
    df=pd.read_csv(data_root+'attributes.csv')
    # df.head()
    # df.shape

    # Remove Rows from df whose images aren't avalilable
    ind=[]
    for i in df['filename']:
        if os.path.exists(os.path.join(data_root,"images",i)):
            ind.append(True)
        else:
            ind.append(False)
    df=df[ind]

    # Replacing NA to Unknown Class

    df['neck'].fillna(7,inplace=True)
    df['sleeve_length'].fillna(4,inplace=True)
    df['pattern'].fillna(10,inplace=True)

    # Train Test Split
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    test.to_csv(data_root+'test.csv')
    train.to_csv(data_root+'train.csv')

    

    train_dataset=MyDataset(data_root,train,data_transforms['train'])
    valid_dataset=MyDataset(data_root,test,data_transforms['val'])


    train_dl=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)




    data_loaders={'train':train_dl,'val':valid_dl}
    
    model, val_acc_history=train_model(m,data_loaders, criterion, optimizer, num_epochs=num_epochs, is_inception=False)

    torch.save(model.state_dict(), model_path+'last_epoch.pth')

    with open(model_path+'val_acc_hist.pkl', 'wb') as f:
        pickle.dump(val_acc_history, f)


if __name__ == '__main__':


    criterion=FocalLoss()
    # criterion=nn.CrossEntropyLoss()

    num_epochs=1
    lr_rate=0.0001

    # Input SIze
    input_size=224
    batch_size=32
    m=MultiLabelModel()

    m.cuda();
    optimizer = torch.optim.Adam(m.parameters(), lr=lr_rate)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation([-30, 30]),
            transforms.Resize((input_size,input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size,input_size)),
    #         transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    

    
    main(m,criterion,num_epochs,lr_rate,optimizer,data_transforms,batch_size)
