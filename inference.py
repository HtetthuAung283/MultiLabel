
import torch
import pandas as pd
from PIL import Image
import time
import os
from model import MultiLabelModel
from torchvision import transforms

model_path='./model/'
best_model_wts=model_path+'model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def predict(model,f):

    input_size= 224
    data_transforms_pred = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img=torch.unsqueeze(data_transforms_pred(Image.open(f).convert('RGB')),0)
    img = img.to(device)
    pred=model(img)

    _, preds1 = torch.max(pred[0], 1)
    _, preds2 = torch.max(pred[1], 1)
    _, preds3 = torch.max(pred[2], 1)

    return preds1.tolist()[0], preds2.tolist()[0], preds3.tolist()[0]


def main(model):

    preds_1=[]
    preds_2=[]
    preds_3=[]
    data_root='./data/'
    test=pd.read_csv('data/test.csv')
    s_t=time.time()
    for ind,name in enumerate(test['filename']):
        if ind%100==0:
            print(ind,time.time()-s_t)
    #     ind+=1
        f=os.path.join(data_root,'images',name)
        try:

            pred1,pred2,pred3=predict(model,f)

            preds_1.append(pred1)
            preds_2.append(pred2)
            preds_3.append(pred3)

        except Exception as e:
            print(f,e)  
            
    test['neck_pred']=preds_2
    test['length_pred']=preds_3
    test['pattern_pred']=preds_1  

    test.to_csv(data_root+'test_pred.csv')

if __name__ == '__main__':

    m=MultiLabelModel(train=False)
    weight=torch.load(best_model_wts,map_location=device)
    m.load_state_dict(weight)
    # m.cuda()
    main(m)
    # predict(m,f)
