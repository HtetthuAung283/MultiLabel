import torch.nn as nn
from torchvision import  models

def get_classifier(n_classes):
    return nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes))


def MultiLabelModel(train=True):

    if train:
        resnet34 = models.resnet34(pretrained=True)
    else:
        resnet34 = models.resnet34(pretrained=False)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*list(resnet34.children())[:-2])
            for p in self.features.parameters():
                p.requires_grad = True
            self.classifier1 = get_classifier(11)
            self.classifier2 = get_classifier(8)
            self.classifier3 = get_classifier(5)

        def forward(self,x):

            x = self.features(x)
            x = x.view(x.size(0), -1)

            y1 = self.classifier1(x)
            y2 = self.classifier2(x)
            y3 = self.classifier3(x)

            return y1,y2,y3
    test = Network()
    return test