import torch 
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models


def build_model(args, device):
        
    if args.model == 'base': 
        model = CNN_Base().to(device)
    elif args.model == 'efficientnet_b4':
        if args.pretrained and args.mode != 'test':
            model = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            model = EfficientNet.from_name('efficientnet-b4')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, args.num_classes)
    elif args.model == 'resnet50':
        model = Resnet50(args.num_classes, dropout=False, pretrained=args.pretrained)

    if device: 
        model = model.to(device)
    
    return model


class Resnet50(nn.Module):
    def __init__(self, num_classes, dropout=False, pretrained=False):
        super().__init__()
        model = models.resnet50(pretrained=pretrained)
        model = list(model.children())[:-1]
        if dropout:
            model.append(nn.Dropout(0.2))
        model.append(nn.Conv2d(2048, num_classes, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()  

        self.cnn_layer = nn.Sequential(            
            nn.Conv2d(3,6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6,12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(12,15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential( 
            nn.Linear(735, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), 
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out