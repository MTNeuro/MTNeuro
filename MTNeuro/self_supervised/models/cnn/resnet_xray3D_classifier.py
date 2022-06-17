from torch import nn
from .resnet3d import  resnet18, resnet50


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

class resnet_xray3D_classifier(nn.Module):
    r"""XRAY-variant of ResNet classifier."""
    def __init__(self, resnet_model, num_classes=4):
        super().__init__()

        if resnet_model == 'resnet18':
            resnet_model = resnet18(
                sample_input_W=32,
                sample_input_H=32,
                sample_input_D=10,
                shortcut_type='B',
                no_cuda=False,
                num_classes=4)
        elif resnet_model == 'resnet50':
            resnet_model = resnet50(
                sample_input_W=32,
                sample_input_H=32,
                sample_input_D=10,
                shortcut_type='B',
                no_cuda=False,
                num_classes=4)
        else:
            raise ValueError

        self.representation_size = resnet_model.linear.in_features

        self.f = []
        for name, module in resnet_model.named_children():
            #if name == 'conv1':
                #module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
                
        self.f.append(Flatten())
        self.f.append(nn.Linear(self.representation_size, num_classes))

        # classifier
        self.f = nn.Sequential(*self.f)
        

    def forward(self, x):
        y = self.f(x)
        return y#.view(y.size(0), -1)

