from torch import nn
from torchvision.models.resnet import resnet18, resnet50


class resnet_xray(nn.Module):
    r"""XRAY-variant of ResNet."""
    def __init__(self, resnet_model, depth = 1):
        super().__init__()

        if resnet_model == 'resnet18':
            resnet_model = resnet18()
        elif resnet_model == 'resnet50':
            resnet_model = resnet50()
        else:
            raise ValueError

        self.representation_size = resnet_model.fc.in_features

        self.f = []
        for name, module in resnet_model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(depth, 64, kernel_size=3, stride=1, padding=1, bias=False)

            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        y = self.f(x)
        return y.view(y.size(0), -1)

    def get_linear_classifier(self, output_dim=4):
        r"""Return linear classification layer."""
        if output_dim > 2:
            return nn.Linear(self.representation_size, output_dim)
        else:
            return nn.Linear(self.representation_size, 1)

class combine_model(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
