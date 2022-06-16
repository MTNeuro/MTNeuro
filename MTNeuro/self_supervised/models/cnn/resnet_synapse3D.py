from torch import nn
# from torchvision.models.resnet import resnet18, resnet50
from .resnet3d import  resnet18, resnet50

class resnet_synapse3D(nn.Module):
    r"""XRAY-variant of ResNet."""
    def __init__(self, resnet_model, depth = 10):
        super().__init__()

        if resnet_model == 'resnet18':
            resnet_model = resnet18(
                sample_input_W=28,
                sample_input_H=28,
                sample_input_D=depth,
                shortcut_type='B',
                no_cuda=False,
                num_classes=2)
        elif resnet_model == 'resnet50':
            resnet_model = resnet50(
                sample_input_W=28,
                sample_input_H=28,
                sample_input_D=depth,
                shortcut_type='B',
                no_cuda=False,
                num_classes=2)
        else:
            raise ValueError

        self.representation_size = resnet_model.linear.in_features

        self.f = []
        for name, module in resnet_model.named_children():
            # if name == 'conv1':
            #     module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        y = self.f(x)
        return y.view(y.size(0), -1)

    def get_linear_classifier(self, output_dim=2):
        r"""Return linear classification layer."""
        return nn.Linear(self.representation_size, output_dim)
