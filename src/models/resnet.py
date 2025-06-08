from .model import ResNet
import torch

def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet("resnet101", [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("resnet101.pth")["model_state_dict"])
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.
    """
    model = ResNet("resnet152", [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("resnet152.pth")["model_state_dict"])
    return model