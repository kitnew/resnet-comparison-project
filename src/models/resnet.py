from .model import ResNet

def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet("resnet101", [3, 4, 23, 3], **kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights loading not implemented")
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.
    """
    model = ResNet("resnet152", [3, 8, 36, 3], **kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights loading not implemented")
    return model