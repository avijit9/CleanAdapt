from models.i3d import InceptionI3d

def select_backbone(network, first_channel = 3):
    param = {'feature_size': 1024}

    if network == "i3d":
        model = InceptionI3d(400, in_channels = first_channel)
    else:
        raise NotImplementedError
    
    return model
