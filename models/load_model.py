import torchvision
import torch

def load_model(model_name="resnet50", weights="IMAGENET1K_V1"):
    """Loads a pretrained vision model from pytorch hub. Choose in respect to task. Weights can be supplied at will, if non is provided the latest option is the default. Returns model."""
    
    print("Evaluating model config...")
    try:
        model = torch.hub.load("pytorch/vision", model_name, weights)
    except:
        print(f"Could not load given model {model_name}")
        raise Exception
    try:
        weight_enum = [w for w  in torch.hub.load("pytorch/vision", "get_model_weights", model_name)]
        if not weights in [str(f).split('.')[1] for f in weight_enum]:
            weights = weight_enum[len(weight_enum)-1]
            print(f"Using alternative weights: {weights}")
    except:
        pass
        raise Exception
    print(f"Model has {model.fc.out_features} output labels")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # set model to device
    model = model.to(device)
    
    return model
