import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model, dataloader, optimizer=None, criterion=None, device="mps", learning_rate=0.01, epochs=5):
    """
    Train a model for a number of epochs on the provided dataloader.

    Parameters
    - model (torch.nn.Module): model to train.
    - dataloader (Iterable): yields (images, labels) pairs for training.
    - optimizer (torch.optim.Optimizer | None): optimizer to use. If None, Adam is created.
    - criterion (callable | None): loss function. If None, nn.CrossEntropyLoss() is used.
    - device (str | torch.device): device for training (default "mps").
    - learning_rate (float): initial learning rate used if optimizer is created (default 0.01).
    - epochs (int): number of training epochs (default 5).

    Returns
    - list[float]: average training loss per epoch.
    """
    train_losses = []
    device = torch.device(device)
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if not criterion:
        criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            labels = labels.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
    
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        
        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)
        print(f"[Epoch {epoch+1}] Average Loss: {epoch_loss:.4f}")

    return train_losses


def make_predictions(model, dataloader, labels_dict, device="mps", max_batches=1, topk=5):
    """
    Run a trained model over batches from a dataloader and collect top-k predictions.

    Parameters
    - model (torch.nn.Module): the classifier to run.
    - dataloader (Iterable): yields tuples (imgs, labels); imgs is a Tensor of shape (B,C,H,W).
    - labels_dict (dict[int,str]): mapping from true label index to human-readable name.
    - device (str | torch.device): device to run inference on (default "mps").
    - max_batches (int): maximum number of batches to process (default 1).
    - topk (int): how many top predictions to return per sample (default 5).

    Returns
    - list[dict]: one entry per sample with keys:
        - "true_idx" (int): ground-truth index
        - "true_name" (str): human name from labels_dict
        - "topk_idxs" (list[int]): predicted ImageNet indices (top-k)
        - "topk_probs" (list[float]): corresponding probabilities
    """
    results = []
    with torch.no_grad():
        for batch_i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            outputs = model(imgs)                      # (B, 1000)
            probs = torch.softmax(outputs, dim=1)      # convert to probabilities
            top_probs, top_idxs = probs.topk(topk, dim=1)

            for i in range(imgs.size(0)):
                results.append({
                    "true_idx": int(labels[i].item()),
                    "true_name": labels_dict[int(labels[i].item())],
                    "topk_idxs": top_idxs[i].cpu().tolist(),
                    "topk_probs": top_probs[i].cpu().tolist(),
                })

            if batch_i + 1 >= max_batches:
                break
    return results

def classify_sample(model, img_tensor, labels_dict=None, true_label=None, device="mps", topk=1):
    """
    Classify a single image tensor.

    - img_tensor: torch.Tensor with shape (C,H,W) or (1,C,H,W)
    - labels_dict: optional dict mapping label idx -> name
    - true_label: optional int label for including true info in result
    - device: device string or torch.device
    - topk: how many top predictions to return

    Returns a dict with keys: topk_idxs (list), topk_probs (list), the image to classify: numpy array and the class name: str
    """
    
    # normalize device
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        t = img_tensor
        if t.dim() == 3:
            t = t.unsqueeze(0)  # (1,C,H,W)
        t = t.to(device, non_blocking=True)
        out = model(t)                         # (1, num_classes)
        probs = torch.softmax(out, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)
        top_probs = top_probs[0].cpu().tolist()
        top_idxs = top_idxs[0].cpu().tolist()

    res = zip(top_idxs, top_probs)

    # Convert tensor (C, H, W) -> (H, W, C) and unnormalize to [0,1]
    img_tensor = img_tensor.cpu().detach()
    img_np = img_tensor.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0.0, 1.0)

    if labels_dict and true_label:
        class_name = labels_dict[int(true_label)]

    # plt.figure(figsize=(4, 4))
    # plt.imshow(img_np)
    # title = f"Label: {class_name}"
    # plt.title(title)
    # plt.axis('off')
    # plt.show()
    
    return res, img_np, class_name

