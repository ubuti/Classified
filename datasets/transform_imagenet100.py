import os
import json
import urllib.request
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageNet100Dataset(Dataset):
    def __init__(self, images, transform=None, labels=None):
        self.images = images
        self.labels = labels
        if transform:
            self.transform = transform
        else:
            # simple image transforms suitable for a training pipeline (source https://pytorch.org/vision/stable/models.html#classification)
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
    
    
def load_image100_data(training_data_path="/Users/inuit/.cache/kagglehub/datasets/ambityga/imagenet100/train/", 
                       labels_path="/Users/inuit/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8/Labels.json",
                       validation_data_path = "/Users/inuit/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8/val.X",
                       train=True, batch_size=16):
    
    with open(labels_path, "r", encoding="utf-8") as f:
        # Has form <directory name, label string>
        labels_dir = json.load(f)
        f.close()
    
    if train:
        # 100 classes are expected from imagenet100
        assert len(labels_dir) == 100
        data_path = training_data_path
    else:
        data_path = validation_data_path
    

    # find class folders in training_data that match keys in labels_dir
    available_classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path , d)) and d in labels_dir]
    assert len(available_classes) == len(labels_dir), "Mismatch between available class folders and labels_dir entries"

    # download ImageNet-1k class index mapping: "index" -> [wnid, name]
    cache_path = os.path.expanduser("~/.cache/imagenet_class_index.json")
    if not os.path.exists(cache_path):
        print("Fetching from source...")
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        urllib.request.urlretrieve(url, cache_path)

    with open(cache_path, "r", encoding="utf-8") as f:
        imagenet_idx = json.load(f)
        f.close()
        
    # rearrange order
    names_to_idx = {int(k): v[1] for k, v in imagenet_idx.items()}
    imagenet_idx = {v[0] : (v[1], k) for k, v in imagenet_idx.items() if v[0] in labels_dir.keys()}

    images = []
    labels = []

    for keys, values in imagenet_idx.items():    
        files = os.listdir(os.path.join(data_path , keys))    
        for file in files:
            file_path = os.path.join(data_path , keys, file)
            if os.path.isfile(file_path):
                images.append(file_path)
                labels.append(int(values[1]))

    assert len(images) == len(labels), "Mismatch between number of images and labels"
    assert len(images) > 0, "No images found in the dataset"
    assert len(labels) > 0, "No labels found in the dataset"

    # create dataset + dataloader
    dataset = ImageNet100Dataset(images, labels=labels)  # 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # quick sanity check: fetch one batch
    batch_imgs, batch_labels  = next(iter(dataloader)) 
    assert batch_imgs.shape[0] == batch_labels.shape[0], "Batch size mismatch between images and labels"
    assert batch_imgs.shape[1] == 3, "Expected 3 channels in batch images"
    
    return dataloader, names_to_idx