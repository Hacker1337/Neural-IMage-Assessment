import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from .model.model import *

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path="."):
    global model
    MODEL_PATH = os.path.join(path, "pretrained_model/epoch-82.pth")

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

seed = 42
torch.manual_seed(seed)

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

def predict(img_path, script_path="."):
    if model is None:
        load_model(script_path)

    im = Image.open(img_path)
    im = im.convert('RGB')
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)

    mean, std = 0.0, 0.0
    for j, e in enumerate(out, 1):
        mean += j * e
    for k, e in enumerate(out, 1):
        std += e * (k - mean) ** 2
    std = std ** 0.5

    return mean.item(), std.item()