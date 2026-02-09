from torchvision import models
from torch import nn
import torch

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)

if torch.cuda.is_available():
    model = model.cuda()

state_dict = torch.load("resnet18_animal2.0.pth", weights_only=True)
model.load_state_dict(state_dict)

model.eval()


