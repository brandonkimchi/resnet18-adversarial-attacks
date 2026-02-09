# Adversarial fine-tuning to improve robustness vs I-FGSM.
# Uses existing weights of defence.py as a starting point.

import torch
from torch import nn, optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# settings
# Start from the pruned + fine-tuned model
start_weights = "resnet18_defended_pruned20_ft.pth"

dataset_path = r"archive/data/train"
batch_size = 64
SEED = 0

# adversarial training settings
EPS_TRAIN = 8 / 255       
ITERS_TRAIN = 5
LAMBDA_ADV = 0.9  
EPOCHS = 4                

lr = 1e-4
momentum = 0.9
weight_decay = 1e-4

# evaluation attack settings (main metric)
EPS_EVAL = 8 / 255
ITERS_EVAL = 10


# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))


# data 
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

full_dataset = ImageFolder(dataset_path, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

g = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

# make val use val transforms
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Classes:", full_dataset.classes)


# model
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model = model.to(device)

state_dict = torch.load(start_weights, map_location=device)
model.load_state_dict(state_dict)
print("Loaded start weights:", start_weights)


# I-FGSM helper (for training + eval) 
def ifgsm_attack(model, images, labels, eps=8/255, iters=10):
    alpha = eps / iters
    loss_fn = nn.CrossEntropyLoss()

    x0 = images.clone().detach()
    x = images.clone().detach()

    for _ in range(iters):
        x.requires_grad_(True)
        out = model(x)
        loss = loss_fn(out, labels)

        model.zero_grad()
        loss.backward()

        grad = x.grad.sign()
        x = x + alpha * grad

        eta = (x - x0).clamp(-eps, eps)
        x = (x0 + eta).clamp(0, 1).detach()

    return x


# adversarial fine-tuning 
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

print("\nAdversarial fine-tuning...")
print(f"EPS_TRAIN={EPS_TRAIN:.5f}, ITERS_TRAIN={ITERS_TRAIN}, epochs={EPOCHS}\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss_list = []
    train_acc_list = []

    for x, y in tqdm(train_loader, desc=f"Train epoch {epoch+1}", unit="batch"):
        x, y = x.to(device), y.to(device)

        # make adversarial examples using the current model (white-box)
        x_adv = ifgsm_attack(model, x, y, eps=EPS_TRAIN, iters=ITERS_TRAIN)

        # forward on clean + adv
        out_clean = model(x)
        out_adv = model(x_adv)

        loss_clean = loss_fn(out_clean, y)
        loss_adv = loss_fn(out_adv, y)

        loss = (1 - LAMBDA_ADV) * loss_clean + LAMBDA_ADV * loss_adv

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss_list.append(loss.item())
        train_acc_list.append((out_clean.argmax(dim=1) == y).float().mean().item())

    print(f"Epoch {epoch+1}: train loss {torch.tensor(train_loss_list).mean():.3f}, "
          f"train clean acc {torch.tensor(train_acc_list).mean():.3f}")

    # quick clean val check each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch+1}: val clean acc {100*correct/total:.2f}%\n")


# save new weights
out_name = f"resnet18_adversarialtrained2.pth"
torch.save(model.state_dict(), out_name)
print("Saved adversarially fine-tuned model:", out_name)
