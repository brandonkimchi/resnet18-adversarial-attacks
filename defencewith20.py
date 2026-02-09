import torch
from torch import nn, optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn.utils.prune as prune


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

dataset_path = r"archive/data/train"   
full_dataset = ImageFolder(dataset_path, transform=train_transform)

print("Images:", len(full_dataset))
print("Classes:", full_dataset.classes)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# fixed split for fair comparisons
g = torch.Generator().manual_seed(0)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

# make validation use val transforms
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# load baseline model 
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)
model = model.to(device)

base_weights = r"resnet18_animal20.pth"   
state_dict = torch.load(base_weights, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("Loaded baseline weights:", base_weights)


# structured pruning 
# prune whole filters in layer4 conv layers
prune_amount = 0.2  #mayb try 0.1, 0.3, 0.4

for name, layer in model.named_modules():
    if name.startswith("layer4") and isinstance(layer, nn.Conv2d):
        prune.ln_structured(layer, name="weight", amount=prune_amount, n=1, dim=0)

# make pruning permanent
for layer in model.modules():
    if isinstance(layer, nn.Conv2d) and hasattr(layer, "weight_mask"):
        prune.remove(layer, "weight")

print(f"Pruned layer4 conv filters: {int(prune_amount*100)}%")


# fine-tune 
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

epochs = 6

for epoch in range(epochs):
    # train
    model.train()
    train_losses = []
    train_accs = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = loss_fn(out, y)

        model.zero_grad()
        loss.backward()
        optimiser.step()

        train_losses.append(loss.item())
        train_accs.append((out.argmax(dim=1) == y).float().mean().item())

    print(f"Epoch {epoch+1}", end=", ")
    print(f"training loss: {torch.tensor(train_losses).mean():.2f}", end=", ")
    print(f"training acc: {torch.tensor(train_accs).mean():.2f}")

    # validate
    model.eval()
    val_losses = []
    val_accs = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            val_losses.append(loss.item())
            val_accs.append((out.argmax(dim=1) == y).float().mean().item())

    print(f"Epoch {epoch+1}", end=", ")
    print(f"val loss: {torch.tensor(val_losses).mean():.2f}", end=", ")
    print(f"val acc: {torch.tensor(val_accs).mean():.2f}")


# save the model
out_name = f"resnet18_defended_pruned{int(prune_amount*100)}_new_ft.pth"
torch.save(model.state_dict(), out_name)
print("Saved defended model:", out_name)
