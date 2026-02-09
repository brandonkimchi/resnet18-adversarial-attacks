import torch
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# Check for gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))

# Validation Data
val_transform = transforms.Compose([    # Resnet expects 224x224 RGB and tensor values between 0 and 1
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset_path = r"archive/data/train"

full_dataset = ImageFolder( # Load the dataset
    dataset_path,
    transform=val_transform
)

train_size = int(0.8 * len(full_dataset))   # 80% of the dataset is train data 20% validation data
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # Loads data in batches of 64

# Model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)
model = model.to(device)

# Load trained weights
state_dict = torch.load(
    "resnet18_animal10.pth",
    map_location=device
)

# Update model to trained model
model.load_state_dict(state_dict)   # Load the trained weights in the resnet model
model.eval()

# I-FGSM Attack
def ifgsm_attack(model, images, labels, eps=8/255, iters=10):
    alpha = eps / iters # stepsize per iteration

    images = images.clone().detach().to(device) # we want a new tensor, and send it to the gpu after detaching from old graph
    labels = labels.to(device)  # send label to gpu too

    loss_fn = nn.CrossEntropyLoss() # loss function
    ori_images = images.clone().detach()    # save original image as I-FGSM cant keep changing as it would completely destroy the image instead of small disturbance

    for _ in range(iters):
        images.requires_grad_(True) # Turn on gradients on input (pixels instead of weights)

        outputs = model(images)     #   Process images through resnet model
        loss = loss_fn(outputs, labels) # Determine the loss

        # Calculate the gradients
        model.zero_grad()   # prevents gradient accumulation
        loss.backward()

        grad = images.grad.sign()   # Take gradient direction
        images = images + alpha * grad  # FGSM update

        # epsilon constraint
        eta = torch.clamp(images - ori_images, min=-eps, max=eps)   # ensure the disturbance remains <= eps
        images = torch.clamp(ori_images + eta, 0, 1).detach()   # keep valid pixel values

    return images   #   adversarial images

# Evaluation with progress bar
correct_clean = 0
correct_adv = 0
total = 0

progress_bar = tqdm(
    val_loader,
    desc="Evaluating",
    unit="batch"
)

for images, labels in progress_bar:
    images = images.to(device)
    labels = labels.to(device)

    # Clean prediction
    with torch.no_grad():
        outputs = model(images)
    preds = outputs.argmax(dim=1)
    correct_clean += (preds == labels).sum().item()

    # Adversarial prediction
    adv_images = ifgsm_attack(model, images, labels)
    outputs_adv = model(adv_images)
    preds_adv = outputs_adv.argmax(dim=1)
    correct_adv += (preds_adv == labels).sum().item()

    total += labels.size(0)

    # Live update in progress bar
    progress_bar.set_postfix({
        "Clean Acc": f"{100 * correct_clean / total:.2f}%",
        "Adv Acc": f"{100 * correct_adv / total:.2f}%"
    })

# Final results
clean_acc = correct_clean / total
adv_acc = correct_adv / total

print("\nFinal Results")
print("Clean accuracy:", round(clean_acc * 100, 2), "%")
print("Adversarial accuracy (I-FGSM):", round(adv_acc * 100, 2), "%")
