import torch
from torch import nn
from torchvision import models
from tqdm import tqdm


class IFGSMAttackRunner:
    def __init__(self, val_loader, num_classes=10, eps=1/255, iters=1, device=None):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.val_loader = val_loader
        self.eps = eps
        self.iters = iters
        self.loss_fn = nn.CrossEntropyLoss()

        self.model = self._build_model(num_classes)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, num_classes):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        return model

    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()


    def attack(self, images, labels):
        alpha = self.eps / self.iters

        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        ori_images = images.clone().detach()

        for _ in range(self.iters):
            images.requires_grad_(True)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            grad = images.grad.sign()
            images = images + alpha * grad

            eta = torch.clamp(images - ori_images, -self.eps, self.eps)
            images = torch.clamp(ori_images + eta, 0, 1).detach()

        return images

    def evaluate(self):
        correct_clean = 0
        correct_adv = 0
        total = 0

        progress_bar = tqdm(self.val_loader, desc="Evaluating", unit="batch")

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Clean accuracy
            with torch.no_grad():
                outputs = self.model(images)
            preds = outputs.argmax(dim=1)
            correct_clean += (preds == labels).sum().item()

            # Adversarial accuracy
            adv_images = self.attack(images, labels)
            outputs_adv = self.model(adv_images)
            preds_adv = outputs_adv.argmax(dim=1)
            correct_adv += (preds_adv == labels).sum().item()

            total += labels.size(0)

            progress_bar.set_postfix({
                "Clean Acc": f"{100 * correct_clean / total:.2f}%",
                "Adv Acc": f"{100 * correct_adv / total:.2f}%"
            })

        return {
            "clean_acc": 100 * correct_clean / total,
            "adv_acc": 100 * correct_adv / total
        }
