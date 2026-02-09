import os
from model.resnet import ResNet18Trainer
from attacks.ifgsm import IFGSMAttackRunner
import numpy as np
import matplotlib.pyplot as plt

def run_ifgsm_sweep(attacker, eps_list, iter_list):
    clean_acc = np.zeros((len(iter_list), len(eps_list)))
    adv_acc = np.zeros((len(iter_list), len(eps_list)))

    runs = len(eps_list) * len(iter_list)
    counter = 1
    for i, iters in enumerate(iter_list):
        for j, eps in enumerate(eps_list):
            print(f"Run: {counter}/{runs}")
            attacker.iters = iters
            attacker.eps = eps

            results = attacker.evaluate()
            clean_acc[i, j] = results["clean_acc"]
            adv_acc[i, j] = results["adv_acc"]
            counter+= 1

    return clean_acc, adv_acc

def plot_attack_heatmaps(clean_acc, adv_acc, eps_list, iter_list, save_path="media/ifgsm_sweep_defencew20.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(clean_acc, aspect="auto", origin="lower")
    axes[0].set_title("Clean Accuracy")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Iterations")
    axes[0].set_xticks(range(len(eps_list)))
    axes[0].set_xticklabels([f"{e:.3f}" for e in eps_list])
    axes[0].set_yticks(range(len(iter_list)))
    axes[0].set_yticklabels(iter_list)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(adv_acc, aspect="auto", origin="lower")
    axes[1].set_title("Adversarial Accuracy")
    axes[1].set_xlabel("Epsilon")
    axes[1].set_ylabel("Iterations")
    axes[1].set_xticks(range(len(eps_list)))
    axes[1].set_xticklabels([f"{e:.3f}" for e in eps_list])
    axes[1].set_yticks(range(len(iter_list)))
    axes[1].set_yticklabels(iter_list)
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#save_path = "resnet18_animal10_4epochs.pth"
save_path = "resnet18_defended_pruned20_new_ft.pth"

trainer = ResNet18Trainer(data_dir=r"data/train", epochs=20, save_path=save_path)
if not os.path.exists(save_path):
    print("Training the base model...")
    trainer.train()
else:
    print("The base model already exists")

val_loader = trainer.val_loader

eps_list = [i/255 for i in range(1,5)]  #1/255 -- 2/255
iter_list = list(range(1,7))   #1 -- 2

attacker = IFGSMAttackRunner(val_loader=val_loader, eps=8/255, iters=10)
attacker.load_weights(save_path)

clean_acc, adv_acc = run_ifgsm_sweep(attacker, eps_list, iter_list)

plot_attack_heatmaps(clean_acc, adv_acc, eps_list, iter_list)

