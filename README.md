# ResNet18 Adversarial Attacks

## Overview

This project investigates the **security and robustness of deep neural networks** by attacking and defending a **ResNet‑18** convolutional neural network trained on the **Animals‑10 dataset**. The work focuses on **adversarial machine learning**, a subfield of modern cybersecurity that studies how machine learning models can be intentionally manipulated and how such vulnerabilities can be mitigated.

We evaluate a strong **white‑box adversarial attack**—the *Iterative Fast Gradient Sign Method (I‑FGSM)*—and analyze multiple defense strategies, including **structured pruning** and **adversarial fine‑tuning**. The results highlight the limitations of capacity‑based defenses against strong gradient‑based attacks.

---

## Project Goals

* Train a ResNet‑18 image classifier using transfer learning
* Evaluate model vulnerability to iterative adversarial attacks
* Assess structured pruning as a defense mechanism
* Explore adversarial fine‑tuning to improve robustness
* Analyze the trade‑off between clean accuracy and adversarial robustness

---

## Dataset

**Animals‑10 Dataset**
A multi‑class image dataset containing 10 animal categories:

* Dog, Cat, Horse, Spider, Butterfly
* Chicken, Sheep, Cow, Squirrel, Elephant

**Preprocessing & Augmentation**

* Images resized to **224 × 224**
* Random horizontal flip (training only)
* 80% training / 20% validation split

---

## Model Architecture

* **Backbone:** ResNet‑18 (pretrained on ImageNet)
* **Transfer Learning:** Final fully connected layer replaced with 10‑class output
* **Loss Function:** Cross‑Entropy Loss
* **Optimizer:** Stochastic Gradient Descent (SGD)
* **Batch Size:** 64

Two baseline models were trained:

* **4‑epoch model** (fast convergence baseline)
* **20‑epoch model** (better stability and convergence)

---

## Adversarial Attack: I‑FGSM

The **Iterative Fast Gradient Sign Method (I‑FGSM)** is a gradient‑based white‑box attack that perturbs input images iteratively to maximize classification loss under an (L∞) constraint.

**Key Properties**

* Uses model gradients to craft adversarial examples
* More effective than single‑step FGSM
* Exploits local linearity of ReLU‑based CNNs

**Findings**

* Clean accuracy remains unchanged
* Adversarial accuracy drops sharply as ε and iterations increase
* Models trained for more epochs show *slightly* improved robustness
* Attack remains highly effective in all cases

---

## Defense 1: Structured Pruning

Structured pruning removes entire convolutional filters to reduce model capacity.

**Method**

* ℓ₁‑norm structured pruning
* Applied to `layer4` of ResNet‑18
* Pruning ratios tested: **20%** and **40%**
* Followed by fine‑tuning for 6 epochs

**Results**

* Clean accuracy preserved (~99%)
* Adversarial accuracy remained **0.0%** under I‑FGSM
* Increasing pruning strength did not improve robustness

**Conclusion**

> Pruning reduces model size but does **not** meaningfully change gradient behavior exploited by iterative attacks.

---

## Defense 2: Adversarial Fine‑Tuning

To explicitly address adversarial vulnerability, adversarial fine‑tuning was applied on top of the pruned model.

**Training Setup**

* Adversarial examples generated on‑the‑fly using I‑FGSM
* Combined loss: clean loss + adversarial loss

**Experiments**

1. **Light adversarial training**

   * ε = 4/255, 3 iterations, λ = 0.5
   * Clean accuracy: ~93%
   * Adversarial accuracy: ~0.06%

2. **Stronger adversarial training**

   * ε = 8/255, 5 iterations, λ = 0.9
   * Clean accuracy dropped to ~83%
   * No meaningful robustness improvement

**Key Insight**

> Stronger adversarial training introduces a significant clean‑accuracy trade‑off without substantially improving robustness against strong white‑box attacks.

---

## Key Takeaways

* I‑FGSM is highly effective against ResNet‑18
* More training improves stability but not security
* Structured pruning preserves accuracy but offers no robustness
* Adversarial fine‑tuning yields limited gains with large trade‑offs
* Robust ML security requires defenses that fundamentally alter gradient behavior

---

## Future Work

* Compare against stronger attacks (PGD, MI‑FGSM)
* Evaluate black‑box attack transferability
* Combine pruning with gradient regularization
* Explore randomized smoothing or certified defenses

---

## Project Context

This project lies at the intersection of:

* **Cybersecurity**
* **Adversarial Machine Learning**
* **AI Safety & Trustworthy ML**

It demonstrates how modern neural networks can be systematically attacked and why defending them remains a challenging security problem.

---

## License

Apache License 2.0

---

## Authors

* Guus Branderhorst
* Lizzy‑Milou de Bruijn
* Kim E‑Shawn Brandon
