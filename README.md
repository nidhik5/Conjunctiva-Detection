#  Conjunctiva-Detection: Eye Segmentation + Grad-CAM

##  Context: Eye Conjunctiva Segmentation Dataset

This project implements a **U-Net architecture in PyTorch** to segment the conjunctiva region in eye images, achieving a Dice score of **0.86** after 50 epochs.

**The core feature:** It includes a complete implementation of **Grad-CAM (Gradient-weighted Class Activation Mapping)**, a critical Explainable AI (XAI) technique, to visually explain *which regions* the model focuses on during segmentation predictions.

##  XAI Demo: Before and After

The primary output is a visual demonstration of model explainability. It shows the original eye image, the predicted segmentation mask, and the Grad-CAM heatmap overlaid, highlighting the exact regions the model focused on during the segmentation decision-making process.

![XAI Visualization Output](https://github.com/nidhik5/Conjunctiva-Detection/blob/main/assets/Screenshot%202025-12-10%20192830.png)

##  What This Code Does

### 1. U-Net Segmentation Architecture

* **Goal:** Binary Segmentation (Conjunctiva vs. Background).
* **Structure:** Classic U-Net encoder-decoder architecture with skip connections, implemented from scratch in PyTorch.
* **Layers:** Uses multiple `Conv2d`, `ReLU`, and `MaxPool2d` layers in the encoder, with `ConvTranspose2d` for upsampling in the decoder.
* **Input/Output:** Takes 256×256 RGB images and outputs single-channel segmentation masks.

### 2. Explainability (The Grad-CAM Pipeline)

The script contains the `GradCAM` class, which showcases advanced PyTorch usage for model introspection:

| Technique | Goal |
| :--- | :--- |
| **`register_forward_hook()`** | Captures intermediate feature map activations from target convolutional layers during forward pass. |
| **`register_backward_hook()`** | Captures gradients flowing back through the target layer during backpropagation. |
| **`torch.mean(gradients)`** | Computes global average pooling of gradients to weight the importance of each feature channel. |
| **Heatmap Superimposition** | Rescales and overlays the calculated activation map onto the original eye image, providing visual proof of the model's attention. |

### 3. Training Pipeline

* **Dataset:** 547 eye images with corresponding conjunctiva masks (80/20 train/validation split).
* **Preprocessing:** Images resized to 256×256, normalized to [0,1], masks extracted from red channel with threshold.
* **Augmentation:** Horizontal flips (p=0.5), random brightness/contrast (p=0.2), rotation ±15° (p=0.3) using Albumentations.
* **Loss Function:** Combined BCE + Dice Loss (0.5 weight each) for optimal segmentation performance.
* **Optimizer:** Adam with learning rate 1e-4.

### Key Metrics

* **Final Training Dice Score:** 0.8601
* **Image Resolution:** 256×256
* **Batch Size:** 8
* **Training Dataset:** 437 images (after 80/20 split)
* **Validation Dataset:** 110 images

##  Why Grad-CAM Matters for Medical Imaging

In medical applications, **model interpretability is crucial**:

* **Clinical Trust:** Doctors need to understand *why* a model made a prediction before acting on it.
* **Error Detection:** Grad-CAM can reveal when a model focuses on irrelevant artifacts (e.g., image borders, equipment).
* **Model Debugging:** Comparing Grad-CAM across different layers helps identify where the network learns meaningful vs. spurious features.
* **Regulatory Compliance:** Many medical AI regulations require explainability components.

This implementation demonstrates that the model correctly focuses on the conjunctiva region rather than irrelevant background features.

##  References

* **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
* **Grad-CAM:** Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
* **Dice Loss:** Milletari, F., et al. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV.

##  License

MIT License
