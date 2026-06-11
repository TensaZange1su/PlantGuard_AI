#gradcam_utils.py
import torch
import cv2
import numpy as np
from torchvision import transforms


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):

        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        loss = output[:, class_idx]

        self.model.zero_grad()

        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=[2,3], keepdim=True)

        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224,224))

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam