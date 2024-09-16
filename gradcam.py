import torch 
import numpy as np
import cv2

class gradcamexpanded:
    def __init__(self, model, target_layer, lambd):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
        self.lambd = lambd

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output.mean()  # Use the mean of the output to get class-agnostic gradients
        target.backward(retain_graph=True)

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2)) # Shape = (,128)

        cam = np.zeros(activations.shape[1:], dtype=np.float32) # Shape = (55,55)

         # Calculate the CAM using neighboring values
        for i, w in enumerate(weights):
            activation = activations[i]
            neighbors = [activation]
            if i > 0:
                neighbors.append(activations[i - 1])
            if i < len(weights) - 1:
                neighbors.append(activations[i + 1])
            cam += (w * activation + self.lambd * np.mean(neighbors, axis=0)) / 2

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))

        # Normalize the CAM, adding a small epsilon to avoid division by zero
        epsilon = 1e-8
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + epsilon)

        return cam