from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

class processimage:
    def __init__(self, x, model, target_class, gradcam):
        self.image = Image.open(x).convert("RGB")
        self.model = model
        self.target_class = target_class
        self.gradcam = gradcam

    def preprocess_images(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(self.image).unsqueeze(0)
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return img_tensor.to(device)

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_RAINBOW)
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        plt.imshow(np.uint8(255 * cam))
        plt.axis('off')
        plt.show()

    def process_and_visualize(self):
        img_tensor = self.preprocess_images()
        heatmap = self.gradcam.generate_cam(img_tensor)
        img_array = np.array(self.image) / 255.0
        self.show_cam_on_image(img_array, heatmap)
        return heatmap