from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

class imageanalysis():
    def __init__(self, x):
        self.image = Image.open(x).convert("RGB")

    def get_average_color(self):
        np_image = np.array(self.image)
        average_color = np.mean(np_image, axis=(0, 1))
        return average_color

    def get_dominant_color(self):
        average_color = self.get_average_color()
        colors = {'red': average_color[0], 'green': average_color[1], 'blue': average_color[2]}
        dominant_color = max(colors, key=colors.get)
        return dominant_color

    def get_color_distribution(self):
        np_image = np.array(self.image)
        total_pixels = np_image.shape[0] * np_image.shape[1]
        red_pixels = np.sum(np_image[:,:,0]) / total_pixels
        green_pixels = np.sum(np_image[:,:,1]) / total_pixels
        blue_pixels = np.sum(np_image[:,:,2]) / total_pixels
        return {'red': red_pixels, 'green': green_pixels, 'blue': blue_pixels}

    def get_hsv_values(self):
        image_array = np.array(self.image)
        image_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image_hsv)
        
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(image_array)
        plt.title('Original Image')
        plt.show()

        # Calculate the mean saturation and value
        mean_saturation = np.mean(s)
        mean_value = np.mean(v)

        # Determine the saturation category
        if mean_saturation < 51:
            saturation_category = "Very Low"
        elif mean_saturation < 102:
            saturation_category = "Low"
        elif mean_saturation < 153:
            saturation_category = "Medium"
        elif mean_saturation < 204:
            saturation_category = "High"
        else:
            saturation_category = "Very High"

        # Determine the value category
        if mean_value < 51:
            value_category = "Very Low"
        elif mean_value < 102:
            value_category = "Low"
        elif mean_value < 153:
            value_category = "Medium"
        elif mean_value < 204:
            value_category = "High"
        else:
            value_category = "Very High"

        return mean_saturation, mean_value, saturation_category, value_category

    def orange_values(self):
        image_array = np.array(self.image)
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
        orange_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        orange_coverage = (orange_pixels / total_pixels) * 100

           # Determine the value category
        if orange_coverage < 20:
            value_category = "Very Low"
        elif orange_coverage < 40:
            value_category = "Low"
        elif orange_coverage < 70:
            value_category = "Medium"
        elif orange_coverage < 90:
            value_category = "High"
        else:
            value_category = "Very High"

        return value_category

    def get_brightness_contrast(self):
        np_image = np.array(self.image)
        brightness = np.mean(np_image)
        contrast = np.std(np_image)
        return brightness, contrast


    def extract_features(self):
        dominant_color = self.get_dominant_color()
        color_distribution = self.get_color_distribution()
        orange_value = self.orange_values()
        saturation_mean, value_mean, saturation_category, value_category = self.get_hsv_values()
        brightness, contrast = self.get_brightness_contrast()
        return dominant_color, orange_value, saturation_mean, value_mean, brightness, contrast, saturation_category, value_category
    
