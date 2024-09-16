import os
from PIL import Image
import numpy as np
from heatmap import heatmap_analysis  
from gpt_connect import OpenAIHandler

def save_image(image, path, size=(512, 512)):
    """Save the image to the specified path with the given size."""
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Resize the image
    newsize = (512, 512)
    im1 = pil_image.resize(newsize)
    # Save the resized image
    im1.save(path)

def main(image_path):
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image at {image_path} does not exist.")
        return
    
    # Perform heatmap analysis
    image_analysis, heatmap = heatmap_analysis(image_path)
    
    # Save heatmap to a local path
    heatmap_path = os.path.join('static', 'heatmap.png')
    os.makedirs('static', exist_ok=True)
    save_image(heatmap, heatmap_path)

    # Print the results of image analysis
    dominant_color, orange_value, saturation_mean, value_mean, brightness, contrast, saturation_category, value_category = image_analysis.extract_features()
    print(f"Dominant Color: {dominant_color}")
    print(f"Orange Value: {orange_value}")
    print(f"Saturation Category: {saturation_category}")
    print(f"Brightness: {value_category}")

    print(f"Heatmap saved at: {heatmap_path}")

    # Optionally, perform GPT analysis
    handler = OpenAIHandler()
    insights = handler.analyze_heatmap(heatmap_path)
    print(f"Insights: {insights}")

if __name__ == '__main__':
    # Example usage with a local image path
    image_path = 'Maxxx Energy/Imagen 2.png'  
    main(image_path)

