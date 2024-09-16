import torch
import torchvision.models as models
from gradcamexpanded import gradcamexpanded
from image_analysis import imageanalysis
from process_image import processimage

def heatmap_analysis(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.squeezenet1_1(pretrained=True).to(device)
    model.eval()

    target_layer = model.features[4]
    target_class = 0
    lambd = 0.5
    grad_cam = gradcamexpanded(model, target_layer, lambd)
    process_image_instance = processimage(image_path, model, target_class, grad_cam)
    image_analysis_instance = imageanalysis(image_path)
    heatmap = process_image_instance.process_and_visualize()

    return image_analysis_instance, heatmap
