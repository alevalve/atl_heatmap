# ReadMe

## Implementation

### Code Structure

1. **`gradcamexpanded.py`**:
   - Contains a class named `GradCamExpanded`. This script implements a mathematical method used to generate the heatmap

2. **`heatmap.py`**:
   - Contains a function called `heatmap_analysis`. This function manages the different classes and orchestrates the logical processing of the image.

3. **`image_analysis.py`**:
   - Contains a class named `ImageAnalysis`. This script handles the descriptive analysis of the image. Within this class, there are 7 functions that perform the analysis.

4. **`process_image.py`**:
   - Contains a class named `ProcessImage`. This script is responsible for the transformation and preprocessing of the image.

5. **`gpt_connect.py`**:
   - Contains a class named `OpenAIHandler`. This script handles sending the image with the heatmap to GPT-4. It also includes the prompt to ensure the model can analyze the image and return the expected results.

6. **`main.py`**:
   - The main script of the project. It returns the results of the analyses conducted and the feedback from ChatGPT 4 on the heatmap.
