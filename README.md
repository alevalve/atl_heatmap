# Heatmap Analysis for ATL Posts

## Implementation

## Project Description:
This project aims to develop an algorithm that identifies the parts of an image or visual post that are most likely to attract the attention of consumers. The goal is to optimize visual elements, reduce the need for post-launch changes, and understand how a visual is perceived from different positions and distances. The algorithm also generates recommendations for highlighting specific areas of interest.

## Team Members:
- Alexander Valverde

## Objectives:
- Develop a recognition model to identify key areas of a post that are of high relevance to consumers, whether seen on the street or on social media.
- Optimize visual elements to minimize the cost of printing materials that may require changes post-launch.
- Analyze how the artwork would appear from various positions and distances, identifying elements that remain important regardless of the viewing angle.
- Provide recommendations on the most important elements of a visual and suggest possible changes to highlight specific areas.

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
