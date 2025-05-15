from openai import OpenAI
from heatmap import heatmap_analysis
import os
import ssl
import certifi
import base64
import os
import requests

ssl._create_default_https_context = ssl._create_stdlib_context

class OpenAIHandler:
    def __init__(self):
        self.api_key = ""

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_heatmap(self, image_path):
        base64_image = self.encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze the image and provide insights about the image based on the heatmap, the important part of the image are the bright ones. You need to explain what are the areas that based on the input are going to have more visual importance (the brighter areas and red ones are the important ones). Be explicit and provide the areas that you consider can be improved to get more visuality based on the output. Provide also the areas (mention the text is there is) and what improvements you recommend and how to positionate the post on streets or highways to obtain good results. (give the answer in Spanish)"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()


        return response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
