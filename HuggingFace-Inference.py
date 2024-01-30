'''text to image demonstration of using Hugging Face. More details can be found at https://huggingface.co/docs/huggingface_hub/guides/inference#legacy-inferenceapi-client'''
import os
from huggingface_hub import InferenceClient

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

client = InferenceClient(model="prompthero/openjourney-v4")
image = client.text_to_image("An astronaut riding a horse on the moon.")
image.save("astronaut.png")