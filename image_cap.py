from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Change this to your real image filename
img_path = "willy_monster.jpeg"

# Open the image and convert to RGB
image = Image.open(img_path).convert("RGB")

# Prepare inputs for captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate caption
outputs = model.generate(**inputs, max_length=50)

# Decode and print caption
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)