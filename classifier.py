import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import json
import os

print("=" * 50)
print("   IMAGE CLASSIFIER — by Dania Yasir")
print("=" * 50)
print("Loading AI model... please wait!\n")

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Download ImageNet labels
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
urllib.request.urlretrieve(labels_url, "labels.json")
with open("labels.json") as f:
    labels = json.load(f)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Model ready! Let's classify some images!\n")

while True:
    path = input("Enter image path (or 'quit'): ").strip()
    
    if path.lower() == "quit":
        print("Goodbye!")
        break
    
    if not os.path.exists(path):
        print("File not found! Please check the path.\n")
        continue
    
    try:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
        
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5 = torch.topk(probs, 5)
        
        print("\n── TOP 5 PREDICTIONS ──")
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"  {i+1}. {labels[idx]:25s} {prob.item()*100:.2f}%")
        print("─" * 30 + "\n")
        
    except Exception as e:
        print(f"Error: {e}\n")