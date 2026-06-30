from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import json
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading AI model... please wait!")

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Download ImageNet labels (only if not already present)
if not os.path.exists("labels.json"):
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

print("Model ready!")


@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            image_path = save_path

            try:
                img = Image.open(save_path).convert("RGB")
                tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    output = model(tensor)

                probs = torch.nn.functional.softmax(output[0], dim=0)
                top5 = torch.topk(probs, 5)

                predictions = [
                    {"label": labels[idx], "confidence": round(prob.item() * 100, 2)}
                    for prob, idx in zip(top5.values, top5.indices)
                ]
            except Exception as e:
                error = str(e)
        else:
            error = "Please choose an image first."

    return render_template(
        "classifier.html",
        predictions=predictions,
        image_path=image_path,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)