# AI Projects — Dania Yasir

A collection of beginner-to-intermediate AI/ML projects built while learning Python-based AI development. Each project is self-contained and runs locally.

## Projects

### 1. AI Chatbot
A conversational chatbot powered by the Groq API (LLaMA 3.3 70B), available both as a terminal app and a Flask web app.

**Files:** `chatbot.py` (terminal version), `app.py` + `templates/index.html` (web version)

**Run it:**
```bash
pip install flask groq
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

---

### 2. Sentiment Analyzer
A terminal tool that analyzes the sentiment of any text input — classifying it as positive, negative, or neutral, with polarity and subjectivity scores.

**Files:** `sentiment.py`

**Tech:** TextBlob / NLTK

**Run it:**
```bash
pip install textblob
python sentiment.py
```

---

### 3. Image Classifier
A computer vision app that classifies any uploaded image into one of 1,000 ImageNet categories, returning the top 5 predictions with confidence scores. Available as a terminal tool and a Flask web app with a custom UI.

**Files:** `classifier.py` (terminal version), `classifier_app.py` + `templates/classifier.html` (web version)

**Tech:** PyTorch, torchvision (pretrained ResNet-50)

**Run it (web version):**
```bash
pip install flask torch torchvision pillow
python classifier_app.py
```
Then open `http://127.0.0.1:5000` in your browser, upload an image, and view the predictions.

> **Note:** This model is trained on ImageNet, which covers objects, animals, and everyday items — it does not include "person" as a category, so it won't recognize human faces. This is expected behavior, not a bug.

---

## Tech Stack
- **Language:** Python
- **Web framework:** Flask
- **AI/ML:** PyTorch, torchvision, TextBlob, Groq API (LLaMA 3.3)
- **Frontend:** HTML, CSS, vanilla JS

## Setup
Clone the repo and install dependencies for the project you want to run (see individual sections above). Each script is independent — you only need to install the packages relevant to the project you're running.

```bash
git clone <your-repo-url>
cd <repo-folder>
```

## Roadmap
This repo represents Phase 1 (Foundation) of a longer AI learning roadmap. Upcoming work includes deep learning fundamentals and more advanced NLP/computer vision projects.

## Author
**Dania Yasir**
[LinkedIn](#) · [GitHub](#)
