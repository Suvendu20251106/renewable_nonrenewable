# Waste Detection and Classification App ♻️

This is a Streamlit-based web app for classifying waste items as:
- Non-biodegradable
- Biodegradable - Renewable
- Biodegradable - Non-renewable

## 🚀 Features
- Upload an image of waste and classify it.
- Earn recycling points for renewable biodegradable waste.
- Built using **TensorFlow, Streamlit, and Pillow**.

## 📦 Setup
1. Clone this repo:
   ```bash
   git clone <repo-url>
   cd waste_classification_app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained model file as `waste_classifier_model.h5` inside the `models/` folder.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🧩 Model
- The app expects a trained model `waste_classifier_model.h5`.
- You can train your own or request the authors.

## 📜 License
MIT License
