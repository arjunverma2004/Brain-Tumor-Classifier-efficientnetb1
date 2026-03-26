# 🧠 Brain Tumor MRI Classifier

An **AI-powered diagnostic web application** that classifies brain MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor. 

This project leverages **Deep Learning (Transfer Learning)** via a fine-tuned **EfficientNetB1** model to provide rapid, highly accurate predictions. It features an interactive, user-friendly interface built with **Streamlit**, designed to provide medical professionals and users with a reliable "second opinion" and a transparent breakdown of prediction probabilities.

👉 **Live Streamlit App:** [https://x-raydisease-prediction.streamlit.app/](https://brain-tumor-classifier-efficientnetb1.streamlit.app/)

---

## ✨ Key Features

* **Real-Time Classification:** Upload an MRI scan and get an instant prediction.
* **Multi-Class Detection:** Accurately distinguishes between 3 types of tumors (Glioma, Meningioma, Pituitary) and healthy brains (No Tumor).
* **Confidence Breakdown:** Provides a detailed bar chart showing the model's prediction probability for each of the four classes.
* **Accessible UI:** A clean, custom-styled interface that requires no technical expertise to use.

---

## 🛠️ Technology Stack

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Web Framework:** Streamlit
* **Data Manipulation & Visualization:** NumPy, Pandas
* **Image Processing:** Pillow (PIL)

---

## 📊 Dataset & Model Performance

### **The Dataset**
The model was trained on the comprehensive **Brain Tumor MRI Dataset** from Kaggle. Extensive data augmentation (random flips, rotations, translations, and zooms) was applied during training to prevent overfitting and ensure the model generalizes well to new, unseen scans.

### **Model Architecture**
* **Backbone:** `EfficientNetB1` (Pre-trained on ImageNet)
* **Custom Classification Head:** * Global Average Pooling 2D
  * Dropout (30%) for regularization
  * Dense layer with Softmax activation for 4-class probability output

### **Performance Metrics**
The fine-tuned model achieves state-of-the-art results on the testing set:
* **Test Accuracy:** `96.03%`
* **Test AUC:** `0.9949`

---

## 📁 Repository Structure

```text
├── app.py                         # Main Streamlit web application
├── Brain_tumor1.ipynb             # Jupyter Notebook with data pipeline, model training, and evaluation
├── brain_tumor_predictor.keras    # The saved, fine-tuned EfficientNetB1 model weights
├── requirements.txt               # List of Python dependencies
├── Test/                          # Folder containing sample MRI images for testing
└── README.md                      # Project documentation

```

