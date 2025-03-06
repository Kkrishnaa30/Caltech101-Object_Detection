# 📌 Caltech101-Object_Detection  

## 🚀 Project Overview  
This project focuses on **object detection** using a subset of the **Caltech101** dataset. The aim is to train a deep learning model to detect and classify three specific object categories:  

🔹 **Butterfly** 🦋  
🔹 **Dalmatian** 🐶  
🔹 **Dolphin** 🐬  

The model is built using **transfer learning** with pre-trained architectures like **VGG16, VGG19, or ResNet**, fine-tuned for **bounding box prediction** and **class classification**.  

---

## 📂 Dataset  
📌 The dataset is derived from **Caltech101** and includes annotated images.  
📌 Bounding box annotations are pre-processed and normalized for model training.  

---
## 🛠 Implementation Steps  

✅ **Data Preprocessing**  
🔹 Extract image filenames and bounding box annotations.  
🔹 Normalize bounding box coordinates.  
🔹 Convert class labels into a one-hot encoded format.  

✅ **Model Training**  
🔹 Use **transfer learning** with **VGG16, VGG19, or ResNet**.  
🔹 Modify the architecture to predict **bounding boxes** and **class labels**.  
🔹 Implement **custom loss functions** for bounding box regression and classification.  

✅ **Model Evaluation**  
🔹 Predict bounding boxes and class labels on test images.  
🔹 Assess performance using **IoU (Intersection over Union)** and classification accuracy.  
🔹 Save the trained model for further inference.  

---

## ▶ How to Run  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/Kkrishnaa30/Caltech101-Object_Detection.git
```

2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

3️⃣ **Run Jupyter Notebook**  
```bash
jupyter notebook
```

4️⃣ **Open and execute** `Caltech101_Object_Detection Reworked.ipynb`.  

---

## 📊 Results  
✅ The model successfully detects and classifies objects in images.  
✅ Performance evaluation includes **bounding box accuracy, classification accuracy, and visualized predictions**.  

---

## 🔮 Future Enhancements  
🚀 **Improve accuracy** with more advanced architectures like **Faster R-CNN**.  
📈 **Optimize model hyperparameters** using **Keras Tuner**.  
🖼 **Expand dataset** to include more object categories for better generalization.  

