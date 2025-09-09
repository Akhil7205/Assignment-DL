# Assignment-DL
# 🚗 Car Headlamp Defect Detection (Without CNN)

## 📌 Project Overview
This project aims to classify **car headlamps** into three categories **without using CNN-based models**.  

The defect categories are:  
- **Intact** – No defect  
- **Broken** – Cracked or damaged  
- **Moisture** – Water or condensation inside  

The model is built using a **fully connected deep learning architecture (Dense layers only, no CNNs)**.

---

## 📂 Dataset
- Images were collected manually from **Google Images** for each class:
  - `data/intact_img/` → Intact lamps  
  - `data/broken_img/` → Broken lamps  
  - `data/moisture_img/` → Moisture lamps  

### Preprocessing:
- Converted to **RGB**  
- Resized to **64x64** pixels  
- Normalized to range **[0, 1]**  
- Balanced using **data augmentation**:
  - Rotation  
  - Width & height shift  
  - Shear, zoom  
  - Horizontal flip  

---

## 🛠️ Model Architecture
used a **Multi-Layer Perceptron (MLP)** with dropout and batch normalization.

```python
model = Sequential([
    Flatten(input_shape=(64,64,3)),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dense(3, activation="softmax")
])
```

Loss Function: Sparse Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
---

## ⚠️ Challenges Faced

- **Small dataset size** → harder to generalize  
- **Imbalanced classes** → solved with augmentation  
- **No CNN restriction** → limited ability to capture spatial features  
- **Overfitting risk** → handled with dropout + batch normalization  

---

## ✅ Future Improvements

- Collect a **larger, balanced dataset**  
- Use **CNNs / Transfer Learning (ResNet, VGG, MobileNet)** for better accuracy 
- Add **object detection (Faster R-CNN, YOLO)** to locate defects, not just classify  
- Deploy model using **Flask / FastAPI API** for real-world testing  

---



