# 🌾 Paddy Leaf Disease Detection using Deep Learning

An AI-powered Paddy Leaf Disease Detection System built using **TensorFlow/Keras** and **MobileNetV2** for identifying diseases in paddy leaves through image classification.

This project helps farmers and agricultural researchers detect crop diseases early and improve crop productivity using Deep Learning and Computer Vision.

---

# 📌 Project Overview

Paddy crops are affected by several diseases that reduce crop quality and yield. Manual disease detection is time-consuming and requires expert knowledge.

This project uses a Deep Learning model trained on paddy leaf images to automatically classify diseases from uploaded images.

The system uses **Transfer Learning with MobileNetV2**, making the model lightweight, fast, and efficient for real-time prediction.

---

# 🚀 Features

✅ Detects multiple paddy leaf diseases  
✅ Deep Learning based image classification  
✅ Transfer Learning using MobileNetV2  
✅ Real-time disease prediction  
✅ Image preprocessing and augmentation  
✅ Lightweight and efficient model  
✅ User-friendly prediction system  

---

# 🦠 Diseases Detected

| Disease Class | Description |
|---|---|
| bacterial_leaf_blight | Bacterial disease causing yellowing and drying |
| brown_spot | Brown lesions on leaves |
| healthy | Healthy paddy leaf |
| leaf_blast | Fungal disease causing blast-shaped spots |
| leaf_scald | Drying/scalding effect on leaf edges |
| narrow_brown_spot | Narrow brown streak disease |

---

# 🧠 Model Used

## MobileNetV2

The project uses **MobileNetV2**, a lightweight Convolutional Neural Network (CNN) pretrained on the ImageNet dataset.

### Advantages:
- Faster training
- Lightweight architecture
- Suitable for real-time applications
- High accuracy with low computational cost

---

# ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# 📂 Dataset Structure

```bash
Riceleafdisease/
│
├── train/
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   ├── healthy/
│   ├── leaf_blast/
│   ├── leaf_scald/
│   └── narrow_brown_spot/
│
├── val/
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   ├── healthy/
│   ├── leaf_blast/
│   ├── leaf_scald/
│   └── narrow_brown_spot/
```

---

# 🏗️ Project Structure

```bash
paddyleafdisease/
│
├── Riceleafdisease/
├── static/
├── templates/
├── train_model.py
├── app.py
├── predict.py
├── requirements.txt
├── best_paddy_model.h5
├── final_paddy_model.h5
└── README.md
```

---

# 🔄 Workflow

## Step 1 — Data Collection
Collect paddy leaf images for different disease categories.

## Step 2 — Data Preprocessing
Images are:
- Resized to 224×224
- Normalized
- Augmented

## Step 3 — Model Training
MobileNetV2 is used as a feature extractor with custom classification layers.

## Step 4 — Validation
The model is validated using validation datasets.

## Step 5 — Prediction
Users upload an image and the model predicts the disease class.

---

# 📸 Data Augmentation

To improve model performance and reduce overfitting, the following techniques are used:

- Rotation
- Zooming
- Width shifting
- Height shifting
- Horizontal flipping
- Shearing

---

# 🧩 Model Architecture

```text
Input Image (224x224)
        ↓
MobileNetV2 (Pretrained on ImageNet)
        ↓
GlobalAveragePooling2D
        ↓
Dropout Layer (0.3)
        ↓
Dense Layer (Softmax)
        ↓
Disease Prediction
```

---

# 📈 Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 25 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |

---

# 💻 Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/DeepakVelmurugan18/paddyleafdisease.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd paddyleafdisease
```

---

## 3️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
```

---

## 4️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python train_model.py
```

The trained model will be saved as:

```bash
final_paddy_model.h5
```

---

# ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```bash
http://127.0.0.1:5000
```

---

# 📷 Prediction Process

1. Upload paddy leaf image
2. Image preprocessing
3. Model prediction
4. Disease class displayed
5. Confidence score shown

---

# 🌍 Real-World Applications

- Smart Agriculture
- Precision Farming
- Crop Monitoring
- Agricultural Research
- Farmer Assistance Systems

---

# 📊 Future Improvements

- Add more disease classes
- Improve accuracy using larger datasets
- Deploy on cloud platforms
- Android/mobile application support
- Real-time camera detection
- Multilingual support

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

---

# 👨‍💻 Author

## Deepak V

- GitHub: https://github.com/DeepakVelmurugan18
- LinkedIn: https://www.linkedin.com/in/deepak-v-18/

---

# ⭐ Support

If you found this project useful:

⭐ Star this repository  
🍴 Fork the project  
📢 Share with others  

---

# 📜 License

This project is developed for educational and research purposes.
