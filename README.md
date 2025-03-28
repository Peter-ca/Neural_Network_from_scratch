# 🧠 Handwritten Digit Recognizer

A simple Python app to draw and recognize digits using a custom neural network trained on the MNIST dataset.

---

## 📷 Preview
https://github.com/user-attachments/assets/61a02b0f-b74d-47ff-ac8a-afd5d5f618c9


---

## 🗂️ Files

- `NeuralNetwork.py` – Neural network logic & training
- `DrawingApp.py` – Tkinter GUI for drawing digits
- `mnist_train.csv` – MNIST training data (download link: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download)
- `weights_biases.pkl` – Trained model weights
- `number.png` – Image saved from the canvas

---

## 🚀 Run

1. **Train the model** (optional):
   ```bash
   python NeuralNetwork.py
   ```
2. **Predict handwritten digit**:
    ```bash
    python DrawingApp.py
    ```
---
## 📦 Requirements

- numpy
- pandas
- Pillow
