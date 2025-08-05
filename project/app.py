import numpy as np
import cv2
import pickle
import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# -------------------------------
# Red Neuronal desde cero
# -------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.01
            b = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        a = x.reshape(-1, 1)
        activations = [a]
        zs = []

        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ a + self.biases[i]
            a = relu(z)
            zs.append(z)
            activations.append(a)

        z = self.weights[-1] @ a + self.biases[-1]
        a = sigmoid(z)
        zs.append(z)
        activations.append(a)

        return activations, zs

    def backprop(self, x, y, lr=0.01):
        activations, zs = self.forward(x)

        dw = [0] * len(self.weights)
        db = [0] * len(self.biases)

        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        dw[-1] = delta @ activations[-2].T
        db[-1] = delta

        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            sp = relu_derivative(z)
            delta = self.weights[-l + 1].T @ delta * sp
            dw[-l] = delta @ activations[-l - 1].T
            db[-l] = delta

        for i in range(len(self.weights)):
            self.weights[i] -= lr * dw[i]
            self.biases[i] -= lr * db[i]

    def predict(self, x):
        a, _ = self.forward(x)
        return a[-1]

    def train(self, X, Y, epochs=500, lr=0.005):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                self.backprop(x, y, lr)
            if epoch % 100 == 0:
                preds = [self.predict(x) for x in X]
                loss = np.mean([(p - y) ** 2 for p, y in zip(preds, Y)])
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases}, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']

# -------------------------------
# Preprocesamiento de imágenes
# -------------------------------
def preprocess_image(path, size=(600, 800)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img.flatten().astype(np.float32) / 255.0
    return img

# -------------------------------
# Interfaz gráfica PyQt5
# -------------------------------
class FaceRecognitionApp(QWidget):
    def __init__(self, neural_net):
        super().__init__()
        self.setWindowTitle("Reconocimiento Facial")
        self.nn = neural_net

        self.image_label = QLabel("Carga una imagen para predecir")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("Cargar Imagen")
        self.button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Images (*.png *.jpg *.jpeg)")

        if file_path:
            # Mostrar imagen
            img = QPixmap(file_path).scaled(300, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(img)

            # Predecir
            input_data = preprocess_image(file_path)
            result = self.nn.predict(input_data)
            prob = float(result[0][0])

            label = "✔ Es tu rostro" if prob > 0.5 else "✘ No es tu rostro"
            self.result_label.setText(f"{label} ({prob:.2f})")

# -------------------------------
# Ejecución principal
# -------------------------------
def main():
    input_size = 600 * 800
    model_path = "modelo_entrenado.pkl"

    nn = NeuralNetwork(input_size=input_size, hidden_sizes=[128, 64], output_size=1)

    if os.path.exists(model_path):
        print("✅ Modelo cargado desde archivo")
        nn.load_model(model_path)
    else:
        print("⚠ Entrenando modelo por primera vez...")
        rostro = preprocess_image("mi_rostro.jpeg")
        no_rostro = preprocess_image("no_rostro.jpg")
        X = [rostro, no_rostro]
        Y = [np.array([[1]]), np.array([[0]])]
        nn.train(X, Y, epochs=500, lr=0.005)
        nn.save_model(model_path)
        print("✅ Modelo entrenado y guardado")

    # Lanzar GUI
    app = QApplication(sys.argv)
    window = FaceRecognitionApp(nn)
    window.resize(400, 500)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
