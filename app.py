import sys
import os
import numpy as np
import cv2
import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage

# Red neuronal simple
class RedNeuronal:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []

        prev_size = input_size
        for size in hidden_sizes:
            self.weights.append(np.random.randn(size, prev_size) * 0.01)
            self.biases.append(np.zeros((size, 1)))
            prev_size = size

        self.weights.append(np.random.randn(output_size, prev_size) * 0.01)
        self.biases.append(np.zeros((output_size, 1)))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        a = x
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)

        return activations, zs

    def backward(self, x, y, activations, zs, lr):
        delta = (activations[-1] - y) * activations[-1] * (1 - activations[-1])
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            sp = activations[-l] * (1 - activations[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        self.weights = [w - lr * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - lr * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, x_data, y_data, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            for x, y in zip(x_data, y_data):
                x = x.reshape(-1, 1)
                y = np.array([[y]])
                activations, zs = self.forward(x)
                self.backward(x, y, activations, zs, lr)

    def predict(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return a

    def save_model(self, path):
        np.savez(path, weights=np.array(self.weights, dtype=object),
                biases=np.array(self.biases, dtype=object))

    def load_model(self, path):
        data = np.load(path, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']

# GUI principal
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocimiento Facial Simple")
        self.layout = QVBoxLayout()
        self.label = QLabel("Selecciona una imagen para predecir")
        self.layout.addWidget(self.label)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.load_btn = QPushButton("Cargar Imagen")
        self.load_btn.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_btn)

        self.train_btn = QPushButton("Entrenar Modelo")
        self.train_btn.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_btn)

        self.setLayout(self.layout)

        self.nn = RedNeuronal(600*800, [64], 1)
        if os.path.exists("modelo_guardado.npz"):
            self.nn.load_model("modelo_guardado.npz")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Abrir Imagen', '', 'Imagenes (*.png *.jpg *.bmp)')
        if file_name:
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (800, 600))
            img_flat = img.reshape(-1, 1) / 255.0
            output = self.nn.predict(img_flat)
            pred = output[0][0]
            clase = "Coincide" if pred > 0.5 else "No coincide"
            self.label.setText(f"Resultado: {clase} ({pred:.2f})")

            # Mostrar imagen
            qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap.scaled(400, 300))

    def train_model(self):
        positivos = self.load_images("imagenes_mias", 1)
        negativos = self.load_images("imagenes_otras", 0)

        data = positivos + negativos
        random.shuffle(data)

        x_data = [x for x, _ in data]
        y_data = [y for _, y in data]

        self.nn.train(x_data, y_data, epochs=500, lr=0.1)
        self.nn.save_model("modelo_guardado.npz")
        self.label.setText("Modelo entrenado y guardado con éxito.")

    def load_images(self, folder, label):
        data = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (800, 600))
            img_flat = img.reshape(-1) / 255.0
            data.append((img_flat, label))
        return data

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = App()
    ventana.show()
    sys.exit(app.exec_())
