import sys
import os
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Importar nuestros módulos
from src.neural_network import RedNeuronal
from src.image_processing import load_and_preprocess_image, load_images_from_folder

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_model()

    def init_ui(self):
        self.setWindowTitle("Reconocimiento Facial Simple")
        self.layout = QVBoxLayout()

        # Etiqueta principal
        self.label = QLabel("Selecciona una imagen para predecir")
        self.layout.addWidget(self.label)

        # Área de imagen
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Barra de progreso
        self.progress = QProgressBar()
        self.progress.hide()
        self.layout.addWidget(self.progress)

        # Botones
        self.load_btn = QPushButton("Cargar Imagen")
        self.load_btn.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_btn)

        self.train_btn = QPushButton("Entrenar Modelo")
        self.train_btn.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_btn)

        self.setLayout(self.layout)

    def setup_model(self):
        # Usar un tamaño de imagen de 128x128
        self.tamanio_objetivo = (128, 128)
        tamanio_entrada = self.tamanio_objetivo[0] * self.tamanio_objetivo[1]  # 16384 píxeles
        self.red = RedNeuronal(
            tamanio_entrada=tamanio_entrada,
            tamanios_capas_ocultas=[512, 256, 128],  # Capas más grandes para manejar la entrada de 16384
            tamanio_salida=1
        )
        
        # Usar ruta relativa para el modelo
        self.ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_guardado.joblib")
        
        # cargar el modelo pre-entrenado
        if os.path.exists(self.ruta_modelo):
            if self.red.cargar_modelo(self.ruta_modelo):
                self.label.setText("Modelo pre-entrenado cargado correctamente")
                self.train_btn.setText("Re-entrenar Modelo")
            else:
                self.label.setText("Error al cargar el modelo pre-entrenado. Se usará un modelo nuevo.")
        else:
            self.label.setText("No se encontró un modelo pre-entrenado. Entrene el modelo primero.")

    def load_image(self):
        nombre_archivo, _ = QFileDialog.getOpenFileName(
            self, 'Abrir Imagen', '',
            'Imagenes (*.png *.jpg *.jpeg *.bmp)'
        )
        if nombre_archivo:
            try:
                img, img_plana = load_and_preprocess_image(
                    nombre_archivo, 
                    target_size=self.tamanio_objetivo
                )
                salida = self.red.predecir(img_plana.reshape(-1, 1))
                prediccion = salida[0][0]
                clase = "Coincide" if prediccion > 0.5 else "No coincide"
                self.label.setText(f"Resultado: {clase} ({prediccion:.2f})")

                # Mostrar imagen
                qimg = QImage(
                    img.data, img.shape[1], img.shape[0],
                    QImage.Format_Grayscale8
                )
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(
                    pixmap.scaled(400, 300, Qt.KeepAspectRatio)
                )
            except Exception as e:
                self.label.setText(f"Error al procesar la imagen: {str(e)}")

    def train_model(self):
        try:
            # Usar rutas relativas para las carpetas de imágenes
            ruta_base = os.path.dirname(__file__)
            print("Cargando imágenes...")
            
            positivos = load_images_from_folder(
                os.path.join(ruta_base, "imagenes_mias"),
                1,
                self.tamanio_objetivo
            )
            print(f"Imágenes positivas cargadas: {len(positivos)}")
            if positivos:
                print(f"Dimensión de la primera imagen positiva: {positivos[0][0].shape}")

            negativos = load_images_from_folder(
                os.path.join(ruta_base, "imagenes_otras"),
                0,
                self.tamanio_objetivo
            )
            print(f"Imágenes negativas cargadas: {len(negativos)}")
            if negativos:
                print(f"Dimensión de la primera imagen negativa: {negativos[0][0].shape}")

            if not positivos or not negativos:
                self.label.setText("Error: No hay suficientes imágenes para entrenar")
                return

            datos = positivos + negativos
            random.shuffle(datos)

            # Extraer x e y de los datos
            datos_x = []
            datos_y = []
            for x, y in datos:
                datos_x.append(x)  # x ya está en formato (-1, 1) desde load_and_preprocess_image
                datos_y.append([[y]])  # Convertir y a formato (1, 1)
            
            print(f"Dimensión del primer dato_x: {datos_x[0].shape}")
            print(f"Dimensión del primer dato_y: {datos_y[0]}")

            # Mostrar la barra de progreso
            self.progress.setMaximum(500)
            self.progress.show()
            
            # Entrenar el modelo
            self.red.entrenar(datos_x, datos_y, epocas=500, tasa_aprendizaje=0.1)
            
            # Guardar el modelo
            self.red.guardar_modelo(self.ruta_modelo)
            self.label.setText("Modelo entrenado y guardado con éxito")
            
            # Ocultar la barra de progreso
            self.progress.hide()
            
        except Exception as e:
            self.label.setText(f"Error durante el entrenamiento: {str(e)}")
            self.progress.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = App()
    ventana.show()
    sys.exit(app.exec_())
