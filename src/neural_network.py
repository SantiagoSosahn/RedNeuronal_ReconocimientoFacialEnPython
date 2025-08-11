import numpy as np
from joblib import dump, load

class RedNeuronal:
    def __init__(self, tamanio_entrada, tamanios_capas_ocultas, tamanio_salida):
        self.pesos = []
        self.sesgos = []

        tamanio_previo = tamanio_entrada
        for tamanio in tamanios_capas_ocultas:
            self.pesos.append(np.random.randn(tamanio, tamanio_previo) * 0.01)
            self.sesgos.append(np.zeros((tamanio, 1)))
            tamanio_previo = tamanio

        self.pesos.append(np.random.randn(tamanio_salida, tamanio_previo) * 0.01)
        self.sesgos.append(np.zeros((tamanio_salida, 1)))

    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        """Implementación de ReLU: f(x) = max(0, x)"""
        return np.maximum(0, z)
    
    def derivada_relu(self, z):
        """Derivada de ReLU: 1 si x > 0, 0 si x ≤ 0"""
        return np.where(z > 0, 1, 0)

    def propagacion_adelante(self, x):
        a = x
        activaciones = [x]
        valores_z = []

        # Usamos ReLU para capas ocultas
        for i, (w, b) in enumerate(zip(self.pesos[:-1], self.sesgos[:-1])):
            z = np.dot(w, a) + b
            valores_z.append(z)
            a = self.relu(z)
            activaciones.append(a)
        
        # Usamos sigmoide solo para la capa de salida (para clasificación binaria)
        z = np.dot(self.pesos[-1], a) + self.sesgos[-1]
        valores_z.append(z)
        a = self.sigmoide(z)  # Mantenemos sigmoide en la última capa
        activaciones.append(a)

        return activaciones, valores_z

    def propagacion_atras(self, x, y, activaciones, valores_z, tasa_aprendizaje):
        # Para la capa de salida (sigmoide)
        delta = (activaciones[-1] - y) * activaciones[-1] * (1 - activaciones[-1])
        gradiente_sesgos = [np.zeros_like(b) for b in self.sesgos]
        gradiente_pesos = [np.zeros_like(w) for w in self.pesos]

        gradiente_sesgos[-1] = delta
        gradiente_pesos[-1] = np.dot(delta, activaciones[-2].T)

        # Para las capas ocultas (ReLU)
        for l in range(2, len(self.pesos) + 1):
            z = valores_z[-l]
            # Usamos la derivada de ReLU en lugar de la derivada de sigmoide
            sp = self.derivada_relu(z)
            delta = np.dot(self.pesos[-l + 1].T, delta) * sp
            gradiente_sesgos[-l] = delta
            gradiente_pesos[-l] = np.dot(delta, activaciones[-l - 1].T)

        self.pesos = [w - tasa_aprendizaje * nw for w, nw in zip(self.pesos, gradiente_pesos)]
        self.sesgos = [b - tasa_aprendizaje * nb for b, nb in zip(self.sesgos, gradiente_sesgos)]

    def entrenar(self, datos_x, datos_y, epocas=1000, tasa_aprendizaje=0.1):
        print("Iniciando entrenamiento...")
        print(f"Dimensiones de la primera capa: {self.pesos[0].shape}")
        print(f"Dimensión del primer dato_x en entrenar: {datos_x[0].shape}")
        print(f"Dimensión del primer dato_y en entrenar: {np.array(datos_y[0]).shape}")
        
        for epoca in range(epocas):
            for x, y in zip(datos_x, datos_y):
                # x ya debe venir en formato (16384, 1)
                # y ya debe venir en formato (1, 1)
                try:
                    activaciones, valores_z = self.propagacion_adelante(x)
                    self.propagacion_atras(x, y, activaciones, valores_z, tasa_aprendizaje)
                except ValueError as e:
                    print(f"Error en propagación: {e}")
                    print(f"Dimensiones de x: {x.shape}")
                    print(f"Dimensiones de y: {np.array(y).shape}")
                    raise

    def predecir(self, x):
        a = x
        # Usar ReLU para capas ocultas
        for w, b in zip(self.pesos[:-1], self.sesgos[:-1]):
            z = np.dot(w, a) + b
            a = self.relu(z)
        # Usar sigmoide solo para la capa de salida
        z = np.dot(self.pesos[-1], a) + self.sesgos[-1]
        return self.sigmoide(z)

    def guardar_modelo(self, ruta):
        """Guarda el modelo usando joblib"""
        datos_modelo = {
            'pesos': self.pesos,
            'sesgos': self.sesgos
        }
        dump(datos_modelo, ruta)

    def cargar_modelo(self, ruta):
        """Carga el modelo usando joblib"""
        try:
            datos_modelo = load(ruta)
            self.pesos = datos_modelo['pesos']
            self.sesgos = datos_modelo['sesgos']
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return False
