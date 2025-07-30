import numpy as np

class RedNeuronal:
    def __init__(self, capas_ocultas, neuronas_por_capa, entrada, salida):
        self.capas = []
        self.biases = []

        capa_anterior = entrada
        for _ in range(capas_ocultas):
            self.capas.append(np.random.randn(neuronas_por_capa, capa_anterior) * 0.01)
            self.biases.append(np.zeros((neuronas_por_capa, 1)))
            capa_anterior = neuronas_por_capa

        # Capa de salida
        self.capas.append(np.random.randn(salida, capa_anterior) * 0.01)
        self.biases.append(np.zeros((salida, 1)))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivada(self, z):
        return z > 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivada(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, x):
        activaciones = [x]
        zs = []

        for i in range(len(self.capas)-1):
            z = self.capas[i] @ activaciones[-1] + self.biases[i]
            zs.append(z)
            a = self.relu(z)
            activaciones.append(a)

        # Capa de salida
        z = self.capas[-1] @ activaciones[-1] + self.biases[-1]
        zs.append(z)
        a = self.sigmoid(z)
        activaciones.append(a)

        return activaciones, zs

    def backpropagation(self, x, y, learning_rate=0.01):
        activaciones, zs = self.forward(x)
        deltas = [None] * len(self.capas)

        # Error de salida
        delta = (activaciones[-1] - y) * self.sigmoid_derivada(zs[-1])
        deltas[-1] = delta

        # Backprop para capas ocultas
        for l in range(len(self.capas)-2, -1, -1):
            delta = self.capas[l+1].T @ deltas[l+1] * self.relu_derivada(zs[l])
            deltas[l] = delta

        # Actualizar pesos y bias
        for i in range(len(self.capas)):
            self.capas[i] -= learning_rate * deltas[i] @ activaciones[i].T
            self.biases[i] -= learning_rate * deltas[i]

    def entrenar(self, datos, etiquetas, epocas=10):
        for e in range(epocas):
            for x, y in zip(datos, etiquetas):
                self.backpropagation(x, y)

    def predecir(self, x):
        salida, _ = self.forward(x)
        return salida[-1]