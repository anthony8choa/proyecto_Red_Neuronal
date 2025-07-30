import os
import numpy as np
from PIL import Image
from red_neuronal import RedNeuronal

# Tamaño de entrada (700x600)
ANCHO = 700
ALTO = 600
INPUT_SIZE = ANCHO * ALTO

def cargar_imagen(ruta):
    imagen = Image.open(ruta).convert('L')
    imagen = imagen.resize((ANCHO, ALTO))
    vector = np.array(imagen).reshape(-1, 1) / 255.0
    return vector

def cargar_dataset(directorio):
    datos = []
    etiquetas = []
    for clase in ["Ia", "No_ia"]:
        carpeta = os.path.join(directorio, clase)
        if not os.path.isdir(carpeta):
            continue
        for archivo in os.listdir(carpeta):
            ruta = os.path.join(carpeta, archivo)
            imagen = cargar_imagen(ruta)
            datos.append(imagen)
            etiquetas.append(np.array([[1]]) if clase == "Ia" else np.array([[0]]))
    return datos, etiquetas

def menu():
    print("----- ENTRENAMIENTO DE RED NEURONAL -----")
    capas_ocultas = int(input("¿Cuántas capas ocultas quieres usar? "))
    neuronas = int(input("¿Cuántas neuronas por capa oculta? "))
    epocas = int(input("¿Cuántas épocas de entrenamiento? "))

    print("\nCargando dataset y entrenando modelo...")
    datos, etiquetas = cargar_dataset("dataset")
    red = RedNeuronal(capas_ocultas, neuronas, INPUT_SIZE, 1)
    red.entrenar(datos, etiquetas, epocas)
    print("Entrenamiento completo.\n")

    while True:
        print("----- MENÚ DE PRUEBA -----")
        ruta = input("Ingresa la ruta de una imagen para predecir (o 'salir'): ")
        if ruta.lower() == "salir":
            break
        if not os.path.isfile(ruta):
            print("Ruta inválida. Intenta de nuevo.")
            continue

        entrada = cargar_imagen(ruta)
        resultado = red.predecir(entrada)
        probabilidad = float(resultado[0][0])

        print(f"\nProbabilidad: {probabilidad:.4f}")
        if probabilidad > 0.5:
            print("✅ Es tu rostro")
        else:
            print("❌ No es tu rostro")
        print("\n")

if __name__ == "__main__":
    menu()