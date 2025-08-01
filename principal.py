import os
import numpy as np
from PIL import Image
from red_neuronal import RedNeuronal

# Tamaño de entrada (700x600)
ANCHO = 700
ALTO = 600
INPUT_SIZE = ANCHO * ALTO

# Parámetros de la red neuronal (ajústalos aquí directamente)
CAPAS_OCULTAS = 1
NEURONAS_POR_CAPA = 64
EPOCAS = 300

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
            try:
                imagen = cargar_imagen(ruta)
                datos.append(imagen)
                etiquetas.append(np.array([[1]]) if clase.lower() == "ia" else np.array([[0]]))
            except Exception as e:
                print(f"⚠️ Error al cargar imagen {archivo}: {e}")
    return datos, etiquetas

def menu():
    print("----- ENTRENAMIENTO DE RED NEURONAL -----")
    print("\nCargando dataset...")
    datos, etiquetas = cargar_dataset("dataset")

    if not datos:
        print("❌ Error: No se encontraron imágenes válidas en el dataset.")
        return

    # Crear red neuronal
    red = RedNeuronal(CAPAS_OCULTAS, NEURONAS_POR_CAPA, INPUT_SIZE, 1)

    # Dividir datos en entrenamiento y prueba
    datos_entrenamiento, etiquetas_entrenamiento, datos_prueba, etiquetas_prueba = red.dividir_datos(datos, etiquetas)

    print(f"Total de datos: {len(datos)}")
    print(f"Datos de entrenamiento: {len(datos_entrenamiento)}")
    print(f"Datos de prueba: {len(datos_prueba)}")

    print("\nEntrenando modelo...\n")
    red.entrenar(datos_entrenamiento, etiquetas_entrenamiento, EPOCAS)
    print("\n✅ Entrenamiento completo.\n")

    print("Evaluando en datos de prueba...")
    aciertos = 0
    for entrada, etiqueta in zip(datos_prueba, etiquetas_prueba):
        salida = red.predecir(entrada)
        pred = 1 if salida[0][0] > 0.85 else 0
        if pred == etiqueta[0][0]:
            aciertos += 1
    precision = aciertos / len(datos_prueba)
    print(f"🎯 Precisión en prueba: {precision * 100:.2f}%\n")

    while True:
        print("----- MENÚ DE PRUEBA -----")
        ruta = input("Ingresa la ruta de una imagen para predecir (o 'salir'): ")
        if ruta.lower() == "salir":
            break
        if not os.path.isfile(ruta):
            print("Ruta inválida. Intenta de nuevo.")
            continue

        try:
            entrada = cargar_imagen(ruta)
            resultado = red.predecir(entrada)
            probabilidad = float(resultado[0][0])

            print(f"\nProbabilidad: {probabilidad:.4f}")
            if probabilidad > 0.85:
                print("✅ Es tu rostro")
            else:
                print("❌ No es tu rostro")
        except Exception as e:
            print(f"⚠️ No se pudo procesar la imagen: {e}")
        print("\n")

if __name__ == "__main__":
    menu()
