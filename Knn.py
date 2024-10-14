import librosa
import numpy as np
import csv
from collections import Counter

# Función para extraer MFCC de un archivo de audio
def extraer_mfcc(ruta_audio, n_mfcc=13):
    señal, tasa_muestreo = librosa.load(ruta_audio)
    mfcc = librosa.feature.mfcc(y=señal, sr=tasa_muestreo, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Función para leer los datos del archivo CSV
def leer_datos_desde_csv(archivo_csv):
    datos_entrenamiento = []
    clases_entrenamiento = []
    
    with open(archivo_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar la cabecera
        for row in reader:
            # Leer el vector MFCC (primeros 13 elementos) y la clase (último elemento)
            mfcc = np.array([float(valor) for valor in row[:-1]])  # Convertir los MFCC a float
            clase = row[-1]  # Última columna es la clase
            datos_entrenamiento.append(mfcc)
            clases_entrenamiento.append(clase)
    
    return datos_entrenamiento, clases_entrenamiento

# Función para calcular la distancia euclidiana
def distancia_euclidiana(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# Implementación del algoritmo Knn
def knn(clases_entrenamiento, datos_entrenamiento, dato_prueba, k=3):
    distancias = []
    
    # Calcular la distancia entre el dato de prueba y todos los datos de entrenamiento
    for i in range(len(datos_entrenamiento)):
        dist = distancia_euclidiana(datos_entrenamiento[i], dato_prueba)
        distancias.append((dist, clases_entrenamiento[i]))
    
    # Ordenar las distancias y obtener los k vecinos más cercanos
    distancias.sort(key=lambda x: x[0])
    vecinos_cercanos = [dist[1] for dist in distancias[:k]]
    
    # Votar la clase más común entre los k vecinos
    clase_predicha = Counter(vecinos_cercanos).most_common(1)[0][0]
    
    return clase_predicha

# Ejemplo de uso: Leer los datos desde el CSV y usar el Knn
archivo_csv = 'audio_datos.csv'  # Archivo CSV donde guardaste los datos MFCC
datos_entrenamiento, clases_entrenamiento = leer_datos_desde_csv(archivo_csv)

# Para probar con un nuevo archivo de audio existente (berenjena_1.wav)
ruta_audio_prueba = './audios1/berenjena_3.wav'  # Asegúrate de que esta ruta exista
dato_prueba = extraer_mfcc(ruta_audio_prueba)  # Extraemos el MFCC del nuevo archivo de audio

# Predecir la clase usando Knn
clase_predicha = knn(clases_entrenamiento, datos_entrenamiento, dato_prueba, k=3)
print("Clase predicha:", clase_predicha)
