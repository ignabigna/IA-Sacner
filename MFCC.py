import librosa
import numpy as np
import os
import csv

# Función para extraer MFCC de un archivo de audio
def extraer_mfcc(ruta_audio, n_mfcc=13):
    señal, tasa_muestreo = librosa.load(ruta_audio)
    mfcc = librosa.feature.mfcc(y=señal, sr=tasa_muestreo, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Función para cargar los audios, extraer MFCC y guardarlos en un archivo CSV
def guardar_mfcc_en_csv(directorio, archivo_salida):
    with open(archivo_salida, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribir la cabecera del archivo CSV
        writer.writerow(["MFCC_" + str(i) for i in range(13)] + ["Clase"])

        for archivo in os.listdir(directorio):
            if archivo.endswith(".wav"):
                ruta_archivo = os.path.join(directorio, archivo)
                mfcc = extraer_mfcc(ruta_archivo)  # Extraer características MFCC

                # Asignar la clase según el nombre del archivo
                if "papa" in archivo:
                    clase = "papa"
                elif "zanahoria" in archivo:
                    clase = "zanahoria"
                elif "choclo" in archivo:
                    clase = "choclo"
                elif "berenjena" in archivo:
                    clase = "berenjena"
                
                # Escribir el vector MFCC y la clase en el archivo CSV
                writer.writerow(list(mfcc) + [clase])

# Ejemplo de uso
directorio_audios = './audios'  # Directorio donde están tus archivos de audio
archivo_salida = 'audio_datos.csv'  # Nombre del archivo CSV donde guardaremos los datos
guardar_mfcc_en_csv(directorio_audios, archivo_salida)
print("Datos MFCC guardados en", archivo_salida)
