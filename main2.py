import cv2
import pydicom
import numpy as np
import speech_recognition as sr
import os
import threading
from fft import apply_fft_contrast, apply_fft_rotation, apply_fft_zoom

# Ruta donde están almacenadas las imágenes DICOM
DICOM_PATH = "./dicom_images/"

# Variables globales
running = True  
current_index = 0
dicom_files = []
contrast_factor = 1.0  # Factor para el contraste
rotation_angle = 0  # Ángulo de rotación
zoom_factor = 1.0  # Factor de zoom

# Función para cargar imágenes DICOM
def load_dicom_images():
    global dicom_files

    if os.path.exists(DICOM_PATH):
        dicom_files = [os.path.join(DICOM_PATH, f) for f in os.listdir(DICOM_PATH) if f.endswith(".dcm")]
        if not dicom_files:
            print("No se encontraron archivos DICOM en la carpeta.")
            return False
        return True
    else:
        print(f"No se encontró la carpeta {DICOM_PATH}. Asegúrate de haber subido las imágenes.")
        return False

# Función para aplicar transformación a la imagen
def process_image(image):
    global contrast_factor, rotation_angle, zoom_factor

    # Aplicar transformaciones usando FFT
    image = apply_fft_contrast(image, contrast_factor)
    image = apply_fft_rotation(image, rotation_angle)
    image = apply_fft_zoom(image, zoom_factor)

    return image

# Función para mostrar una imagen a la vez
def show_dicom_image(index):
    if 0 <= index < len(dicom_files):
        dicom_data = pydicom.dcmread(dicom_files[index])
        image = dicom_data.pixel_array
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)

        # Aplicar transformaciones
        processed_image = process_image(image)

        # Mostrar la imagen en la misma ventana
        cv2.imshow("Imagen DICOM", processed_image)

# Función para manejar los comandos de voz
def recognize_speech():
    global running, current_index, contrast_factor, rotation_angle, zoom_factor

    recognizer = sr.Recognizer()
    
    while running:
        with sr.Microphone() as source:
            print("Di un comando ('siguiente', 'contraste más', 'contraste menos', 'rotar', 'zoom más', 'zoom menos', 'salir')...")
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio_data, language='es-ES').lower()
            print(f"Comando reconocido: {command}")

            commands_siguiente = ["siguiente", "próxima", "pasar imagen", "ver siguiente"]
            commands_anterior = ["anterior", "regresar", "volver atrás", "ver anterior"]
            commands_aum_contraste = ["aumenta contraste", "más contraste", "subir brillo"]
            commands_dis_contraste = ["disminuye contraste", "menos contraste", "bajar brillo"]
            commands_rotar_izq = ["rota a la izquierda", "gira a la izquierda"]
            commands_rotar_der = ["rota a la derecha", "gira a la derecha"]
            commands_zoom_mas = ["haz zoom", "zoom más", "acercar"]
            commands_zoom_menos = ["quita zoom", "zoom menos", "alejar"]

            if command in commands_siguiente:
                current_index = (current_index + 1) % len(dicom_files)
            elif command in commands_anterior:
                current_index = (current_index - 1) % len(dicom_files)
            elif command in commands_aum_contraste:
                contrast_factor += 0.2
            elif command in commands_dis_contraste:
                contrast_factor = max(0.2, contrast_factor - 0.2)
            elif command in commands_rotar_izq:
                rotation_angle = (rotation_angle + 90) % 360
            elif command in commands_rotar_der:
                rotation_angle = (rotation_angle - 90) % 360
            elif command in commands_zoom_mas:
                zoom_factor = min(2.5, zoom_factor + 0.2)
            elif command in commands_zoom_menos:
                zoom_factor = max(1.0, zoom_factor - 0.2)
            elif command == "salir":
                print("Cerrando programa...")
                running = False
                break  

            # Mostrar imagen actualizada con los cambios
            show_dicom_image(current_index)

        except sr.UnknownValueError:
            print("No se entendió el comando")
        except sr.RequestError:
            print("Error en la solicitud de reconocimiento")

# Cargar imágenes
if load_dicom_images():
    show_dicom_image(current_index)
    thread_voice = threading.Thread(target=recognize_speech)
    thread_voice.start()

    while running:
        if cv2.waitKey(100) & 0xFF == 27:
            running = False
            break

    cv2.destroyAllWindows()
