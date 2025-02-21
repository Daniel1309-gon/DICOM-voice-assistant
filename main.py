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

    # Aplicar contraste
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

    # Rotar la imagen
    if rotation_angle != 0:
        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    # Aplicar zoom
    if zoom_factor != 1.0:
        h, w = image.shape
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        image = cv2.resize(image, (new_w, new_h))

        # Recortar el centro de la imagen si el zoom es mayor a 1
        if zoom_factor > 1.0:
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2
            image = image[y_start:y_start+h, x_start:x_start+w]

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

            commands_siguiente = [
                "siguiente", "siguiente imagen", "próxima", "próxima imagen",
                "siguiente foto", "próxima foto", "siguiente radiografía", "próxima radiografía",
                "pasar imagen", "pasar a la siguiente", "ver siguiente"
            ]

            commands_anterior = [
                "anterior", "anterior imagen", "anterior foto", "anterior radiografía",
                "regresar", "regresar imagen", "regresar foto", "regresar radiografía",
                "volver atrás", "volver a la anterior", "imagen anterior", "foto anterior",
                "radiografía anterior", "pasar a la anterior", "ver anterior"
            ]

            commands_aum_contraste = [
                "aumenta contraste", "aumentar contraste", "contraste más", "más contraste",
                "más brillo", "brillo más", "aumenta brillo", "aumentar brillo",
                "aumentar el contraste", "aumenta el contraste", "subir contraste", "subir brillo"
            ]

            commands_dis_contraste = [
                "disminuye contraste", "disminuir contraste", "contraste menos", "menos contraste",
                "menos brillo", "brillo menos", "disminuye brillo", "disminuir brillo",
                "disminuir el contraste", "disminuye el contraste", "bajar contraste", "bajar brillo"
            ]

            commands_rotar_izq = [
                "rota a la izquierda", "gira a la izquierda", "gira la imagen a la izquierda",
                "girar imagen a la izquierda", "rotar a la izquierda", "girar a la izquierda",
                "voltear izquierda", "girar en sentido antihorario", "girar en contra de las agujas del reloj"
            ]

            commands_rotar_der = [
                "rota a la derecha", "gira a la derecha", "gira la imagen a la derecha",
                "girar imagen a la derecha", "rotar a la derecha", "girar a la derecha",
                "voltear derecha", "girar en sentido horario", "girar en el sentido de las agujas del reloj"
            ]

            commands_zoom_mas = [
                "haz zoom", "aumenta zoom", "zoom más", "zoom in", "acercar", "acercar imagen",
                "acercar zoom", "zoom adelante", "agrandar", "hacer más grande", "más grande"
            ]

            commands_zoom_menos = [
                "quita zoom", "disminuye zoom", "zoom menos", "zoom out", "alejar", "alejar imagen",
                "alejar zoom", "zoom atrás", "hacer más pequeño", "reducir imagen", "más pequeño"
            ]


            if command in commands_siguiente:
                current_index += 1
                if current_index >= len(dicom_files):  
                    current_index = 0  # Reiniciar al inicio si se llega al final
            
            elif command in commands_anterior:
                current_index -= 1
                if current_index < 0:
                    current_index = len(dicom_files) - 1

            elif command in commands_aum_contraste:
                contrast_factor += 0.2  # Aumentar contraste

            elif command in commands_dis_contraste:
                contrast_factor = max(0.2, contrast_factor - 0.2)  # Disminuir contraste

            elif command in commands_rotar_izq:
                rotation_angle = (rotation_angle + 90) % 360  # Rotar 90° cada vez
            
            elif command in commands_rotar_der:
                rotation_angle = (rotation_angle - 90) % 360  # Rotar 90° cada vez

            elif command in commands_zoom_mas:
                zoom_factor = min(2.5, zoom_factor + 0.2)  # Aumentar zoom (máx 2.5x)

            elif command in commands_zoom_menos:
                zoom_factor = max(1.0, zoom_factor - 0.2)  # Disminuir zoom (mín 1x)

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
    # Mostrar la primera imagen
    show_dicom_image(current_index)

    # Crear el hilo para el reconocimiento de voz
    thread_voice = threading.Thread(target=recognize_speech)
    thread_voice.start()

    # Mantener la ventana abierta hasta que se diga "salir"
    while running:
        if cv2.waitKey(100) & 0xFF == 27:  # Presionar ESC para salir manualmente
            running = False
            break

    # Cerrar todo al finalizar
    cv2.destroyAllWindows()
