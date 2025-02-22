import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pydicom
import numpy as np
import speech_recognition as sr
import os
import threading
from fft import apply_fft_contrast, apply_fft_rotation, apply_fft_zoom  # Se mantienen, aunque en este ejemplo no se usan

# Ruta donde están almacenadas las imágenes DICOM
DICOM_PATH = "./dicom_images/"

# Variables globales
running = True  
current_index = 0
dicom_files = []
contrast_factor = 1.0  # Factor para el contraste
rotation_angle = 0     # Ángulo de rotación
zoom_factor = 1.0      # Factor de zoom

# Cargar archivos DICOM
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

# Función para aplicar las transformaciones a la imagen
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
        # Si se aumentó el zoom, se recorta el centro
        if zoom_factor > 1.0:
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2
            image = image[y_start:y_start+h, x_start:x_start+w]

    return image

# Función para actualizar la imagen mostrada en la interfaz
def update_image():
    global current_index, dicom_files
    if 0 <= current_index < len(dicom_files):
        dicom_data = pydicom.dcmread(dicom_files[current_index])
        image = dicom_data.pixel_array
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)

        # Aplicar transformaciones
        processed_image = process_image(image)

        # Convertir la imagen a formato PIL y luego a PhotoImage para tkinter
        pil_image = Image.fromarray(processed_image).convert("L")
        tk_image = ImageTk.PhotoImage(pil_image)

        # Actualizar el label de la imagen
        image_label.config(image=tk_image)
        image_label.image = tk_image

        # Actualizar la barra de estado con los parámetros actuales
        status_text.set(f"Imagen: {current_index + 1}/{len(dicom_files)} | Contraste: {contrast_factor:.1f} | Rotación: {rotation_angle}° | Zoom: {zoom_factor:.1f}")

# Función para manejar el reconocimiento de voz en un hilo separado
def recognize_speech():
    global running, current_index, contrast_factor, rotation_angle, zoom_factor

    recognizer = sr.Recognizer()
    # Se definen los comandos de voz
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

    while running:
        with sr.Microphone() as source:
            print("Di un comando ('siguiente', 'contraste más', 'contraste menos', 'rotar', 'zoom más', 'zoom menos', 'salir')...")
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio_data, language='es-ES').lower()
            print(f"Comando reconocido: {command}")

            if command in commands_siguiente:
                current_index += 1
                if current_index >= len(dicom_files):
                    current_index = 0  # Volver al inicio
            elif command in commands_anterior:
                current_index -= 1
                if current_index < 0:
                    current_index = len(dicom_files) - 1
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
                root.quit()
                break

            # Programar la actualización de la imagen en el hilo principal
            root.after(0, update_image)

        except sr.UnknownValueError:
            print("No se entendió el comando")
        except sr.RequestError:
            print("Error en la solicitud de reconocimiento")

# Inicialización de la interfaz tkinter
root = tk.Tk()
root.title("Visor de Imágenes DICOM")
root.geometry("800x600")
root.configure(bg="#f0f4f7")  # Fondo claro y profesional

# Título de la aplicación
title_label = ttk.Label(root, text="Visor de Imágenes DICOM para Médicos", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

# Área para mostrar la imagen
image_label = ttk.Label(root)
image_label.pack(expand=True)

# Etiqueta de estado para mostrar información actual
status_text = tk.StringVar()
status_label = ttk.Label(root, textvariable=status_text, font=("Helvetica", 12))
status_label.pack(pady=5)

# Instrucciones de voz
instructions = (
    "Comandos de voz disponibles:\n"
    "• 'siguiente' / 'anterior' para navegar entre imágenes\n"
    "• 'contraste más' / 'contraste menos' para ajustar el contraste\n"
    "• 'rota a la izquierda' / 'rota a la derecha' para rotar\n"
    "• 'zoom más' / 'zoom menos' para modificar el zoom\n"
    "• 'salir' para cerrar el programa"
)
instr_label = ttk.Label(root, text=instructions, font=("Helvetica", 10), foreground="#555")
instr_label.pack(pady=5)

# Cargar las imágenes DICOM
if load_dicom_images():
    update_image()
else:
    status_text.set("Error al cargar imágenes DICOM.")

# Iniciar el hilo de reconocimiento de voz
thread_voice = threading.Thread(target=recognize_speech, daemon=True)
thread_voice.start()

# Ejecutar el loop principal de tkinter
root.mainloop()

# Al salir se cierran todos los recursos
print("Programa finalizado.")
