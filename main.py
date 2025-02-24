import cv2
import pydicom
import numpy as np
import speech_recognition as sr
import os
import threading
import tkinter as tk
from tkinter import ttk
from openai import OpenAI
import pyttsx3


from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener la clave de API

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY")) 

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Ruta donde están almacenadas las imágenes DICOM
DICOM_PATH = "./dicom_images/"

# Variables globales
running = True  
current_index = 0
dicom_files = []
contrast_factor = 1.0  # Factor para el contraste
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

# Función para procesar la imagen
def process_image(image):
    global contrast_factor, zoom_factor
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    if zoom_factor != 1.0:
        h, w = image.shape
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        image = cv2.resize(image, (new_w, new_h))
    return image

# Función para mostrar imágenes DICOM
def show_dicom_image(index):
    if 0 <= index < len(dicom_files):
        dicom_data = pydicom.dcmread(dicom_files[index])
        image = dicom_data.pixel_array
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)
        processed_image = process_image(image)
        cv2.imshow("Imagen DICOM", processed_image)
        cv2.waitKey(1)

# Función para consultar a OpenAI
def ask_openai(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Eres un asistente de voz médico."},
                  {"role": "user", "content": question}]
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        return "Hubo un error al conectar con la IA."

# Función para manejar comandos de voz
def recognize_speech():
    global running, current_index, contrast_factor, zoom_factor
    recognizer = sr.Recognizer()
    
    while running:
        with sr.Microphone() as source:
            print("Di un comando...")
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio_data, language='es-ES').lower()
            print(f"Comando reconocido: {command}")

            if "siguiente" in command:
                current_index = (current_index + 1) % len(dicom_files)
            elif "anterior" in command:
                current_index = (current_index - 1) % len(dicom_files)
            elif "aumenta brillo" in command:
                contrast_factor += 0.2
            elif "disminuye brillo" in command:
                contrast_factor = max(0.2, contrast_factor - 0.2)
            elif "zoom más" in command:
                zoom_factor = min(2.5, zoom_factor + 0.2)
            elif "zoom menos" in command:
                zoom_factor = max(1.0, zoom_factor - 0.2)
            elif "salir" in command:
                print("Cerrando programa...")
                running = False
                root.quit()
                break  
            elif "ia" in command or "inteligencia artificial" in command:
                print("Dime tu pregunta para la IA...")
                speak("Dime tu pregunta para la inteligencia artificial")
                with sr.Microphone() as source:
                    audio_data = recognizer.listen(source)
                question = recognizer.recognize_google(audio_data, language='es-ES')
                print(f"Pregunta a la IA: {question}")
                response = ask_openai(question)
                print(f"Respuesta de la IA: {response}")
                speak(response)
            
            show_dicom_image(current_index)

        except sr.UnknownValueError:
            print("No se entendió el comando")
        except sr.RequestError:
            print("Error en la solicitud de reconocimiento")

# Interfaz gráfica con Tkinter
def create_ui():
    global root
    root = tk.Tk()
    root.title("Visor DICOM")
    btn_prev = ttk.Button(root, text="Anterior", command=lambda: change_image(-1))
    btn_prev.pack()
    btn_next = ttk.Button(root, text="Siguiente", command=lambda: change_image(1))
    btn_next.pack()
    btn_zoom_in = ttk.Button(root, text="Zoom +", command=lambda: adjust_zoom(0.2))
    btn_zoom_in.pack()
    btn_zoom_out = ttk.Button(root, text="Zoom -", command=lambda: adjust_zoom(-0.2))
    btn_zoom_out.pack()
    btn_brighter = ttk.Button(root, text="Aumentar Brillo", command=lambda: adjust_contrast(0.2))
    btn_brighter.pack()
    btn_darker = ttk.Button(root, text="Disminuir Brillo", command=lambda: adjust_contrast(-0.2))
    btn_darker.pack()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

# Funciones UI
def change_image(step):
    global current_index
    current_index = (current_index + step) % len(dicom_files)
    show_dicom_image(current_index)

def adjust_contrast(amount):
    global contrast_factor
    contrast_factor += amount
    show_dicom_image(current_index)

def adjust_zoom(amount):
    global zoom_factor
    zoom_factor = max(1.0, min(2.5, zoom_factor + amount))
    show_dicom_image(current_index)

# Función para manejar el cierre
def on_close():
    global running
    running = False
    root.quit()
    cv2.destroyAllWindows()

if load_dicom_images():
    show_dicom_image(current_index)
    thread_voice = threading.Thread(target=recognize_speech, daemon=True)
    thread_voice.start()
    create_ui()
    cv2.destroyAllWindows()
