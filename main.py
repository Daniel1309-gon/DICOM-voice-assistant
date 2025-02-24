import cv2
import pydicom
import numpy as np
import speech_recognition as sr
import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from openai import OpenAI
import pyttsx3
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración inicial
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
engine = pyttsx3.init()
DICOM_PATH = "./dicom_images/"  

def speak(text):
    engine.say(text)
    engine.runAndWait()

class DicomViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visor DICOM Mejorado")
        
        # Variables globales
        self.running = True
        self.current_index = 0
        self.dicom_files = []
        self.contrast_factor = 1.0
        self.zoom_factor = 1.0
        
        # Configurar el diseño principal
        self.setup_layout()
        
        # Cargar imágenes
        self.load_dicom_images()
        
        # Iniciar reconocimiento de voz
        self.start_voice_recognition()

    def setup_layout(self):
        # Contenedor principal
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel izquierdo para la imagen
        self.image_frame = ttk.Frame(main_container)
        main_container.add(self.image_frame, weight=2)

        # Canvas para la imagen
        self.canvas = tk.Canvas(self.image_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Panel derecho para controles
        controls_frame = ttk.Frame(main_container)
        main_container.add(controls_frame, weight=1)

        # Estilo para los botones
        style = ttk.Style()
        style.configure('Action.TButton', padding=5)

        # Frame para navegación
        nav_frame = ttk.LabelFrame(controls_frame, text="Navegación", padding=10)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="← Anterior", command=lambda: self.change_image(-1), style='Action.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Siguiente →", command=lambda: self.change_image(1), style='Action.TButton').pack(side=tk.RIGHT, padx=5)

        # Frame para zoom
        zoom_frame = ttk.LabelFrame(controls_frame, text="Zoom", padding=10)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(zoom_frame, text="Zoom +", command=lambda: self.adjust_zoom(0.2)).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Zoom -", command=lambda: self.adjust_zoom(-0.2)).pack(side=tk.RIGHT, padx=5)

        # Frame para brillo
        brightness_frame = ttk.LabelFrame(controls_frame, text="Brillo", padding=10)
        brightness_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(brightness_frame, text="Aumentar", command=lambda: self.adjust_contrast(0.2)).pack(side=tk.LEFT, padx=5)
        ttk.Button(brightness_frame, text="Disminuir", command=lambda: self.adjust_contrast(-0.2)).pack(side=tk.RIGHT, padx=5)

        # Frame para IA
        ai_frame = ttk.LabelFrame(controls_frame, text="Asistente IA", padding=10)
        ai_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.ai_text = tk.Text(ai_frame, height=5, wrap=tk.WORD)
        self.ai_text.pack(fill=tk.X, pady=5)
        
        ttk.Button(ai_frame, text="Hacer Pregunta", command=self.ask_ai_question).pack(fill=tk.X, pady=5)

        # Estado
        self.status_label = ttk.Label(controls_frame, text="Listo")
        self.status_label.pack(side=tk.BOTTOM, pady=5)

    def load_dicom_images(self):
        if os.path.exists(DICOM_PATH):
            self.dicom_files = [os.path.join(DICOM_PATH, f) for f in os.listdir(DICOM_PATH) if f.endswith(".dcm")]
            if self.dicom_files:
                self.show_dicom_image(self.current_index)
            else:
                self.status_label.config(text="No se encontraron archivos DICOM")
        else:
            self.status_label.config(text="Carpeta DICOM no encontrada")

    def process_image(self, image):
        image = cv2.convertScaleAbs(image, alpha=self.contrast_factor, beta=0)
        if self.zoom_factor != 1.0:
            h, w = image.shape
            new_w, new_h = int(w * self.zoom_factor), int(h * self.zoom_factor)
            image = cv2.resize(image, (new_w, new_h))
        return image

    def show_dicom_image(self, index):
        if 0 <= index < len(self.dicom_files):
            dicom_data = pydicom.dcmread(self.dicom_files[index])
            image = dicom_data.pixel_array
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = np.uint8(image)
            processed_image = self.process_image(image)
            
            # Convertir para Tkinter
            image = Image.fromarray(processed_image)
            photo = ImageTk.PhotoImage(image=image)
            
            # Actualizar canvas
            self.canvas.config(width=photo.width(), height=photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Mantener referencia

    def change_image(self, step):
        self.current_index = (self.current_index + step) % len(self.dicom_files)
        self.show_dicom_image(self.current_index)

    def adjust_contrast(self, amount):
        self.contrast_factor = max(0.2, min(3.0, self.contrast_factor + amount))
        self.show_dicom_image(self.current_index)

    def adjust_zoom(self, amount):
        self.zoom_factor = max(1.0, min(2.5, self.zoom_factor + amount))
        self.show_dicom_image(self.current_index)

    def speak(self, text):
        engine.say(text)
        engine.runAndWait()

    def ask_ai_question(self):
        question = self.ai_text.get("1.0", tk.END).strip()
        if question:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Eres un asistente de voz médico, quiero que me des una respuesta clara y concisa."},
                        {"role": "user", "content": question}
                    ]
                )
                answer = response.choices[0].message.content
                self.speak(answer)
                self.status_label.config(text="Respuesta de IA completada")
            except Exception as e:
                self.status_label.config(text="Error al conectar con la IA")
        else:
            self.status_label.config(text="Por favor, ingrese una pregunta")

    def ask_openai(self, question):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente de voz médico, quiero que me des una respuesta clara y concisa."},
                    {"role": "user", "content": question}
                ]
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            return "Hubo un error al conectar con la IA."


    def start_voice_recognition(self):
        self.voice_thread = threading.Thread(target=self.recognize_speech, daemon=True)
        self.voice_thread.start()

    def recognize_speech(self):
        recognizer = sr.Recognizer()
        while self.running:
            with sr.Microphone() as source:
                self.status_label.config(text="Escuchando...")
                recognizer.adjust_for_ambient_noise(source)
                try:
                    audio_data = recognizer.listen(source)
                    command = recognizer.recognize_google(audio_data, language='es-ES').lower()
                    
                    if "siguiente" in command:
                        self.change_image(1)
                    elif "anterior" in command:
                        self.change_image(-1)
                    elif "aumenta brillo" in command:
                        self.adjust_contrast(0.2)
                    elif "disminuye brillo" in command:
                        self.adjust_contrast(-0.2)
                    elif "zoom más" in command:
                        self.adjust_zoom(0.2)
                    elif "zoom menos" in command:
                        self.adjust_zoom(-0.2)
                    elif "ia" in command or "inteligencia artificial" in command:
                        print("Dime tu pregunta para la IA...")
                        speak("Dime tu pregunta para la inteligencia artificial")
                        with sr.Microphone() as source:
                            audio_data = recognizer.listen(source)
                        question = recognizer.recognize_google(audio_data, language='es-ES')
                        print(f"Pregunta a la IA: {question}")
                        response = self.ask_openai(question)
                        print(f"Respuesta de la IA: {response}")
                        speak(response)
                    
                    self.status_label.config(text=f"Comando reconocido: {command}")
                except sr.UnknownValueError:
                    self.status_label.config(text="No se entendió el comando")
                except sr.RequestError:
                    self.status_label.config(text="Error en reconocimiento de voz")

    def on_closing(self):
        self.running = False
        self.root.quit()

def main():
    root = tk.Tk()
    app = DicomViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()