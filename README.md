# Visor DICOM con Asistente IA

Este proyecto es un visor de imágenes DICOM mejorado con capacidades de inteligencia artificial. Permite visualizar imágenes médicas en formato DICOM, ajustar su visualización y realizar análisis mediante IA utilizando la API de OpenAI.

## Características

- 📊 Visualización de imágenes DICOM
- 🔍 Ajuste de zoom y contraste
- 🔄 Navegación entre múltiples imágenes
- 🤖 Análisis de imágenes mediante IA (OpenAI)
- 🎙️ Control por voz
- 💬 Asistente de voz para responder preguntas médicas

## Requisitos

- Python 3.7+
- Bibliotecas de Python (ver sección de instalación)
- Clave API de OpenAI
- Archivos DICOM para visualizar

## Instalación

1. Clone este repositorio:

   ```bash
   git clone https://github.com/Daniel1309-gon/DICOM-voice-assistant
   cd visor-dicom
   ```

2. Instale las dependencias necesarias:

   ```bash
   pip install opencv-python pydicom numpy SpeechRecognition pillow openai pyttsx3 python-dotenv
   ```

3. Cree un archivo `.env` en la raíz del proyecto con su clave API de OpenAI:

   ```
   OPENAI_API_KEY=su_clave_api_aqui
   ```

4. Coloque sus imágenes DICOM en la carpeta `dicom_images/` en la raíz del proyecto.

## Uso

1. Ejecute la aplicación:

   ```bash
   python main.py
   ```

2. La interfaz se dividirá en dos secciones:

   - Panel izquierdo: Visualización de la imagen DICOM
   - Panel derecho: Controles y asistente IA

3. Utilice los botones para:

   - Navegar entre imágenes (Anterior/Siguiente)
   - Ajustar el zoom (Zoom +/-)
   - Modificar el brillo (Aumentar/Disminuir)
   - Hacer preguntas al asistente IA
   - Analizar la imagen actual con IA

4. Control por voz:
   - "siguiente" - Avanza a la siguiente imagen
   - "anterior" - Retrocede a la imagen anterior
   - "aumenta brillo" - Incrementa el brillo
   - "disminuye brillo" - Reduce el brillo
   - "zoom más" - Aumenta el zoom
   - "zoom menos" - Reduce el zoom
   - "analizar imagen" - Realiza análisis IA de la imagen actual
   - "ia" o "inteligencia artificial" - Activa el modo de preguntas a la IA

## Estructura del proyecto

- `main.py`: Archivo principal que contiene toda la lógica de la aplicación
- `dicom_images/`: Carpeta donde se deben colocar las imágenes DICOM
- `.env`: Archivo de configuración para la clave API de OpenAI

## Tecnologías utilizadas

- **OpenCV**: Procesamiento de imágenes
- **PyDICOM**: Lectura de archivos DICOM
- **Tkinter**: Interfaz gráfica
- **OpenAI API**: Análisis de imágenes y asistente de voz
- **SpeechRecognition**: Reconocimiento de voz
- **pyttsx3**: Síntesis de voz

## Notas importantes

- Asegúrese de tener una conexión a Internet activa para las funciones de IA y reconocimiento de voz.
- El análisis de imágenes consume tokens de la API de OpenAI, tenga en cuenta los costos asociados.
- Para un mejor reconocimiento de voz, utilice un micrófono de buena calidad en un entorno con poco ruido.


