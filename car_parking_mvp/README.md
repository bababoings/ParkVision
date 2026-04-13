# CETYS Parking Detection System - MVP

This project is a web application that detects and counts free and occupied parking spaces in a video feed. It uses a pre-trained Convolutional Neural Network (CNN) based on **MobileNetV2** (via Transfer Learning) to classify whether a parking space is occupied by a car or not.

The application is built using Flask, OpenCV, and TensorFlow/Keras. It represents an MVP adapted for CETYS Universidad.

### 📝 Cambios y Reemplazos Respecto al Proyecto Original

Este proyecto ha sido refactorizado significativamente respecto al repositorio original:

- **Se reemplazó el archivo principal:** `app.py` se ha convertido en `main.py` con una estructura orientada al MVP.
- **Se reemplazó el modelo base:** Se reemplazó el modelo original (`model_final.h5`) por uno nuevo basado en Transfer Learning con MobileNetV2 (`parking_mobilenetv2.h5`) para usar pesos de ImageNet y lograr un mejor F1-score.
- **Se reemplazó el Dataset de Entrenamiento:** Se descartaron las imágenes originales en favor del **Dataset PKLot** (en formato COCO), sumando decenas de miles de imágenes de entornos, climas y ángulos en 3D del mundo real.
- **Cambio Radical en Procesamiento (Adiós al método de Chando):** En el proyecto original de Chando, el sistema usaba `warpPerspective` para deformar el objeto a un cuadrado 2D (lo que derretía los autos lejanos). **Mi nuevo método** hace el proceso inverso: hace una *adaptación volumétrica* en la que deformamos nuestra caja matemática para que abarque al objeto real en la foto, recortando sin distorsión mantener el volumen 3D del automóvil.
- **Se modificó el manejo de regiones (ROIs):** Se modificó la forma estática de delimitar los lugares de estacionamiento. Ahora incluye una herramienta visual interactiva (`SpacePicker` vía web) que permite dibujar regiones con *Topes/Salidas* directamente en el navegador.
- **Se modificó la ingesta de video:** Ya no es obligatorio probar con el único video estático `car_test.mp4`. Ahora puedes **subir dinámicamente** tus propios videos vía una nueva ruta (interfaz web de carga y validación de frames).
- **Se enriqueció la API:** El endpoint `/space_count` no solo cuenta los disponibles. Ahora devuelve una respuesta JSON detallada con totales, disponibles, ocupados, una marca de tiempo y un **desglose por zonas**.

## Features

- **Real-time Video Processing:** Detects cars in real-time from a video stream.
- **SpacePicker UI:** Permite dibujar visualmente rectángulos en el primer frame del video para delimitar zonas sin tocar el código fuente.
- **Administrador de Zonas:** Clasificación de los cajones de estacionamiento por zonas definidas (Ej. "Zona A").
- **Dynamic Video Handling:** Subida de video directamente desde UI.
- **Enriched API:** Datos detallados en formato JSON usando `/space_count`.

## Technologies Used

- **Flask:** A micro web framework for Python.
- **OpenCV:** A library for real-time computer vision.
- **TensorFlow/Keras:** Deep learning framework para nuestro modelo `MobileNetV2`.
- **NumPy:** A library for numerical computations.
- **Pickle:** Used to load/save parking space positions.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Chando0185/car_parking_detection_space_count.git
cd parking-space-detection
```

### 2. Install Dependencies

Make sure you have Python installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Model and Data

- **Model:** Ensure the new trained model (`parking_mobilenetv2.h5`) is placed in the project directory.
- **Video Feed:** El video por defecto (`car_test.mp4`) se cargará si existe, pero puedes utilizar la página de `/upload` para procesar cualquier otro video.

### 4. Run the Application

Start the Flask application:

```bash
python main.py
```

The application will be available at `http://127.0.0.1:5000/`.

### 5. Access the Application

- **Homepage (Dashboard):** View the video stream at `http://127.0.0.1:5000/`.
- **Upload Video:** Accede a la carga dinámica de video en `http://127.0.0.1:5000/upload`.
- **SpacePicker:** Delimita tú mismo los cajones interactuando con el mouse en `http://127.0.0.1:5000/space_picker`.
- **Space Count API:** Get the current JSON details at `http://127.0.0.1:5000/space_count`.

## Project Structure

- `main.py`: The main Flask application file.
- `parking_mobilenetv2.h5`: El modelo actualizado por Transfer Learning.
- `carposition.pkl`: Metadata con las coordenadas (ahora con soporte para alto, ancho y zona).
- `templates/`: Formularios de carga, SpacePicker y vista principal.

---

Este es un proyecto modernizado, mantenible y escalable gracias a su nueva arquitectura orientada a un MVP.
