# Guía del Proyecto (Walkthrough) - CETYS Parking Detection MVP

Este documento está diseñado para ayudarte a ti (o a cualquier nuevo desarrollador) a entender cómo funciona internamente el proyecto, para qué sirve cada archivo y cómo se comunican las distintas partes del sistema para lograr la detección de estacionamiento en tiempo real.

## 1. Arquitectura General y Sinergia

El proyecto funciona bajo un modelo Cliente-Servidor utilizando **Flask** como motor principal. El flujo de trabajo típico de la aplicación es:

1. **Ingesta de Video:** El usuario carga un video mediante el portal web (`/upload`).
2. **Configuración de Zonas:** El usuario utiliza la herramienta visual (`/space_picker`) para dibujar los cajones de estacionamiento. Estas coordenadas se envían al servidor y se guardan en el disco.
3. **Procesamiento de Video:** El servidor extrae frames del video en tiempo real.
4. **Preprocesamiento:** Cada frame pasa por una capa de validación de calidad.
5. **Inferencia (IA):** Los recortes de la imagen correspondientes a los cajones de estacionamiento pasan por nuestro modelo **MobileNetV2**, el cual predice si están ocupados o disponibles.
6. **Consumo de Datos:** El front-end recibe el streaming de video modificado (con recuadros verdes y rojos) y además consume nuestra API (`/space_count`) para actualizar las estadísticas.

---

## 2. Guía Completa de Instalación y Puesta en Marcha

Cuando alguien clona este repositorio desde GitHub, **no obtendrá todos los archivos necesarios para ejecutar el proyecto**. Esto es intencional: ciertos archivos son demasiado pesados o son generados localmente en cada máquina, por lo que están excluidos mediante el `.gitignore`. A continuación se explica cada paso para reconstruirlos desde cero.

### Paso 1: Clonar el repositorio

```bash
git clone <URL-del-repositorio>
cd car_parking_mvp
```

Esto te dará únicamente los archivos de código fuente: `main.py`, `preprocessing.py`, `train_model.py`, la carpeta `templates/`, `static/`, `requirements.txt`, etc.

### Paso 2: Crear un Entorno Virtual (venv)

```bash
python -m venv venv
```

**¿Por qué?** Un entorno virtual es una copia aislada de Python que vive dentro de tu carpeta de proyecto. Sirve para instalar las librerías (TensorFlow, Flask, OpenCV, etc.) **sin contaminar** la instalación global de Python de tu computadora. Si no usas un `venv`, podrías generar conflictos de versiones con otros proyectos. Es por esto que la carpeta `venv/` está en el `.gitignore`: cada persona debe crear la suya propia en su máquina.

### Paso 3: Activar el Entorno Virtual

Cada vez que abras una terminal nueva para trabajar en el proyecto, necesitas activar el entorno virtual primero.

**En Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**En Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**En macOS / Linux:**
```bash
source venv/bin/activate
```

Sabrás que está activo porque verás `(venv)` al inicio de tu línea de comandos.

### Paso 4: Instalar las Dependencias

```bash
pip install -r requirements.txt
```

**¿Por qué?** El archivo `requirements.txt` es una lista con los nombres de las librerías que el proyecto necesita. Este comando le dice a `pip` (el instalador de paquetes de Python) que descargue e instale todas dentro de tu `venv`. Las librerías que se instalarán son:

| Librería             | Para qué se usa                                                        |
|----------------------|------------------------------------------------------------------------|
| `Flask`              | Servidor web que sirve las páginas HTML y la API                       |
| `tensorflow` / `keras` | Framework de Deep Learning que carga y ejecuta el modelo de IA       |
| `opencv-python`      | Lectura de video, recorte de frames, dibujo de rectángulos             |
| `opencv-contrib-python` | Módulos extra de OpenCV                                             |
| `numpy`              | Operaciones numéricas y manejo de arreglos (arrays) de imágenes        |
| `scikit-learn`       | Utilidades de Machine Learning (métricas, evaluación)                  |

> **Nota:** La instalación de TensorFlow puede tardar varios minutos y pesar varios GB. Esto es normal.

### Paso 5: Obtener el Modelo de IA (.h5)

Los archivos `.h5` están en el `.gitignore` porque son modelos binarios que pesan entre 10 MB y 110 MB, demasiado para un repositorio de Git convencional. Tienes dos opciones:

**Opción A – Entrenar tu propio modelo** (recomendado si tienes los datos):
1. Asegúrate de tener la carpeta `train_data/` con las imágenes de entrenamiento (subcarpetas `car` y `no_car`). Si no la tienes, descomprime `train_data.zip`.
2. Ejecuta:
   ```bash
   python train_model.py
   ```
3. Esto generará el archivo `parking_mobilenetv2.h5` en la raíz del proyecto.

**Opción B – Pedir el modelo al equipo:**
Solicita a un miembro del equipo que te comparta el archivo `parking_mobilenetv2.h5` y colócalo en la raíz del proyecto. Alternativamente, el sistema también soporta el modelo legacy `model_final.h5` como respaldo automático.

### Paso 6: Preparar las carpetas auxiliares

Estas carpetas también están en el `.gitignore` porque su contenido es temporal o generado por el usuario:

```bash
mkdir uploads
```

| Carpeta        | Para qué sirve                                                                 |
|----------------|--------------------------------------------------------------------------------|
| `uploads/`     | Aquí se guardan los videos que el usuario sube vía la interfaz `/upload`       |
| `train_data/`  | Imágenes de entrenamiento para el modelo (no necesaria para solo ejecutar)     |
| `__pycache__/` | Caché interna de Python, se genera sola automáticamente                        |

> **Nota:** `uploads/` y `__pycache__/` se crean automáticamente cuando ejecutas la aplicación, pero puedes crearlas manualmente si lo deseas.

### Paso 7: Ejecutar la Aplicación

```bash
python main.py
```

Si todo está correctamente configurado, verás en la terminal un mensaje similar a:

```
[INFO] Loaded X positions in Y zones
[INFO] Default video loaded: car_test.mp4
 * Running on http://127.0.0.1:5000
```

### Paso 8: Usar la Aplicación

Abre tu navegador y visita las siguientes rutas:

| Ruta                | Qué hace                                                                 |
|---------------------|--------------------------------------------------------------------------|
| `localhost:5000/`          | Dashboard principal con el video en vivo y las estadísticas       |
| `localhost:5000/upload`    | Subir un nuevo video para analizar                                |
| `localhost:5000/space_picker` | Dibujar los cajones de estacionamiento sobre el primer frame   |
| `localhost:5000/space_count`  | API JSON con el conteo actual (total, disponibles, ocupados, zonas) |

### Resumen: ¿Qué archivos NO están en GitHub y por qué?

| Archivo / Carpeta          | ¿Por qué está excluido?                                    | ¿Cómo obtenerlo?                     |
|----------------------------|-------------------------------------------------------------|---------------------------------------|
| `venv/` , `venv2/`        | Entorno virtual, se genera por máquina                      | `python -m venv venv`                 |
| `__pycache__/`             | Caché de Python, se genera automáticamente                  | Se crea solo al ejecutar              |
| `*.h5` (modelos)           | Archivos binarios pesados (10-110 MB)                       | Entrenar con `train_model.py` o pedir al equipo |
| `carposition.pkl`          | Coordenadas de cajones, se genera desde el SpacePicker      | Usar la interfaz `/space_picker`      |
| `train_data/`              | Imágenes de entrenamiento (cientos de archivos)             | Descomprimir `train_data.zip`         |
| `uploads/`                 | Videos subidos por el usuario                               | Se crea automáticamente               |
| Logs de instalación        | Archivos de depuración, no relevantes para el proyecto      | No necesarios                         |

---

## 3. Descripción de Archivos Clave

### El Motor Principal
- **`main.py`**: Es el cerebro de la aplicación.
  - Levanta el servidor web con Flask.
  - Contiene las rutas (End-points) a las páginas web (`/`, `/upload`, `/space_picker`).
  - Mantiene en memoria el archivo de video actual y el estado general de la aplicación.
  - Incluye la función `check_parking_spaces()` que es la encargada de hacer el "match" entre los cajones definidos y las predicciones del modelo.

### Inteligencia y Procesamiento Analítico
- **`preprocessing.py`**: Contiene la función `preprocess_frame`. Antes de que un frame trate de adivinar si hay carros o no, este script valida que el frame tenga buena resolución, brillo adecuado y no esté completamente borroso.
- **`parking_mobilenetv2.h5`**: El modelo de Red Neuronal profunda. Fue entrenado usando Transfer Learning, diseñado para ser rápido y eficiente tomando pedacitos pequeños de imagen (96x96 pixeles) y devolviendo una predicción.
- **`carposition.pkl`**: Un pequeño archivo binario. Aquí es donde se guardan temporalmente de forma física todos los rectángulos que dibujaste en el *Space Picker*. Incluye coordenadas `X`, `Y`, Ancho, Alto, y la `Zona` a la que pertenecen.

### La Interfaz Web (Carpeta `templates/`)
- **`index.html`**: El Dashboard principal. Es la pantalla donde visualizas el video corriendo en tiempo real y el resumen general.
- **`upload.html`**: Un simple formulario web que permite postear (`POST`) archivos de video de tu computadora al servidor (específicamente a la carpeta `uploads/`).
- **`space_picker.html`**: La interfaz donde ocurre la magia interactiva. Carga el primer frame del video activo y, usando JavaScript puro del lado del cliente, te permite arrastrar el mouse para crear los cuadritos y guardarlos.

---

## 4. ¿Cómo realizar cambios en el sistema?

Si alguien nuevo se une al proyecto, estos son los flujos según el tipo de cambio que se desee hacer:

**Si quieres cambiar el diseño, los colores o el Dashboard:**
Ve a la carpeta `templates/` y modifica los archivos HTML. Si quieres agregar gráficas, deberás consumirlas desde `/space_count`.

**Si el modelo predice mal y quieres uno mejor:**
Tendrás que entrenar un modelo nuevo externamente (por ejemplo en un Jupyter Notebook). Una vez obtenido un nuevo archivo `.h5`, reemplaza el que existe e instruye a `main.py` en la parte de `load_model_lazy()` para leer el nuevo archivo. Modifica la variable de `input_size` si tu nuevo modelo utiliza dimensiones diferentes.

**Si quieres que el SpacePicker tenga nuevas características (ej. cajones en diagonal):**
Tendrás que modificar el JavaScript dentro de `templates/space_picker.html` para soportar dibujo de polígonos, y luego tendrías que modificar a `main.py` para que en vez de guardar coordenadas simples (`x, y, w, h`), reciba y guarde los vértices del polígono en `carposition.pkl`.

---

## 5. Notas Técnicas y Restricciones
- El sistema utiliza **MJPEG (Motion JPEG)** para el streaming del video. Esto significa que manda imagen por imagen en la ruta `/video_feed`. Esto es algo intensivo para la red pero la mejor forma de integrarlo sin servidores complejos de video.
- Para evitar que la predicción "parpadee" (cambie rápidamente de verde a rojo entre frames), `main.py` contiene lógica de "Anti-Flickering" comparando contra un umbral de confianza.

---

## 6. Integración con Supabase (Base de Datos en Tiempo Real)

El proyecto incluye sincronización automática de la ocupación en tiempo real con una base de datos PostgreSQL alojada en **Supabase**. Esto permite alimentar páneles (dashboards), aplicaciones móviles o sistemas de terceros sin saturar el servidor de inferencia.

### Dependencias Necesarias
Para que la conexión funcione, debes instalar los paquetes requeridos usando el `python` explícito de tu entorno virtual. Ojo con este paso, ya que **si usas solo `pip install` podrías instalar los paquetes en una instalación global de Python por error**.

```powershell
.\venv2\Scripts\python.exe -m pip install supabase python-dotenv
```
**Problema común:** Si ves un error tipo `ModuleNotFoundError: No module named 'supabase'`, significa que instalaste las dependencias en otro entorno. Ejecutar el comando con la ruta explícita al `python.exe` local (como se muestra arriba) resuelve el problema.
> Nota: **Nunca** instales la librería `@supabase/supabase-js` con `npm` para el backend de este proyecto. Aquí usamos Python (`supabase-py`), no Node.js.

### Configuración del Entorno (.env)
La conexión requiere credenciales sensibles que **nunca deben subirse a GitHub**. En la raíz de tu proyecto, crea un archivo llamado `.env` (éste ya está excluido en el `.gitignore`) con el siguiente formato:

```env
SUPABASE_URL=https://<TU-PROYECTO>.supabase.co
SUPABASE_KEY=ey... (TU CLAVE)
```
- El "SUPABASE_URL" se encuentra en "Integrations -> Data API". Ahi mismo hay un apartado que dice "API URL".
- El "SUPABASE_KEY" se encuentra en "Settings -> API Keys". Ahi mismo hay un apartado que dice "Legacy anon, service_role API keys" y dentro de ese apartado hay un apartado que dice "anon" "public".
**Sobre la Clave (RLS Policies):**
El script de sincronización realiza operaciones combinadas de inserción y actualización (`upsert`).
- Si utilizas la clave pública (`anon`), **obligatoriamente** debes configurar las políticas de Row Level Security (RLS) en el panel de Supabase. Deberás permitir políticas `WITH CHECK ( true )` para Insert y Update.

### La Tabla `occupancy`
La sincronización asume que tienes una tabla llamada `occupancy` con el siguiente esquema:
- `zone_id` (PK, text): El nombre de la zona, por ejemplo `n1` o `pb`. **IMPORTANTE:** Para que la llave foránea no genere error, la zona debe estar declarada primeramente en tu tabla cruzada de `zones`.
- `available_spaces` (int4): Espacios vacíos de esa zona.
- `occupied_spaces` (int4): Cajones ocupados por autos.
- `confidence` (float8): El porcentaje de disponibilidad.
- `updated_at` (timestamptz): Fecha y hora del momento del guardado.

### Lógica del Nivel de Confianza (`confidence`)
En envíos hacia la base de datos, repensamos lo que significa "confianza". En vez de enviar qué tan seguro está el modelo de que un coche es un coche visualmente, rediseñamos la métrica como **"Porcentaje de Disponibilidad"**.
¿Qué significa esto para el usuario final?
- Si una zona de 10 cajones tiene 2 libres, la confianza es del **20%** (`0.20`), indicando que es poco probable que encuentre lugar al llegar.
- Si tiene 8 libres, la confianza sube al **80%** (`0.80`), indicando alta certidumbre de lograr estacionarse.
