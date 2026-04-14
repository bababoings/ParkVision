# Algoritmo de Proyección Vectorial Geométrica (Adaptación Volumétrica)

Este documento detalla el funcionamiento lógico y matemático detrás de la función `crop_with_volume()` implementada en el sistema, la cual reemplaza las técnicas heredadas para la extracción de regiones de interés (ROIs). 

## 1. El Problema Base: Perspectiva 3D en Cámaras 2D

Las cámaras de seguridad, especialmente cuando miran áreas lejanas como la **Zona A**, capturan el mundo con una fuerte perspectiva. Un cajón de estacionamiento cuadrado en la vida real se proyecta en la cámara como un **trapecio oblicuo**.

Cuando intentamos evaluar ese espacio, el desafío no es el piso, **es el objeto tridimensional (el automóvil)** que está encima de él. Debido a la perspectiva, el techo y la carrocería del auto no recaen físicamente dentro del polígono del suelo, sino que "sobresalen" visualmente hacia el horizonte (punto de fuga).

---

## 2. El Método Descartado: DLT (Transformación Lineal Directa) / Homografía

En el sistema original de Chando se utilizaba `cv2.warpPerspective()`. Este método utiliza el algoritmo **DLT (Direct Linear Transformation)** para calcular una **Matriz de Homografía**.

### ¿Cómo funciona el DLT?
Mapea puntos de un plano (el trapecio del piso) para forzarlos a ser otro plano (un cuadrado 2D perfecto de 96x96). Interpola matemáticamente los píxeles internos para aplanar la superficie.

### ¿Por qué falla con automóviles?
El DLT asume que todo lo que está dentro de los 4 puntos **es plano (2D)**. Si hay un automóvil (3D), el algoritmo intentará "aplastar" el techo para forzarlo contra el piso. 
*   **Ejemplo visual:** Imagina que imprimes una foto de un carro en una hoja de papel de hule cuadrado. Luego tomas las cuatro esquinas, las estiras y las tuerces de forma desigual hasta formar un trapecio. La foto quedará completamente irreconocible. Eso es lo que la inteligencia artificial recibía.

---

## 3. Nuestro Nuevo Algoritmo: Proyección Vectorial Geométrica

Para evitar la deformación, nuestro algoritmo realiza una **"extrusión virtual"** basada en la geometría nativa de la propia cámara, extrayendo los píxeles puros.

### Paso a Paso del Cálculo:

Supongamos que el usuario dibujó 4 puntos:
*   `P1, P2`: Puntos del "Tope" o fondo del cajón (más lejanos a la cámara).
*   `P3, P4`: Puntos de la "Salida" o línea de acceso (más cercanos a la cámara).

**Paso 1: Cálculo de Vectores de Fuga Laterales**
La perspectiva hace que las líneas laterales del cajón (`P4` hacia `P1`, y `P3` hacia `P2`) apunten naturalmente hacia el punto de fuga de la cámara en el horizonte. 
Calculamos la dirección (el vector matemático) de estas líneas laterales:
*   `Vector_Izquierdo = P1 - P4`
*   `Vector_Derecho = P2 - P3`

**Paso 2: Extrusión (Proyección Hacia Arriba)**
Sabiendo que el toldo y cuerpo físico del vehículo se proyectan visualmente siguiendo ese mismo punto de fuga, tomamos los puntos del fondo (`P1` y `P2`) y los "empujamos" a lo largo de sus vectores. 
Usamos un multiplicador o escalar (por ejemplo, el `EXTRUSION_FACTOR = 0.35`):
*   `P1_Extruido = P1 + (Vector_Izquierdo * 0.35)`
*   `P2_Extruido = P2 + (Vector_Derecho * 0.35)`

Ahora tenemos un polígono expandido: No envuelve solo el piso, sino el **volumen virtual** donde debería estar estacionado un carro de tamaño promedio.

**Paso 3: Bounding Box (Recorte Ortogonal Puro)**
Para evitar usar matrices que derritan la foto, simplemente buscamos la "Caja Contenedora" perfectamente rectangular más pequeña que logre encerrar nuestro nuevo polígono expandido:
*   Buscamos la `X mínima`, `Y mínima` entera.
*   Buscamos el ancho (`W`) y alto (`H`) que envuelva todos los puntos.
*   Finalmente aplicamos un `Crop` (recorte directo a la matriz `[ Y : Y+H, X : X+W ]`).

El resultado es un rectángulo intacto (píxeles 100% reales) que contiene el auto entero visto con el ángulo de inclinación nativo del lente de seguridad. Dado que nuestra IA fue ahora entrenada con PKLot para reconocer autos inclinados, la predicción es casi perfecta.

---

## 4. Comparativa con Otras Referencias del Estado del Arte

Para dar perspectiva técnica de en qué lugar se encuentra esta solución, podemos compararla con otros algoritmos del sector:

| Tecnología / Algoritmo | Descripción | Pro / Contra frente a nuestro sistema |
| :--- | :--- | :--- |
| **Homografía (DLT)** | *(El descartado)* Mapeo matricial plano a plano. | **Contra:** Falla catastróficamente con objetos 3D volumétricos. Es útil solo para escanear documentos de texto ladeados. |
| **Nuestra Proyección Vectorial (Bounding Rect)** | Aproximación volumétrica estirando el polígono hacia el punto de fuga, aislando sus píxeles brutos en un recuadro. | **Pro:** Ultra-ligero (solo álgebra de vectores sencilla), no distorsiona el objeto y retiene texturas reales. Solución óptima para un MVP basado en clasificación 2D. |
| **3D Bounding Box Regression (Ej. MediaPipe Objectron)** | Modelos IA pesados que en lugar de arrojar cajas 2D, infieren las dimensiones exactas (Largo, ancho, alto) dibujando un prisma 3D sobre los autos mediante *Pose Estimation*. | **Contra:** Computacionalmente demandante y requiere reescribir toda la arquitectura actual a redes YOLO-3D complejas. Innecesario si solo queremos saber si está libre u ocupado. |
| **PnP (Perspective-n-Point)** | Algoritmo que puede estimar la "pose" de la cámara basándose en objetos conocidos en la escena para deducir un 3D rústico. | **Contra:** Requiere conocer puntos de calibración intrínseca de cada cámara física en el CETYS. El sistema actual Vectorial asume dinámicamente dichos puntos sin calibrar. |
| **Nubes de Puntos (LiDAR)** | Escáner físico de profundidad tridimensional con láser. | **Contra:** Requiere instalar hardware costoso. Nuestra aproximación simula este "volumen" usando pura inteligencia artificial y geometría sobre cámaras instaladas tradicionales. |

## Conclusión

El cambio de un **algoritmo matemático distorsivo (DLT)** a un **algoritmo de envolvente geométrica (Vectorial + Bounding Box)** le otorga a este desarrollo la capacidad de interpretar el mundo en su dimensión volumétrica sin tener que sacrificar los limitados recursos de procesamiento computacional. Funciona exactamente brindando equilibrio entre viabilidad algorítmica y excelente rendimiento predictivo.
