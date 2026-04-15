# 02 — Generación de imágenes con IA

> **Bloque:** Multimodalidad · **Nivel:** Práctico · **Tiempo estimado:** 60 min

---

## Índice

1. [Panorama de modelos de generación](#1-panorama-de-modelos-de-generación)
2. [DALL-E 3 con la API de OpenAI](#2-dall-e-3-con-la-api-de-openai)
3. [Prompt engineering para imágenes](#3-prompt-engineering-para-imágenes)
4. [Edición de imágenes con DALL-E (inpainting)](#4-edición-de-imágenes-con-dall-e-inpainting)
5. [Stable Diffusion local con diffusers](#5-stable-diffusion-local-con-diffusers)
6. [Pipeline LLM + generación de imágenes](#6-pipeline-llm--generación-de-imágenes)
7. [Consideraciones éticas y de uso](#7-consideraciones-éticas-y-de-uso)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Panorama de modelos de generación

La generación de imágenes a partir de texto ha evolucionado rápidamente. Hoy existen varias opciones según tus necesidades:

| Modelo | Empresa | Acceso | Calidad | Velocidad | Privacidad |
|---|---|---|---|---|---|
| DALL-E 3 | OpenAI | API | Alta | Rápida | Datos enviados a OpenAI |
| DALL-E 2 | OpenAI | API | Media | Muy rápida | Datos enviados a OpenAI |
| Stable Diffusion | Stability AI | Local / API | Alta (configurable) | Media (GPU) | Total (local) |
| Flux | Black Forest Labs | Local / API | Muy alta | Media | Total (local) |
| Midjourney | Midjourney | Discord / API | Muy alta | Media | Datos enviados a Midjourney |
| Imagen 3 | Google | API (Vertex AI) | Alta | Rápida | Datos en Google Cloud |

### ¿Qué modelo elegir?

- **Prototipado rápido o producción en la nube**: DALL-E 3 o Imagen 3.
- **Privacidad total o sin costes de API**: Stable Diffusion o Flux localmente.
- **Máxima calidad artística**: Midjourney (aunque tiene API limitada) o Flux.
- **Integración en pipelines Python**: DALL-E 3 o Stable Diffusion con `diffusers`.

### Cómo funcionan (conceptualmente)

Los modelos modernos de generación de imágenes usan **difusión latente** (Latent Diffusion Models, LDM). El proceso es:

1. El prompt de texto se convierte en embeddings con un encoder (ej. CLIP).
2. Se genera ruido aleatorio en el espacio latente.
3. Una red neuronal aprende a eliminar el ruido paso a paso, guiada por los embeddings del texto.
4. El resultado latente se decodifica a una imagen en píxeles.

---

## 2. DALL-E 3 con la API de OpenAI

DALL-E 3 es el modelo de generación de imágenes de OpenAI. Genera imágenes de alta calidad a partir de texto y admite tres tamaños: `1024x1024`, `1792x1024` y `1024x1792`.

### Generar una imagen desde texto

```python
import openai
from pathlib import Path
import requests
import base64


def generar_imagen_dalle(
    prompt: str,
    nombre_archivo: str = "imagen_generada.png",
    tamano: str = "1024x1024",
    calidad: str = "standard",
    estilo: str = "vivid",
) -> str:
    """
    Genera una imagen con DALL-E 3 y la guarda en disco.

    Args:
        prompt: Descripción textual de la imagen a generar.
        nombre_archivo: Nombre del archivo de salida (PNG).
        tamano: "1024x1024", "1792x1024" o "1024x1792".
        calidad: "standard" (más rápido) o "hd" (más detallado).
        estilo: "vivid" (colores vivos, dramático) o "natural" (más realista).

    Returns:
        Ruta al archivo guardado.
    """
    cliente = openai.OpenAI()  # Usa OPENAI_API_KEY del entorno

    respuesta = cliente.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=tamano,
        quality=calidad,
        style=estilo,
        n=1,  # DALL-E 3 solo permite n=1
        response_format="url",  # o "b64_json" para recibir base64
    )

    # La API devuelve la URL de la imagen generada
    url_imagen = respuesta.data[0].url
    prompt_revisado = respuesta.data[0].revised_prompt  # DALL-E 3 puede modificar el prompt

    print(f"Prompt original:  {prompt}")
    print(f"Prompt revisado:  {prompt_revisado}")
    print(f"URL de la imagen: {url_imagen}")

    # Descargar y guardar la imagen
    contenido = requests.get(url_imagen).content
    ruta = Path(nombre_archivo)
    ruta.write_bytes(contenido)

    print(f"Imagen guardada en: {ruta.resolve()}")
    return str(ruta.resolve())


# --- Uso básico ---
if __name__ == "__main__":
    ruta = generar_imagen_dalle(
        prompt="Un gato astronauta flotando en el espacio, con la Tierra al fondo, estilo fotografía realista",
        nombre_archivo="gato_astronauta.png",
        calidad="hd",
    )
```

### Recibir la imagen en base64 (sin descargar desde URL)

```python
import openai
import base64
from pathlib import Path


def generar_imagen_base64(prompt: str, nombre_archivo: str = "imagen.png") -> str:
    """
    Genera una imagen con DALL-E 3 y la recibe directamente en base64.
    Útil cuando la URL temporal puede expirar o en entornos sin acceso a Internet.
    """
    cliente = openai.OpenAI()

    respuesta = cliente.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        response_format="b64_json",  # Recibir en base64 en lugar de URL
    )

    datos_b64 = respuesta.data[0].b64_json
    bytes_imagen = base64.b64decode(datos_b64)

    ruta = Path(nombre_archivo)
    ruta.write_bytes(bytes_imagen)
    print(f"Imagen guardada: {ruta.resolve()}")
    return str(ruta.resolve())


# --- Uso ---
if __name__ == "__main__":
    generar_imagen_base64(
        prompt="Mapa del mundo ilustrado estilo acuarela, colores suaves",
        nombre_archivo="mapa_acuarela.png",
    )
```

---

## 3. Prompt engineering para imágenes

La calidad de la imagen generada depende enormemente de cómo se escribe el prompt. A diferencia de los prompts de texto, los prompts de imagen se benefician de ser descriptivos, específicos y con referencias estilísticas.

### Estructura de un buen prompt

```
[Sujeto principal] + [Acción/estado] + [Escenario/contexto] + [Estilo artístico] + [Iluminación] + [Calidad]
```

### Ejemplos comparativos

```python
import openai
import requests
from pathlib import Path


prompts_comparativos = [
    # Prompt básico (resultado mediocre)
    {
        "nombre": "basico",
        "prompt": "un perro en un parque",
    },
    # Prompt intermedio (mejor)
    {
        "nombre": "intermedio",
        "prompt": (
            "Un golden retriever jugando en un parque verde en otoño, "
            "hojas doradas en el suelo, luz de tarde"
        ),
    },
    # Prompt avanzado (resultado óptimo)
    {
        "nombre": "avanzado",
        "prompt": (
            "Un golden retriever corriendo entre hojas otoñales doradas y rojizas en un parque londinense. "
            "Luz de tarde cálida y dorada (hora mágica), bokeh suave en el fondo con árboles desenfocados. "
            "Fotografía profesional de mascotas, lente 85mm f/1.8, alta resolución, detalle en el pelaje."
        ),
    },
    # Prompt estilo ilustración
    {
        "nombre": "ilustracion",
        "prompt": (
            "Un golden retriever en un parque de otoño, estilo ilustración de libro infantil, "
            "acuarela suave, paleta de colores cálidos, trazo delicado, fondo blanco"
        ),
    },
]


def comparar_prompts(prompts: list, carpeta: str = "comparacion_prompts") -> None:
    """Genera una imagen por cada prompt y las guarda para comparar."""
    Path(carpeta).mkdir(exist_ok=True)
    cliente = openai.OpenAI()

    for config in prompts:
        print(f"Generando: {config['nombre']}...")
        respuesta = cliente.images.generate(
            model="dall-e-3",
            prompt=config["prompt"],
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = respuesta.data[0].url
        contenido = requests.get(url).content
        ruta = Path(carpeta) / f"{config['nombre']}.png"
        ruta.write_bytes(contenido)
        print(f"  Guardado: {ruta}")


# Técnicas de modificación de estilo
MODIFICADORES_ESTILO = {
    "fotografico": "professional photography, DSLR, 50mm lens, natural lighting",
    "acuarela": "watercolor painting, soft colors, paper texture, artistic",
    "ilustracion": "digital illustration, flat design, clean lines, vibrant colors",
    "oil_painting": "oil painting on canvas, impasto technique, rich colors, museum quality",
    "anime": "anime style, Studio Ghibli inspired, cel shading, vibrant",
    "cinefilm": "cinematic, 35mm film grain, anamorphic lens flare, Hollywood style",
    "minimalista": "minimalist design, white background, simple shapes, clean aesthetic",
}

if __name__ == "__main__":
    comparar_prompts(prompts_comparativos)
```

### Modificadores de calidad más efectivos

| Categoría | Modificadores |
|---|---|
| Calidad técnica | `highly detailed`, `sharp focus`, `8K resolution`, `professional` |
| Iluminación | `golden hour`, `studio lighting`, `dramatic shadows`, `soft diffused light` |
| Estilo artístico | `oil painting`, `watercolor`, `digital art`, `photorealistic` |
| Cámara/óptica | `85mm portrait lens`, `wide angle`, `macro photography`, `bokeh` |
| Atmósfera | `cinematic`, `moody`, `ethereal`, `vibrant` |

---

## 4. Edición de imágenes con DALL-E (inpainting)

El inpainting permite editar zonas específicas de una imagen existente. Se necesita la imagen original y una máscara en blanco y negro donde el blanco indica la zona a regenerar.

```python
import openai
import requests
from pathlib import Path
from PIL import Image
import numpy as np
import io


def crear_mascara_circular(
    ancho: int, alto: int, centro_x: int, centro_y: int, radio: int
) -> bytes:
    """
    Crea una máscara PNG con un círculo blanco (zona a editar)
    sobre fondo negro (zona a conservar).
    """
    # Fondo negro (transparente en formato RGBA)
    mascara = np.zeros((alto, ancho, 4), dtype=np.uint8)

    # Crear círculo blanco/transparente en la zona a editar
    y, x = np.ogrid[:alto, :ancho]
    distancia = np.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)
    zona_editar = distancia <= radio

    # En DALL-E: píxeles TRANSPARENTES = zona a editar
    mascara[zona_editar, 3] = 0    # Alpha = 0 (transparente) → editar aquí
    mascara[~zona_editar, 3] = 255  # Alpha = 255 (opaco) → conservar esto

    # Convertir a bytes PNG
    img_mascara = Image.fromarray(mascara, mode="RGBA")
    buffer = io.BytesIO()
    img_mascara.save(buffer, format="PNG")
    return buffer.getvalue()


def preparar_imagen_rgba(ruta_imagen: str) -> bytes:
    """
    Prepara una imagen en formato RGBA PNG, que es lo que requiere DALL-E para edición.
    La imagen debe ser cuadrada de 1024x1024.
    """
    img = Image.open(ruta_imagen).convert("RGBA")
    img = img.resize((1024, 1024), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def editar_imagen_inpainting(
    ruta_imagen: str,
    prompt_edicion: str,
    nombre_salida: str = "imagen_editada.png",
    region_editar: tuple = (512, 512, 200),  # (centro_x, centro_y, radio)
) -> str:
    """
    Edita una zona específica de una imagen con DALL-E.

    Args:
        ruta_imagen: Imagen original (se convertirá a 1024x1024 RGBA).
        prompt_edicion: Descripción de qué poner en la zona editada.
        nombre_salida: Nombre del archivo de salida.
        region_editar: Tupla (centro_x, centro_y, radio) definiendo la zona circular a editar.

    Returns:
        Ruta al archivo editado.
    """
    # Preparar imagen y máscara
    bytes_imagen = preparar_imagen_rgba(ruta_imagen)
    centro_x, centro_y, radio = region_editar
    bytes_mascara = crear_mascara_circular(1024, 1024, centro_x, centro_y, radio)

    cliente = openai.OpenAI()

    # Llamada a la API de edición (usa DALL-E 2, ya que DALL-E 3 no soporta edición aún)
    respuesta = cliente.images.edit(
        model="dall-e-2",
        image=("imagen.png", bytes_imagen, "image/png"),
        mask=("mascara.png", bytes_mascara, "image/png"),
        prompt=prompt_edicion,
        n=1,
        size="1024x1024",
    )

    url = respuesta.data[0].url
    contenido = requests.get(url).content
    ruta = Path(nombre_salida)
    ruta.write_bytes(contenido)
    print(f"Imagen editada guardada: {ruta.resolve()}")
    return str(ruta.resolve())


# --- Uso ---
if __name__ == "__main__":
    # Ejemplo: reemplazar el cielo de una fotografía
    editar_imagen_inpainting(
        ruta_imagen="paisaje.jpg",
        prompt_edicion="Un cielo dramático al atardecer con nubes naranjas y rosadas",
        nombre_salida="paisaje_editado.png",
        region_editar=(512, 200, 250),  # Zona superior de la imagen
    )
```

---

## 5. Stable Diffusion local con diffusers

Stable Diffusion se puede ejecutar completamente en local usando la biblioteca `diffusers` de Hugging Face. Esto garantiza privacidad total y sin costes por imagen, pero requiere una GPU con al menos 6 GB de VRAM (o paciencia en CPU).

```bash
pip install diffusers transformers torch accelerate
```

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path


def cargar_pipeline_sd(
    modelo_id: str = "runwayml/stable-diffusion-v1-5",
    dispositivo: str = "auto",
) -> StableDiffusionPipeline:
    """
    Carga el pipeline de Stable Diffusion en el dispositivo disponible.

    Args:
        modelo_id: ID del modelo en Hugging Face Hub.
        dispositivo: "cuda" para GPU NVIDIA, "mps" para Apple Silicon,
                     "cpu" para CPU, "auto" para detectar automáticamente.

    Returns:
        Pipeline de Stable Diffusion listo para generar imágenes.
    """
    if dispositivo == "auto":
        if torch.cuda.is_available():
            dispositivo = "cuda"
        elif torch.backends.mps.is_available():
            dispositivo = "mps"
        else:
            dispositivo = "cpu"

    print(f"Usando dispositivo: {dispositivo}")

    # Usar float16 en GPU para reducir memoria y acelerar
    dtype = torch.float16 if dispositivo in ("cuda", "mps") else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        modelo_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # Scheduler más rápido: DPM-Solver++ (20 pasos vs 50 del DDIM original)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    pipeline = pipeline.to(dispositivo)

    # Optimización de memoria en CUDA
    if dispositivo == "cuda":
        pipeline.enable_attention_slicing()

    return pipeline


def generar_imagen_sd(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    prompt_negativo: str = "low quality, blurry, distorted, ugly, bad anatomy",
    nombre_archivo: str = "sd_imagen.png",
    pasos: int = 20,
    escala_guia: float = 7.5,
    semilla: int = None,
) -> str:
    """
    Genera una imagen con Stable Diffusion.

    Args:
        pipeline: Pipeline cargado con cargar_pipeline_sd().
        prompt: Descripción positiva de la imagen.
        prompt_negativo: Elementos a evitar en la imagen.
        nombre_archivo: Ruta del archivo de salida.
        pasos: Número de pasos de difusión (más = más calidad, más lento).
        escala_guia: Fuerza del prompt (7-12 para prompts estrictos, 3-5 para creatividad).
        semilla: Semilla aleatoria para reproducibilidad (None = aleatoria).

    Returns:
        Ruta al archivo guardado.
    """
    generador = None
    if semilla is not None:
        dispositivo = pipeline.device.type
        generador = torch.Generator(device=dispositivo).manual_seed(semilla)

    resultado = pipeline(
        prompt=prompt,
        negative_prompt=prompt_negativo,
        num_inference_steps=pasos,
        guidance_scale=escala_guia,
        generator=generador,
        width=512,
        height=512,
    )

    imagen = resultado.images[0]
    ruta = Path(nombre_archivo)
    imagen.save(ruta)
    print(f"Imagen guardada: {ruta.resolve()}")
    return str(ruta.resolve())


def generar_variaciones_sd(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    cantidad: int = 4,
    carpeta: str = "variaciones_sd",
) -> list[str]:
    """
    Genera múltiples variaciones de un mismo prompt con semillas distintas.
    """
    Path(carpeta).mkdir(exist_ok=True)
    rutas = []
    for i in range(cantidad):
        nombre = f"{carpeta}/variacion_{i+1:02d}.png"
        ruta = generar_imagen_sd(
            pipeline=pipeline,
            prompt=prompt,
            nombre_archivo=nombre,
            semilla=i * 42,  # Semillas distintas para variaciones distintas
        )
        rutas.append(ruta)
    return rutas


# --- Uso ---
if __name__ == "__main__":
    # Cargar el pipeline una sola vez (tarda ~2 min la primera vez por la descarga)
    pipe = cargar_pipeline_sd()

    # Generar una imagen
    generar_imagen_sd(
        pipeline=pipe,
        prompt=(
            "A futuristic city at night, neon lights reflecting on wet streets, "
            "cyberpunk aesthetic, detailed architecture, cinematic lighting"
        ),
        prompt_negativo="low quality, blurry, oversaturated, ugly buildings",
        nombre_archivo="ciudad_cyberpunk.png",
        pasos=25,
        semilla=1234,
    )

    # Generar 4 variaciones
    variaciones = generar_variaciones_sd(
        pipeline=pipe,
        prompt="A serene Japanese garden in spring, cherry blossoms, koi pond, traditional architecture",
        cantidad=4,
    )
    print(f"Variaciones generadas: {variaciones}")
```

---

## 6. Pipeline LLM + generación de imágenes

Un patrón muy potente es usar un LLM para **crear el prompt de imagen** a partir de una descripción en lenguaje natural, y luego pasar ese prompt optimizado al generador de imágenes. Esto permite a usuarios sin experiencia en prompt engineering obtener mejores resultados.

```python
import anthropic
import openai
import requests
from pathlib import Path


def generar_prompt_imagen(descripcion_usuario: str, estilo: str = "fotografico") -> str:
    """
    Usa Claude para convertir una descripción simple en un prompt
    optimizado para DALL-E 3.

    Args:
        descripcion_usuario: Lo que el usuario quiere en lenguaje natural.
        estilo: "fotografico", "ilustracion", "pintura", "minimalista", "anime".

    Returns:
        Prompt optimizado para generación de imágenes.
    """
    estilos = {
        "fotografico": "professional photography, DSLR camera, realistic lighting, high detail",
        "ilustracion": "digital illustration, vibrant colors, clean lines, professional design",
        "pintura": "oil painting on canvas, impressionist style, rich textures, museum quality",
        "minimalista": "minimalist, clean design, flat colors, simple shapes, white background",
        "anime": "anime art style, Studio Ghibli inspired, detailed, beautiful backgrounds",
    }

    modificador_estilo = estilos.get(estilo, estilos["fotografico"])

    cliente_claude = anthropic.Anthropic()

    mensaje = cliente_claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=(
            "Eres un experto en prompt engineering para generación de imágenes con IA. "
            "Tu tarea es convertir la descripción del usuario en un prompt optimizado para DALL-E 3. "
            "El prompt debe ser descriptivo, específico y en inglés. "
            "Incluye detalles sobre composición, iluminación y atmósfera. "
            "Responde ÚNICAMENTE con el prompt, sin explicaciones ni comillas."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Descripción del usuario: {descripcion_usuario}\n"
                    f"Estilo visual a aplicar: {modificador_estilo}\n\n"
                    f"Crea el prompt optimizado en inglés:"
                ),
            }
        ],
    )

    return mensaje.content[0].text.strip()


def pipeline_descripcion_a_imagen(
    descripcion: str,
    estilo: str = "fotografico",
    nombre_salida: str = "resultado_pipeline.png",
    calidad: str = "hd",
) -> dict:
    """
    Pipeline completo: descripción en español → Claude → prompt → DALL-E 3 → imagen.

    Args:
        descripcion: Descripción en lenguaje natural (puede ser en español).
        estilo: Estilo visual deseado.
        nombre_salida: Archivo de salida.
        calidad: "standard" o "hd".

    Returns:
        Diccionario con el prompt generado y la ruta del archivo.
    """
    print("Paso 1: Generando prompt optimizado con Claude...")
    prompt_optimizado = generar_prompt_imagen(descripcion, estilo)
    print(f"Prompt generado: {prompt_optimizado}\n")

    print("Paso 2: Generando imagen con DALL-E 3...")
    cliente_openai = openai.OpenAI()

    respuesta = cliente_openai.images.generate(
        model="dall-e-3",
        prompt=prompt_optimizado,
        size="1024x1024",
        quality=calidad,
        style="vivid",
        n=1,
    )

    url = respuesta.data[0].url
    prompt_revisado = respuesta.data[0].revised_prompt

    print(f"Prompt revisado por DALL-E: {prompt_revisado}\n")

    print("Paso 3: Guardando imagen...")
    contenido = requests.get(url).content
    ruta = Path(nombre_salida)
    ruta.write_bytes(contenido)
    print(f"Imagen guardada: {ruta.resolve()}")

    return {
        "descripcion_original": descripcion,
        "prompt_claude": prompt_optimizado,
        "prompt_dalle": prompt_revisado,
        "ruta_imagen": str(ruta.resolve()),
    }


# --- Uso ---
if __name__ == "__main__":
    # El usuario describe lo que quiere en español, sin preocuparse por el prompt
    resultado = pipeline_descripcion_a_imagen(
        descripcion=(
            "Una tortuga marina nadando entre corales de colores en el Mar Caribe, "
            "con rayos de sol penetrando el agua azul turquesa"
        ),
        estilo="fotografico",
        nombre_salida="tortuga_marina.png",
        calidad="hd",
    )

    print("\n=== RESULTADO DEL PIPELINE ===")
    for clave, valor in resultado.items():
        print(f"{clave}: {valor}")

    # Otro ejemplo con estilo diferente
    resultado2 = pipeline_descripcion_a_imagen(
        descripcion="Mi abuela haciendo pan en una cocina de pueblo, atmósfera cálida y nostálgica",
        estilo="pintura",
        nombre_salida="abuela_panadería.png",
    )
```

---

## 7. Consideraciones éticas y de uso

### Derechos de autor y propiedad intelectual

- Las imágenes generadas con DALL-E 3 son propiedad del usuario según los términos de OpenAI, pero no puedes reclamar derechos sobre los estilos de artistas reales.
- Evita pedir imágenes "al estilo de [artista vivo]" en contextos comerciales: puede ser legalmente problemático.
- Los modelos como Stable Diffusion fueron entrenados con imágenes de Internet; existe debate sobre el consentimiento de los artistas originales.

### Deepfakes y contenido dañino

- **Nunca generes imágenes de personas reales** en situaciones comprometedoras, violentas o sin su consentimiento.
- Las APIs de OpenAI y Anthropic tienen filtros de contenido activos. Violar sus políticas de uso puede resultar en suspensión de la cuenta.
- Para Stable Diffusion local: eres responsable del uso que hagas del modelo.

### Transparencia

- Etiqueta siempre las imágenes generadas por IA cuando se publiquen como si fueran reales.
- En contextos periodísticos, legales o médicos, una imagen generada por IA puede ser peligrosamente engañosa.

### Huella de carbono

- Generar imágenes con modelos grandes consume energía. DALL-E 3 en la nube tiene una huella controlada por OpenAI. Stable Diffusion local en GPU también consume energía; prefiere fuentes renovables si generas en volumen.

---

## 8. Extensiones sugeridas

- **Galería web automática**: genera una colección de imágenes temáticas y crea una página HTML con todas ellas usando Python y Jinja2.
- **Variaciones con feedback**: muestra la imagen al usuario, recibe su opinión en lenguaje natural, y vuelve a generarla con las correcciones.
- **ControlNet con diffusers**: guía la generación de Stable Diffusion con un boceto o pose específica usando ControlNet.
- **Upscaling con Real-ESRGAN**: mejora la resolución de imágenes generadas de 512px a 2048px con modelos de super-resolución.
- **Generación en lote para redes sociales**: dado un CSV con conceptos, genera una imagen por fila y las guarda con nombres descriptivos.

---

**Siguiente:** [03 — Voz e IA](./03-voz-ia.md)
