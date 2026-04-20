# Video e IA — Generación, análisis y transcripción

El video es la frontera más activa de la IA multimodal en 2026.
Este artículo cubre análisis de video con Gemini, generación con APIs de texto-a-video
y pipelines de transcripción y resumen.

---

## 1. Análisis de video con Google Gemini

Gemini 1.5 Pro acepta video directamente (hasta 1 hora). Claude actualmente solo procesa
fotogramas individuales como imágenes.

```python
import google.generativeai as genai
import os, time
from pathlib import Path

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def analizar_video_archivo(ruta_video: str, pregunta: str) -> str:
    """Sube un archivo de video y lo analiza con Gemini."""
    # Subir el video a la API de Google
    archivo = genai.upload_file(ruta_video, mime_type='video/mp4')

    # Esperar a que el archivo esté procesado
    while archivo.state.name == 'PROCESSING':
        print('Procesando video...')
        time.sleep(2)
        archivo = genai.get_file(archivo.name)

    if archivo.state.name == 'FAILED':
        raise ValueError(f'El procesamiento del video falló: {archivo.state.name}')

    modelo = genai.GenerativeModel('gemini-1.5-pro')
    respuesta = modelo.generate_content([archivo, pregunta])

    # Limpiar el archivo después de usarlo
    genai.delete_file(archivo.name)

    return respuesta.text

# Análisis completo de un video de presentación
resultado = analizar_video_archivo(
    'presentacion_producto.mp4',
    '''Analiza este video de presentación y devuelve:
    1. Resumen ejecutivo (3 puntos)
    2. Momentos clave con timestamps aproximados
    3. Tono y calidad de la presentación (1-10)
    4. Sugerencias de mejora'''
)
print(resultado)
```

---

## 2. Extraer fotogramas y analizarlos con Claude

Para videos cortos o cuando no tienes acceso a Gemini, puedes extraer fotogramas y
analizarlos con Claude Vision:

```python
import anthropic
import base64
import subprocess
from pathlib import Path

client = anthropic.Anthropic()

def extraer_fotogramas(ruta_video: str, fps: float = 1.0, directorio_salida: str = '/tmp/frames') -> list[str]:
    """Extrae fotogramas de un video usando ffmpeg."""
    Path(directorio_salida).mkdir(parents=True, exist_ok=True)
    patron_salida = f'{directorio_salida}/frame_%04d.jpg'

    subprocess.run([
        'ffmpeg', '-i', ruta_video,
        '-vf', f'fps={fps}',       # 1 fotograma por segundo
        '-q:v', '2',               # calidad alta
        patron_salida,
        '-y', '-loglevel', 'quiet'
    ], check=True)

    return sorted(Path(directorio_salida).glob('frame_*.jpg'))

def imagen_a_base64(ruta: str) -> str:
    with open(ruta, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode()

def analizar_fotogramas_claude(fotogramas: list[str], pregunta: str) -> str:
    """Analiza múltiples fotogramas con Claude Vision."""
    # Máximo ~20 imágenes para no superar el límite de tokens
    fotogramas_seleccionados = fotogramas[::max(1, len(fotogramas) // 20)][:20]

    contenido = []
    for i, fotograma in enumerate(fotogramas_seleccionados):
        contenido.append({
            'type': 'text',
            'text': f'Fotograma {i + 1} (segundo ~{i * len(fotogramas) // len(fotogramas_seleccionados)}):'
        })
        contenido.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': 'image/jpeg',
                'data': imagen_a_base64(str(fotograma)),
            }
        })

    contenido.append({'type': 'text', 'text': f'\nPregunta: {pregunta}'})

    resp = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=1024,
        messages=[{'role': 'user', 'content': contenido}]
    )
    return resp.content[0].text

# Pipeline completo
def analizar_video_con_claude(ruta_video: str, pregunta: str) -> str:
    fotogramas = extraer_fotogramas(ruta_video, fps=0.5)  # 1 fotograma cada 2 segundos
    print(f'Extraídos {len(fotogramas)} fotogramas')
    return analizar_fotogramas_claude(fotogramas, pregunta)
```

---

## 3. Transcripción de video con Whisper

```python
import anthropic
import subprocess
from pathlib import Path

client = anthropic.Anthropic()

def extraer_audio_de_video(ruta_video: str, ruta_audio: str = '/tmp/audio.mp3') -> str:
    """Extrae el audio de un video con ffmpeg."""
    subprocess.run([
        'ffmpeg', '-i', ruta_video,
        '-q:a', '0', '-map', 'a',
        ruta_audio, '-y', '-loglevel', 'quiet'
    ], check=True)
    return ruta_audio

def transcribir_audio(ruta_audio: str, idioma: str = 'es') -> str:
    """Transcribe audio usando la API de Whisper de OpenAI."""
    from openai import OpenAI
    openai_client = OpenAI()

    with open(ruta_audio, 'rb') as f:
        transcripcion = openai_client.audio.transcriptions.create(
            model='whisper-1',
            file=f,
            language=idioma,
            response_format='verbose_json',  # incluye timestamps
        )

    return transcripcion

def transcribir_audio_largo(ruta_audio: str, duracion_chunk: int = 600) -> str:
    """Divide audios >25MB en chunks de 10 minutos."""
    from openai import OpenAI
    import math

    openai_client = OpenAI()
    tamaño = Path(ruta_audio).stat().st_size

    if tamaño < 25 * 1024 * 1024:  # < 25MB
        return transcribir_audio(ruta_audio)

    # Dividir con ffmpeg
    duracion_total = int(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', ruta_audio
    ]).decode().strip().split('.')[0])

    n_chunks = math.ceil(duracion_total / duracion_chunk)
    transcripciones = []

    for i in range(n_chunks):
        chunk_path = f'/tmp/chunk_{i:03d}.mp3'
        subprocess.run([
            'ffmpeg', '-i', ruta_audio,
            '-ss', str(i * duracion_chunk),
            '-t', str(duracion_chunk),
            chunk_path, '-y', '-loglevel', 'quiet'
        ])
        transcripciones.append(transcribir_audio(chunk_path))
        Path(chunk_path).unlink()

    return ' '.join(transcripciones)

def resumir_transcripcion(transcripcion: str) -> dict:
    """Usa Claude para generar un resumen estructurado de la transcripción."""
    resp = client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=800,
        messages=[{'role': 'user', 'content': f'''Analiza esta transcripción de video y devuelve JSON:
{{
  "duracion_estimada_min": 0,
  "temas_principales": [],
  "puntos_clave": [],
  "citas_destacadas": [],
  "accion_items": [],
  "resumen_ejecutivo": "..."
}}

TRANSCRIPCIÓN:
{transcripcion}'''}]
    )
    import json
    texto = resp.content[0].text
    if '```' in texto:
        texto = texto.split('```')[1].replace('json', '').strip()
    return json.loads(texto)

# Pipeline completo: video → audio → transcripción → resumen
def pipeline_video_completo(ruta_video: str) -> dict:
    print('1. Extrayendo audio...')
    ruta_audio = extraer_audio_de_video(ruta_video)

    print('2. Transcribiendo...')
    transcripcion = transcribir_audio_largo(ruta_audio)

    print('3. Resumiendo con Claude...')
    resumen = resumir_transcripcion(transcripcion.text if hasattr(transcripcion, 'text') else transcripcion)

    Path(ruta_audio).unlink(missing_ok=True)

    return {
        'transcripcion': transcripcion,
        'resumen': resumen,
    }
```

---

## 4. Generación de video con APIs de texto-a-video

En 2026 las principales APIs de generación de video son:

```python
import httpx, os, time, json
from pathlib import Path

# Runway ML — API REST
class RunwayClient:
    BASE_URL = 'https://api.runwayml.com/v1'

    def __init__(self, api_key: str):
        self.client = httpx.Client(
            headers={'Authorization': f'Bearer {api_key}', 'X-Runway-Version': '2024-11-06'},
            base_url=self.BASE_URL,
        )

    def texto_a_video(
        self,
        prompt: str,
        duracion: int = 5,       # segundos: 5 o 10
        ratio: str = '1280:768', # o '768:1280' para vertical
    ) -> str:
        """Crea un video a partir de texto. Devuelve la URL del video."""
        resp = self.client.post('/image_to_video', json={
            'promptText': prompt,
            'duration': duracion,
            'ratio': ratio,
        })
        resp.raise_for_status()
        tarea_id = resp.json()['id']

        # Polling hasta que termine
        while True:
            estado = self.client.get(f'/tasks/{tarea_id}').json()
            if estado['status'] == 'SUCCEEDED':
                return estado['output'][0]
            if estado['status'] == 'FAILED':
                raise Exception(f'Generación fallida: {estado.get("failure")}')
            time.sleep(5)

    def imagen_a_video(self, url_imagen: str, prompt: str = '', duracion: int = 5) -> str:
        """Anima una imagen estática."""
        resp = self.client.post('/image_to_video', json={
            'promptImage': url_imagen,
            'promptText': prompt,
            'duration': duracion,
        })
        resp.raise_for_status()
        return self._esperar_resultado(resp.json()['id'])

    def _esperar_resultado(self, tarea_id: str) -> str:
        while True:
            estado = self.client.get(f'/tasks/{tarea_id}').json()
            if estado['status'] == 'SUCCEEDED':
                return estado['output'][0]
            if estado['status'] in ('FAILED', 'CANCELLED'):
                raise Exception(f'Tarea {estado["status"]}: {estado.get("failure")}')
            time.sleep(5)

# Uso
runway = RunwayClient(os.getenv('RUNWAY_API_KEY', ''))

# Pipeline: Claude mejora el prompt → Runway genera el video
def generar_video_con_claude(descripcion_usuario: str) -> str:
    import anthropic
    client = anthropic.Anthropic()

    # Claude optimiza el prompt para generación de video
    resp = client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=200,
        messages=[{'role': 'user', 'content': f'''Convierte esta descripción en un prompt optimizado
para generación de video con IA. Incluye: movimiento de cámara, iluminación, estilo visual.
Máximo 200 palabras. Solo el prompt, sin explicaciones.

Descripción: {descripcion_usuario}'''}]
    )
    prompt_optimizado = resp.content[0].text

    print(f'Prompt optimizado: {prompt_optimizado[:100]}...')
    url_video = runway.texto_a_video(prompt_optimizado, duracion=5)
    return url_video
```

---

## 5. Subtítulos automáticos con Claude

```python
import anthropic
import json

client = anthropic.Anthropic()

def transcripcion_a_srt(transcripcion_con_timestamps: list[dict]) -> str:
    """Convierte timestamps de Whisper al formato SRT."""
    lineas = []
    for i, segmento in enumerate(transcripcion_con_timestamps, 1):
        inicio = formatear_tiempo_srt(segmento['start'])
        fin = formatear_tiempo_srt(segmento['end'])
        texto = segmento['text'].strip()
        lineas.append(f'{i}\n{inicio} --> {fin}\n{texto}\n')
    return '\n'.join(lineas)

def formatear_tiempo_srt(segundos: float) -> str:
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    ms = int((segundos % 1) * 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def mejorar_subtitulos_con_claude(texto_srt: str) -> str:
    """Mejora la puntuación y el formato de subtítulos con Claude."""
    resp = client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=2000,
        messages=[{'role': 'user', 'content': f'''Mejora estos subtítulos SRT:
1. Corrige errores de puntuación y ortografía
2. Divide líneas muy largas (máximo 42 caracteres por línea)
3. Mantiene exactamente el mismo formato SRT y los mismos timestamps
4. No cambies el contenido, solo corrije errores obvios

SUBTÍTULOS:
{texto_srt}

Devuelve solo el SRT mejorado, sin explicaciones.'''}]
    )
    return resp.content[0].text
```

---

## 6. Casos de uso empresariales

| Caso de uso | Tecnología | Coste estimado |
|---|---|---|
| Transcripción de reuniones (60 min) | Whisper + Claude resumen | ~$0.08 |
| Análisis de video de formación | Gemini 1.5 Pro | ~$0.25/video |
| Subtítulos automáticos (30 min) | Whisper + SRT | ~$0.04 |
| Generación video promocional (5s) | Runway Gen-3 | ~$0.50 |
| Demo de producto animado | Imagen → Video | ~$0.50 |

---

→ Anterior: [Voz e IA](03-voz-ia.md) | → Siguiente: [Bloque 25 — Proyectos integradores](../proyectos-integradores/README.md)
