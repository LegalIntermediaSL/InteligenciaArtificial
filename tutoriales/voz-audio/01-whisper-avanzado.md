# Whisper Avanzado: Transcripción, Diarización y Post-procesado

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/voz-audio/01-whisper-avanzado.ipynb)

Whisper de OpenAI es el estándar de facto para transcripción de audio. Este tutorial cubre los casos que van más allá del uso básico: archivos largos, múltiples hablantes, idiomas mezclados y limpieza del texto con Claude.

---

## 1. Whisper: API vs local

| | OpenAI API | faster-whisper (local) |
|---|---|---|
| Coste | $0.006/min | Gratis |
| Velocidad | Rápida (cloud) | Depende de GPU |
| Privacidad | Audio a OpenAI | 100% local |
| Límite | 25 MB por archivo | Sin límite |
| Idiomas | 99 idiomas | 99 idiomas |

**Regla práctica**: usa la API para prototipos y archivos cortos; `faster-whisper` local para producción con datos sensibles o volumen alto.

---

## 2. Chunking para archivos largos

La API de OpenAI tiene un límite de 25 MB. Para audios largos (reuniones, podcasts, entrevistas), hay que partir el archivo:

```python
from pydub import AudioSegment
import math

def chunkar_audio(ruta: str, max_mb: float = 20.0) -> list[str]:
    """Divide un archivo de audio en chunks de tamaño máximo."""
    audio = AudioSegment.from_file(ruta)
    
    # Calcular duración máxima por chunk
    tam_archivo_mb = len(audio.raw_data) / (1024 * 1024)
    n_chunks = math.ceil(tam_archivo_mb / max_mb)
    duracion_chunk_ms = len(audio) // n_chunks
    
    rutas_chunks = []
    for i in range(n_chunks):
        inicio = i * duracion_chunk_ms
        fin = min((i + 1) * duracion_chunk_ms, len(audio))
        chunk = audio[inicio:fin]
        
        ruta_chunk = f"/tmp/chunk_{i:03d}.mp3"
        chunk.export(ruta_chunk, format="mp3", bitrate="64k")
        rutas_chunks.append(ruta_chunk)
    
    return rutas_chunks

def transcribir_audio_largo(ruta: str, idioma: str = "es") -> str:
    """Transcribe un archivo de audio largo con chunking automático."""
    import openai
    client = openai.OpenAI()
    
    chunks = chunkar_audio(ruta)
    transcripciones = []
    
    for i, chunk in enumerate(chunks):
        with open(chunk, "rb") as f:
            respuesta = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=idioma,
                response_format="verbose_json",  # incluye timestamps
                timestamp_granularities=["word"],
            )
        transcripciones.append(respuesta.text)
        print(f"  Chunk {i+1}/{len(chunks)} transcrito")
    
    return " ".join(transcripciones)
```

---

## 3. Transcripción con timestamps de palabras

```python
def transcribir_con_timestamps(ruta_audio: str) -> dict:
    """Transcribe con timestamps a nivel de palabra."""
    import openai
    client = openai.OpenAI()
    
    with open(ruta_audio, "rb") as f:
        respuesta = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    
    return {
        "texto": respuesta.text,
        "idioma": respuesta.language,
        "duracion": respuesta.duration,
        "palabras": [
            {
                "palabra": w.word,
                "inicio": w.start,
                "fin": w.end,
            }
            for w in (respuesta.words or [])
        ],
        "segmentos": [
            {
                "texto": s.text,
                "inicio": s.start,
                "fin": s.end,
            }
            for s in (respuesta.segments or [])
        ],
    }
```

---

## 4. Diarización: quién habla cuándo

Whisper no distingue hablantes de forma nativa. La solución más sencilla es combinar `faster-whisper` con `pyannote-audio`:

```python
# pip install faster-whisper pyannote.audio

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

def transcribir_con_diarizacion(ruta_audio: str, hf_token: str) -> list[dict]:
    """
    Transcribe y asigna cada fragmento al hablante correspondiente.
    Requiere token de HuggingFace con acceso a pyannote/speaker-diarization-3.1
    """
    # 1. Diarización: quién habla cuándo
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarizacion = pipeline(ruta_audio)
    
    # 2. Transcripción con timestamps de segmentos
    modelo = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segmentos, _ = modelo.transcribe(ruta_audio, language="es")
    
    # 3. Cruzar transcripción con diarización
    resultado = []
    for segmento in segmentos:
        # Encontrar qué hablante domina en este segmento
        t_medio = (segmento.start + segmento.end) / 2
        hablante = "DESCONOCIDO"
        for turno, _, etiqueta in diarizacion.itertracks(yield_label=True):
            if turno.start <= t_medio <= turno.end:
                hablante = etiqueta
                break
        
        resultado.append({
            "inicio": round(segmento.start, 2),
            "fin": round(segmento.end, 2),
            "hablante": hablante,
            "texto": segmento.text.strip(),
        })
    
    return resultado

def formatear_transcripcion(segmentos: list[dict]) -> str:
    """Formatea los segmentos como una transcripción legible."""
    lineas = []
    hablante_actual = None
    for seg in segmentos:
        if seg["hablante"] != hablante_actual:
            hablante_actual = seg["hablante"]
            lineas.append(f"\n**{hablante_actual}** [{seg['inicio']}s]")
        lineas.append(f"  {seg['texto']}")
    return "\n".join(lineas)
```

---

## 5. Post-procesado con Claude

El texto crudo de Whisper suele necesitar limpieza. Claude es ideal para esto:

```python
import anthropic

client = anthropic.Anthropic()

def limpiar_transcripcion(texto_crudo: str, contexto: str = "") -> str:
    """
    Limpia y mejora una transcripción: puntuación, nombres propios,
    corrección de errores fonéticos, formato de párrafos.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Eres un editor especializado en transcripciones.
Limpia y mejora el siguiente texto transcrito de audio.

CONTEXTO (si se conoce): {contexto or 'No especificado'}

INSTRUCCIONES:
- Añade puntuación correcta (puntos, comas, signos de interrogación)
- Corrige errores fonéticos obvios (ej: "a ver" vs "haber")
- Separa en párrafos según cambios de tema
- Mantén el texto original sin añadir información
- No resumas ni parafrasees

TRANSCRIPCIÓN CRUDA:
{texto_crudo}

TRANSCRIPCIÓN LIMPIA:""",
        }],
    )
    return response.content[0].text

def resumir_reunion(transcripcion: str) -> dict:
    """Extrae puntos clave y tareas de una transcripción de reunión."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[{
            "name": "resumen_reunion",
            "description": "Resumen estructurado de una reunión",
            "input_schema": {
                "type": "object",
                "properties": {
                    "resumen_ejecutivo": {"type": "string"},
                    "puntos_clave": {"type": "array", "items": {"type": "string"}},
                    "decisiones": {"type": "array", "items": {"type": "string"}},
                    "tareas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tarea": {"type": "string"},
                                "responsable": {"type": "string"},
                                "plazo": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["resumen_ejecutivo", "puntos_clave", "decisiones", "tareas"],
            },
        }],
        tool_choice={"type": "tool", "name": "resumen_reunion"},
        messages=[{"role": "user", "content": f"Resume esta reunión:\n\n{transcripcion}"}],
    )
    return next(b for b in response.content if b.type == "tool_use").input
```

---

## 6. Generación de subtítulos SRT

```python
def generar_srt(segmentos: list[dict], ruta_salida: str = "subtitulos.srt") -> str:
    """Genera un archivo SRT a partir de los segmentos de Whisper."""
    
    def segundos_a_srt(segundos: float) -> str:
        h = int(segundos // 3600)
        m = int((segundos % 3600) // 60)
        s = int(segundos % 60)
        ms = int((segundos % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    lineas = []
    for i, seg in enumerate(segmentos, 1):
        lineas.append(str(i))
        lineas.append(f"{segundos_a_srt(seg['inicio'])} --> {segundos_a_srt(seg['fin'])}")
        lineas.append(seg["texto"].strip())
        lineas.append("")
    
    contenido = "\n".join(lineas)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(contenido)
    
    return contenido

# Uso
# segmentos = transcribir_con_timestamps("video.mp4")["segmentos"]
# srt = generar_srt(segmentos, "video.srt")
```

---

## Cuándo usar cada herramienta

| Caso | Herramienta |
|------|-------------|
| Prototipo rápido, audio < 25MB | OpenAI Whisper API |
| Datos sensibles o volumen alto | faster-whisper local |
| Múltiples hablantes | faster-whisper + pyannote-audio |
| Limpieza de texto | Claude Haiku |
| Resumen de reuniones | Claude Sonnet |
| Subtítulos para vídeo | Whisper + generador SRT |
