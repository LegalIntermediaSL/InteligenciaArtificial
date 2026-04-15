# 03 — Voz e IA: transcripción, síntesis y pipelines de voz

> **Bloque:** Multimodalidad · **Nivel:** Práctico · **Tiempo estimado:** 75 min

---

## Índice

1. [Ecosistema de voz en IA](#1-ecosistema-de-voz-en-ia)
2. [Transcripción con Whisper local](#2-transcripción-con-whisper-local)
3. [Transcripción via API de OpenAI](#3-transcripción-via-api-de-openai)
4. [Síntesis de voz (TTS) con OpenAI](#4-síntesis-de-voz-tts-con-openai)
5. [Pipeline completo voz → texto → IA → voz](#5-pipeline-completo-voz--texto--ia--voz)
6. [Caso práctico — asistente de voz para reuniones](#6-caso-práctico--asistente-de-voz-para-reuniones)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Ecosistema de voz en IA

El trabajo con voz en IA se divide en tres grandes categorías:

### STT — Speech-to-Text (voz a texto)

Convierte audio hablado en texto escrito. Casos de uso: transcripción de reuniones, subtítulos automáticos, asistentes de voz, dictado.

| Modelo | Empresa | Acceso | Idiomas | Velocidad |
|---|---|---|---|---|
| Whisper large-v3 | OpenAI | Local / API | 99 | Moderada (local GPU) |
| Deepgram Nova | Deepgram | API | 30+ | Muy rápida (streaming) |
| Google Speech-to-Text | Google | API | 125 | Rápida |
| Azure Speech | Microsoft | API | 100+ | Rápida |
| Faster-Whisper | Community | Local | 99 | Muy rápida (CTranslate2) |

### TTS — Text-to-Speech (texto a voz)

Convierte texto escrito en audio hablado. Casos de uso: lectores de pantalla, asistentes de voz, audiolibros automáticos, notificaciones.

| Modelo | Empresa | Calidad | Voces |
|---|---|---|---|
| TTS-1 / TTS-1-HD | OpenAI | Alta | 6 voces |
| ElevenLabs | ElevenLabs | Muy alta (clonación de voz) | Ilimitadas |
| Google Cloud TTS | Google | Alta | 380+ voces |
| Azure Neural TTS | Microsoft | Alta | 400+ voces |

### Pipelines completos

Combinan STT + LLM + TTS para crear asistentes de voz end-to-end:

```
Micrófono → Grabación → Whisper (STT) → Claude (razonamiento) → OpenAI TTS → Altavoz
```

---

## 2. Transcripción con Whisper local

Whisper es el modelo de transcripción de OpenAI. La versión local se ejecuta en tu máquina sin necesidad de API key ni conexión a Internet.

```bash
pip install openai-whisper
# En macOS con Apple Silicon también necesitas:
pip install torch torchaudio
```

### Transcripción básica

```python
import whisper
from pathlib import Path


def transcribir_audio_local(
    ruta_audio: str,
    modelo: str = "base",
    idioma: str = None,
) -> dict:
    """
    Transcribe un archivo de audio usando Whisper local.

    Modelos disponibles (más pequeño = más rápido, menos preciso):
    - "tiny"   → ~39 MB, muy rápido, baja precisión
    - "base"   → ~74 MB, rápido, buena precisión
    - "small"  → ~244 MB, moderado, mejor precisión
    - "medium" → ~769 MB, lento, alta precisión
    - "large"  → ~1.5 GB, muy lento, máxima precisión

    Args:
        ruta_audio: Ruta al archivo de audio (MP3, WAV, M4A, FLAC, OGG, etc.).
        modelo: Tamaño del modelo de Whisper.
        idioma: Código de idioma ISO (ej. "es", "en", "fr"). None = detección automática.

    Returns:
        Diccionario con "texto", "idioma" y "segmentos" (con timestamps).
    """
    print(f"Cargando modelo Whisper '{modelo}'...")
    modelo_whisper = whisper.load_model(modelo)

    ruta = Path(ruta_audio)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_audio}")

    print(f"Transcribiendo: {ruta.name}")

    opciones = {}
    if idioma:
        opciones["language"] = idioma

    resultado = modelo_whisper.transcribe(str(ruta), **opciones)

    return {
        "texto": resultado["text"].strip(),
        "idioma": resultado["language"],
        "segmentos": resultado["segments"],
    }


def transcribir_con_timestamps(ruta_audio: str, modelo: str = "base") -> str:
    """
    Transcribe un audio y devuelve el texto con timestamps en formato SRT simplificado.
    Útil para crear subtítulos.
    """
    resultado = transcribir_audio_local(ruta_audio, modelo)
    lineas = []

    for segmento in resultado["segmentos"]:
        inicio = segmento["start"]
        fin = segmento["end"]
        texto = segmento["text"].strip()

        # Formato: [MM:SS.ms] texto
        inicio_fmt = f"{int(inicio // 60):02d}:{inicio % 60:06.3f}"
        fin_fmt = f"{int(fin // 60):02d}:{fin % 60:06.3f}"
        lineas.append(f"[{inicio_fmt} → {fin_fmt}] {texto}")

    return "\n".join(lineas)


def exportar_subtitulos_srt(ruta_audio: str, ruta_srt: str, modelo: str = "base") -> None:
    """
    Genera un archivo de subtítulos .srt a partir de un audio.
    """
    resultado = transcribir_audio_local(ruta_audio, modelo)

    with open(ruta_srt, "w", encoding="utf-8") as f:
        for i, segmento in enumerate(resultado["segmentos"], 1):
            inicio = segmento["start"]
            fin = segmento["end"]
            texto = segmento["text"].strip()

            # Formato SRT: HH:MM:SS,mmm
            def segundos_a_srt(s):
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                seg = s % 60
                return f"{h:02d}:{m:02d}:{seg:06.3f}".replace(".", ",")

            f.write(f"{i}\n")
            f.write(f"{segundos_a_srt(inicio)} --> {segundos_a_srt(fin)}\n")
            f.write(f"{texto}\n\n")

    print(f"Subtítulos guardados en: {ruta_srt}")


# --- Uso ---
if __name__ == "__main__":
    # Transcripción simple
    resultado = transcribir_audio_local("reunion.mp3", modelo="small", idioma="es")
    print("=== TRANSCRIPCIÓN ===")
    print(resultado["texto"])
    print(f"\nIdioma detectado: {resultado['idioma']}")

    # Con timestamps
    print("\n=== CON TIMESTAMPS ===")
    texto_timestamps = transcribir_con_timestamps("reunion.mp3")
    print(texto_timestamps)

    # Exportar subtítulos
    exportar_subtitulos_srt("reunion.mp3", "reunion.srt", modelo="base")
```

---

## 3. Transcripción via API de OpenAI

La API de OpenAI ofrece Whisper como servicio en la nube. Es más rápida que ejecutarlo localmente (especialmente sin GPU) y no requiere descargar el modelo.

```bash
pip install openai pydub
```

```python
import openai
from pathlib import Path
from pydub import AudioSegment
import math


def transcribir_con_api_openai(
    ruta_audio: str,
    idioma: str = "es",
    formato_respuesta: str = "text",
) -> str:
    """
    Transcribe audio usando la API de OpenAI (Whisper en la nube).

    Args:
        ruta_audio: Ruta al archivo de audio.
        idioma: Código de idioma ISO 639-1 (ej. "es", "en"). Mejora la precisión.
        formato_respuesta: "text" (solo texto), "verbose_json" (con timestamps y metadatos),
                           "srt" (subtítulos SRT), "vtt" (subtítulos WebVTT).

    Returns:
        Transcripción en el formato solicitado.
    """
    cliente = openai.OpenAI()  # Usa OPENAI_API_KEY del entorno

    ruta = Path(ruta_audio)
    with open(ruta, "rb") as f:
        transcripcion = cliente.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=idioma,
            response_format=formato_respuesta,
        )

    # Para "text", devuelve un string directamente
    if formato_respuesta == "text":
        return transcripcion

    # Para "verbose_json" devuelve un objeto con atributos
    if formato_respuesta == "verbose_json":
        return transcripcion.text

    # Para "srt" y "vtt" devuelve el texto del formato
    return transcripcion


def transcribir_audio_largo(
    ruta_audio: str,
    idioma: str = "es",
    tamano_fragmento_min: int = 10,
) -> str:
    """
    Transcribe archivos de audio largos partiéndolos en fragmentos.
    La API de OpenAI tiene un límite de 25 MB por archivo.

    Args:
        ruta_audio: Ruta al archivo de audio (MP3, WAV, M4A, etc.).
        idioma: Código de idioma.
        tamano_fragmento_min: Duración de cada fragmento en minutos.

    Returns:
        Transcripción completa del audio.
    """
    ruta = Path(ruta_audio)

    # Cargar audio con pydub
    extension = ruta.suffix.lower().lstrip(".")
    if extension == "mp3":
        audio = AudioSegment.from_mp3(str(ruta))
    elif extension == "wav":
        audio = AudioSegment.from_wav(str(ruta))
    elif extension == "m4a":
        audio = AudioSegment.from_file(str(ruta), format="m4a")
    else:
        audio = AudioSegment.from_file(str(ruta))

    duracion_ms = len(audio)
    fragmento_ms = tamano_fragmento_min * 60 * 1000  # Convertir minutos a ms
    num_fragmentos = math.ceil(duracion_ms / fragmento_ms)

    print(f"Audio: {duracion_ms / 60000:.1f} min → {num_fragmentos} fragmentos")

    cliente = openai.OpenAI()
    transcripciones = []

    for i in range(num_fragmentos):
        inicio = i * fragmento_ms
        fin = min((i + 1) * fragmento_ms, duracion_ms)
        fragmento = audio[inicio:fin]

        # Guardar fragmento temporalmente
        ruta_temp = f"fragmento_temp_{i:03d}.mp3"
        fragmento.export(ruta_temp, format="mp3")

        print(f"  Transcribiendo fragmento {i+1}/{num_fragmentos}...")
        with open(ruta_temp, "rb") as f:
            resultado = cliente.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=idioma,
            )
        transcripciones.append(resultado.text)

        # Eliminar archivo temporal
        Path(ruta_temp).unlink()

    return " ".join(transcripciones)


# --- Uso ---
if __name__ == "__main__":
    # Transcripción simple (archivos < 25 MB)
    texto = transcribir_con_api_openai("entrevista.mp3", idioma="es")
    print("=== TRANSCRIPCIÓN ===")
    print(texto)

    # Obtener subtítulos SRT directamente
    subtitulos = transcribir_con_api_openai(
        "conferencia.mp3",
        idioma="es",
        formato_respuesta="srt",
    )
    with open("conferencia.srt", "w", encoding="utf-8") as f:
        f.write(subtitulos)
    print("Subtítulos SRT guardados.")

    # Archivo largo (> 25 MB)
    texto_completo = transcribir_audio_largo("reunion_larga.mp3", idioma="es")
    print(f"Transcripción completa ({len(texto_completo)} caracteres):")
    print(texto_completo[:500] + "...")
```

---

## 4. Síntesis de voz (TTS) con OpenAI

La API de OpenAI ofrece síntesis de voz de alta calidad con 6 voces distintas y dos modelos: `tts-1` (rápido) y `tts-1-hd` (mayor calidad).

```python
import openai
from pathlib import Path


# Voces disponibles
VOCES_DISPONIBLES = {
    "alloy":   "Voz neutra, clara y profesional",
    "echo":    "Voz masculina, grave y tranquila",
    "fable":   "Voz cálida con acento británico",
    "onyx":    "Voz masculina profunda y autoritaria",
    "nova":    "Voz femenina enérgica y amigable",
    "shimmer": "Voz femenina suave y expresiva",
}


def sintetizar_voz(
    texto: str,
    nombre_archivo: str = "audio_sintetizado.mp3",
    voz: str = "nova",
    modelo: str = "tts-1",
    velocidad: float = 1.0,
) -> str:
    """
    Convierte texto en audio usando la API TTS de OpenAI.

    Args:
        texto: Texto a convertir en voz (máximo ~4096 caracteres por llamada).
        nombre_archivo: Archivo de salida (se recomienda .mp3).
        voz: Una de las 6 voces disponibles (ver VOCES_DISPONIBLES).
        modelo: "tts-1" (rápido) o "tts-1-hd" (mayor calidad).
        velocidad: Factor de velocidad (0.25 = muy lento, 4.0 = muy rápido).

    Returns:
        Ruta al archivo de audio guardado.
    """
    if voz not in VOCES_DISPONIBLES:
        raise ValueError(f"Voz no válida. Opciones: {list(VOCES_DISPONIBLES.keys())}")

    cliente = openai.OpenAI()

    respuesta = cliente.audio.speech.create(
        model=modelo,
        voice=voz,
        input=texto,
        speed=velocidad,
        response_format="mp3",  # También disponible: "opus", "aac", "flac", "wav", "pcm"
    )

    ruta = Path(nombre_archivo)
    respuesta.stream_to_file(str(ruta))

    print(f"Audio guardado: {ruta.resolve()}")
    print(f"Voz: {voz} — {VOCES_DISPONIBLES[voz]}")
    return str(ruta.resolve())


def comparar_voces(texto: str, carpeta: str = "comparacion_voces") -> None:
    """
    Genera el mismo texto con todas las voces disponibles para comparar.
    """
    Path(carpeta).mkdir(exist_ok=True)

    for nombre_voz in VOCES_DISPONIBLES:
        nombre_archivo = f"{carpeta}/{nombre_voz}.mp3"
        print(f"Generando voz '{nombre_voz}'...")
        sintetizar_voz(
            texto=texto,
            nombre_archivo=nombre_archivo,
            voz=nombre_voz,
        )

    print(f"\nTodas las voces guardadas en: {carpeta}/")


def tts_texto_largo(
    texto: str,
    nombre_archivo: str = "texto_largo.mp3",
    voz: str = "nova",
    longitud_fragmento: int = 4000,
) -> str:
    """
    Sintetiza textos largos partiéndolos en fragmentos y concatenando el audio.
    La API TTS tiene un límite aproximado de 4096 caracteres por llamada.
    """
    from pydub import AudioSegment
    import tempfile
    import os

    cliente = openai.OpenAI()

    # Dividir el texto en fragmentos respetando oraciones completas
    palabras = texto.split()
    fragmentos = []
    fragmento_actual = []
    longitud_actual = 0

    for palabra in palabras:
        longitud_actual += len(palabra) + 1
        fragmento_actual.append(palabra)
        if longitud_actual >= longitud_fragmento:
            fragmentos.append(" ".join(fragmento_actual))
            fragmento_actual = []
            longitud_actual = 0

    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))

    print(f"Texto dividido en {len(fragmentos)} fragmentos.")

    # Generar audio para cada fragmento
    archivos_temp = []
    for i, fragmento in enumerate(fragmentos):
        print(f"  Sintetizando fragmento {i+1}/{len(fragmentos)}...")
        ruta_temp = f"tts_temp_{i:03d}.mp3"
        respuesta = cliente.audio.speech.create(
            model="tts-1",
            voice=voz,
            input=fragmento,
        )
        respuesta.stream_to_file(ruta_temp)
        archivos_temp.append(ruta_temp)

    # Concatenar los fragmentos con pydub
    print("Concatenando fragmentos...")
    audio_final = AudioSegment.empty()
    for ruta_temp in archivos_temp:
        segmento = AudioSegment.from_mp3(ruta_temp)
        audio_final += segmento
        Path(ruta_temp).unlink()  # Eliminar temporal

    audio_final.export(nombre_archivo, format="mp3")
    print(f"Audio completo guardado: {Path(nombre_archivo).resolve()}")
    return nombre_archivo


# --- Uso ---
if __name__ == "__main__":
    # Síntesis básica
    sintetizar_voz(
        texto="Bienvenido al tutorial de síntesis de voz con inteligencia artificial.",
        nombre_archivo="bienvenida.mp3",
        voz="nova",
        modelo="tts-1-hd",
    )

    # Comparar todas las voces
    comparar_voces(
        texto="La inteligencia artificial está transformando la forma en que interactuamos con la tecnología.",
    )
```

---

## 5. Pipeline completo voz → texto → IA → voz

Este pipeline captura audio del micrófono, lo transcribe con Whisper, envía el texto a Claude para obtener una respuesta, y sintetiza esa respuesta en voz.

```bash
pip install pyaudio openai anthropic pydub
# En macOS: brew install portaudio
```

```python
import anthropic
import openai
import pyaudio
import wave
import tempfile
import os
from pathlib import Path
import subprocess
import sys


# Configuración de grabación de audio
TASA_MUESTREO = 16000   # Hz (óptimo para Whisper)
CANALES = 1              # Mono
FORMATO = pyaudio.paInt16
CHUNK = 1024             # Frames por buffer


def grabar_audio(duracion_segundos: int = 5, nombre_archivo: str = "grabacion.wav") -> str:
    """
    Graba audio del micrófono durante los segundos indicados.

    Args:
        duracion_segundos: Duración de la grabación.
        nombre_archivo: Archivo WAV de salida.

    Returns:
        Ruta al archivo de audio grabado.
    """
    audio = pyaudio.PyAudio()

    print(f"Grabando {duracion_segundos} segundos... Habla ahora.")

    stream = audio.open(
        format=FORMATO,
        channels=CANALES,
        rate=TASA_MUESTREO,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    num_chunks = int(TASA_MUESTREO / CHUNK * duracion_segundos)

    for _ in range(num_chunks):
        datos = stream.read(CHUNK)
        frames.append(datos)

    print("Grabación finalizada.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Guardar como WAV
    with wave.open(nombre_archivo, "wb") as archivo_wav:
        archivo_wav.setnchannels(CANALES)
        archivo_wav.setsampwidth(audio.get_sample_size(FORMATO))
        archivo_wav.setframerate(TASA_MUESTREO)
        archivo_wav.writeframes(b"".join(frames))

    return nombre_archivo


def transcribir_audio(ruta_audio: str) -> str:
    """Transcribe audio con la API de OpenAI Whisper."""
    cliente = openai.OpenAI()
    with open(ruta_audio, "rb") as f:
        resultado = cliente.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="es",
        )
    return resultado.text


def consultar_ia(pregunta: str, historial: list) -> tuple[str, list]:
    """
    Envía la pregunta a Claude y devuelve la respuesta y el historial actualizado.

    Args:
        pregunta: Texto transcrito del usuario.
        historial: Lista de mensajes anteriores para mantener contexto.

    Returns:
        Tupla (respuesta, historial_actualizado).
    """
    cliente = anthropic.Anthropic()

    # Añadir la nueva pregunta al historial
    historial.append({"role": "user", "content": pregunta})

    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=(
            "Eres un asistente de voz útil y conciso. "
            "Responde en español. "
            "Mantén las respuestas breves (2-4 oraciones máximo) "
            "ya que serán convertidas a audio."
        ),
        messages=historial,
    )

    respuesta = mensaje.content[0].text
    historial.append({"role": "assistant", "content": respuesta})

    return respuesta, historial


def sintetizar_y_reproducir(texto: str, voz: str = "nova") -> None:
    """
    Sintetiza el texto en voz y lo reproduce inmediatamente.
    """
    cliente = openai.OpenAI()

    # Usar archivo temporal para el audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        ruta_temp = f.name

    respuesta = cliente.audio.speech.create(
        model="tts-1",
        voice=voz,
        input=texto,
    )
    respuesta.stream_to_file(ruta_temp)

    # Reproducir el audio (usando el reproductor del sistema)
    if sys.platform == "darwin":
        subprocess.run(["afplay", ruta_temp], check=True)
    elif sys.platform == "linux":
        subprocess.run(["mpg123", ruta_temp], check=True)
    elif sys.platform == "win32":
        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{ruta_temp}').PlaySync()"], check=True)

    # Limpiar temporal
    Path(ruta_temp).unlink(missing_ok=True)


def asistente_voz_interactivo(duracion_escucha: int = 5, turnos: int = 5) -> None:
    """
    Ejecuta el pipeline completo de asistente de voz durante N turnos.

    Args:
        duracion_escucha: Segundos que el sistema escucha en cada turno.
        turnos: Número de intercambios de conversación.
    """
    historial = []
    archivo_grabacion = "turno_actual.wav"

    print("=== ASISTENTE DE VOZ ACTIVADO ===")
    print(f"Tendremos {turnos} turnos de conversación.")
    print("Habla cuando veas 'Grabando...'.\n")

    for turno in range(1, turnos + 1):
        print(f"--- Turno {turno}/{turnos} ---")

        # Paso 1: Grabar audio del usuario
        grabar_audio(duracion_segundos=duracion_escucha, nombre_archivo=archivo_grabacion)

        # Paso 2: Transcribir con Whisper
        print("Transcribiendo...")
        texto_usuario = transcribir_audio(archivo_grabacion)
        print(f"  Tú dijiste: {texto_usuario}")

        if not texto_usuario.strip():
            print("  No se detectó voz. Saltando turno.")
            continue

        # Paso 3: Consultar a Claude
        print("Pensando...")
        respuesta_ia, historial = consultar_ia(texto_usuario, historial)
        print(f"  Asistente: {respuesta_ia}")

        # Paso 4: Sintetizar y reproducir respuesta
        print("Hablando...")
        sintetizar_y_reproducir(respuesta_ia)
        print()

    # Limpiar
    Path(archivo_grabacion).unlink(missing_ok=True)
    print("=== CONVERSACIÓN FINALIZADA ===")


# --- Uso ---
if __name__ == "__main__":
    # Iniciar el asistente de voz con 5 turnos de 5 segundos cada uno
    asistente_voz_interactivo(duracion_escucha=5, turnos=5)
```

---

## 6. Caso práctico — asistente de voz para reuniones

Este caso práctico procesa una grabación de reunión completa: la transcribe, la resume con Claude y extrae los puntos de acción.

```python
import anthropic
import openai
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
import json
import math
from pydub import AudioSegment


# Modelos de datos para la salida estructurada
class PuntoAccion(BaseModel):
    responsable: Optional[str] = Field(None, description="Persona responsable")
    tarea: str = Field(..., description="Descripción de la tarea")
    plazo: Optional[str] = Field(None, description="Fecha límite o plazo estimado")
    prioridad: str = Field("media", description="alta, media o baja")


class ResumenReunion(BaseModel):
    titulo: str = Field(..., description="Título descriptivo de la reunión")
    fecha_aproximada: Optional[str] = Field(None, description="Fecha si se menciona en la transcripción")
    participantes: list[str] = Field(default_factory=list, description="Nombres mencionados")
    resumen_ejecutivo: str = Field(..., description="Resumen de 3-5 oraciones")
    temas_tratados: list[str] = Field(default_factory=list, description="Lista de temas discutidos")
    decisiones_tomadas: list[str] = Field(default_factory=list, description="Decisiones acordadas")
    puntos_accion: list[PuntoAccion] = Field(default_factory=list, description="Tareas asignadas")
    proxima_reunion: Optional[str] = Field(None, description="Fecha de próxima reunión si se menciona")


def transcribir_reunion(ruta_audio: str, idioma: str = "es") -> str:
    """
    Transcribe una grabación de reunión, manejando archivos largos automáticamente.
    """
    cliente = openai.OpenAI()
    ruta = Path(ruta_audio)

    # Verificar tamaño del archivo
    tamano_mb = ruta.stat().st_size / (1024 * 1024)
    print(f"Archivo: {ruta.name} ({tamano_mb:.1f} MB)")

    if tamano_mb <= 24:
        # Archivo pequeño: transcribir directamente
        print("Transcribiendo en una sola llamada...")
        with open(ruta, "rb") as f:
            resultado = cliente.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=idioma,
                response_format="verbose_json",
            )
        return resultado.text
    else:
        # Archivo grande: dividir en fragmentos de 10 minutos
        print("Archivo grande, dividiendo en fragmentos...")
        extension = ruta.suffix.lower().lstrip(".")
        audio = AudioSegment.from_file(str(ruta))
        fragmento_ms = 10 * 60 * 1000  # 10 minutos
        num_fragmentos = math.ceil(len(audio) / fragmento_ms)

        transcripciones = []
        for i in range(num_fragmentos):
            inicio = i * fragmento_ms
            fin = min((i + 1) * fragmento_ms, len(audio))
            fragmento = audio[inicio:fin]

            ruta_temp = f"reunion_fragmento_{i:03d}.mp3"
            fragmento.export(ruta_temp, format="mp3")

            print(f"  Transcribiendo fragmento {i+1}/{num_fragmentos}...")
            with open(ruta_temp, "rb") as f:
                resultado = cliente.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=idioma,
                )
            transcripciones.append(resultado.text)
            Path(ruta_temp).unlink()

        return " ".join(transcripciones)


def analizar_reunion_con_claude(transcripcion: str) -> ResumenReunion:
    """
    Usa Claude para analizar la transcripción y extraer información estructurada.
    """
    cliente = anthropic.Anthropic()

    # Si la transcripción es muy larga, resumirla antes de estructurar
    if len(transcripcion) > 15000:
        print("Transcripción larga, procesando en dos pasadas...")
        # Primera pasada: resumir para reducir tokens
        resumen_previo = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Resume esta transcripción de reunión, manteniendo todos los detalles "
                        f"importantes, nombres, decisiones y tareas asignadas:\n\n{transcripcion[:15000]}"
                    ),
                }
            ],
        )
        texto_a_analizar = resumen_previo.content[0].text
    else:
        texto_a_analizar = transcripcion

    esquema_json = json.dumps(ResumenReunion.model_json_schema(), ensure_ascii=False, indent=2)

    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=(
            "Eres un asistente especializado en análisis de reuniones de trabajo. "
            "Analiza la transcripción y extrae la información estructurada. "
            f"Responde ÚNICAMENTE con un JSON válido que siga este esquema:\n{esquema_json}\n"
            "No incluyas texto fuera del JSON."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Analiza esta transcripción de reunión:\n\n{texto_a_analizar}",
            }
        ],
    )

    texto_json = mensaje.content[0].text.strip()
    if texto_json.startswith("```"):
        lineas = texto_json.split("\n")
        texto_json = "\n".join(lineas[1:-1])

    datos = json.loads(texto_json)
    return ResumenReunion(**datos)


def generar_informe_reunion(resumen: ResumenReunion) -> str:
    """
    Genera un informe de texto formateado a partir del resumen estructurado.
    """
    lineas = [
        f"# {resumen.titulo}",
        "",
    ]

    if resumen.fecha_aproximada:
        lineas.append(f"**Fecha:** {resumen.fecha_aproximada}")

    if resumen.participantes:
        lineas.append(f"**Participantes:** {', '.join(resumen.participantes)}")

    lineas.extend(["", "## Resumen ejecutivo", "", resumen.resumen_ejecutivo, ""])

    if resumen.temas_tratados:
        lineas.append("## Temas tratados")
        lineas.append("")
        for tema in resumen.temas_tratados:
            lineas.append(f"- {tema}")
        lineas.append("")

    if resumen.decisiones_tomadas:
        lineas.append("## Decisiones tomadas")
        lineas.append("")
        for decision in resumen.decisiones_tomadas:
            lineas.append(f"- {decision}")
        lineas.append("")

    if resumen.puntos_accion:
        lineas.append("## Puntos de acción")
        lineas.append("")
        for i, accion in enumerate(resumen.puntos_accion, 1):
            responsable = f" — **{accion.responsable}**" if accion.responsable else ""
            plazo = f" (Plazo: {accion.plazo})" if accion.plazo else ""
            lineas.append(f"{i}. [{accion.prioridad.upper()}]{responsable}: {accion.tarea}{plazo}")
        lineas.append("")

    if resumen.proxima_reunion:
        lineas.append(f"## Próxima reunión")
        lineas.append("")
        lineas.append(resumen.proxima_reunion)

    return "\n".join(lineas)


def procesar_reunion_completa(
    ruta_audio: str,
    idioma: str = "es",
    guardar_transcripcion: bool = True,
) -> dict:
    """
    Pipeline completo: audio → transcripción → análisis → informe.

    Args:
        ruta_audio: Ruta al archivo de audio de la reunión.
        idioma: Idioma hablado en la reunión.
        guardar_transcripcion: Si guardar la transcripción en disco.

    Returns:
        Diccionario con transcripcion, resumen_estructurado e informe.
    """
    nombre_base = Path(ruta_audio).stem

    print("\n=== PROCESANDO REUNIÓN ===\n")

    # Paso 1: Transcribir
    print("1/3 Transcribiendo audio...")
    transcripcion = transcribir_reunion(ruta_audio, idioma)

    if guardar_transcripcion:
        ruta_transcripcion = f"{nombre_base}_transcripcion.txt"
        Path(ruta_transcripcion).write_text(transcripcion, encoding="utf-8")
        print(f"    Transcripción guardada: {ruta_transcripcion}")

    print(f"    Palabras transcritas: {len(transcripcion.split())}")

    # Paso 2: Analizar con Claude
    print("\n2/3 Analizando con Claude...")
    resumen = analizar_reunion_con_claude(transcripcion)
    print(f"    Puntos de acción detectados: {len(resumen.puntos_accion)}")

    # Paso 3: Generar informe
    print("\n3/3 Generando informe...")
    informe = generar_informe_reunion(resumen)

    ruta_informe = f"{nombre_base}_informe.md"
    Path(ruta_informe).write_text(informe, encoding="utf-8")
    print(f"    Informe guardado: {ruta_informe}")

    print("\n=== PROCESAMIENTO COMPLETADO ===\n")
    print(informe)

    return {
        "transcripcion": transcripcion,
        "resumen": resumen.model_dump(),
        "informe": informe,
        "ruta_informe": ruta_informe,
    }


# --- Uso ---
if __name__ == "__main__":
    resultado = procesar_reunion_completa(
        ruta_audio="reunion_equipo.mp3",
        idioma="es",
        guardar_transcripcion=True,
    )

    print(f"\nInforme disponible en: {resultado['ruta_informe']}")
    print(f"Puntos de acción: {len(resultado['resumen']['puntos_accion'])}")
```

---

## 7. Extensiones sugeridas

- **Transcripción en tiempo real con streaming**: usa la API de Deepgram o AssemblyAI para transcribir mientras el usuario habla, sin esperar a que termine la grabación.
- **Identificación de hablantes (diarización)**: con la API de AssemblyAI puedes saber cuándo habla cada persona en una reunión con múltiples participantes.
- **Bot de reuniones para Google Meet o Zoom**: conecta el pipeline al audio de una videollamada usando una extensión del navegador o la API de Zoom.
- **Notificaciones automáticas**: tras procesar la reunión, envía el informe y los puntos de acción por email (SendGrid) o Slack (webhook).
- **Búsqueda semántica en transcripciones**: indexa múltiples transcripciones con embeddings para buscar "¿en qué reunión se decidió X?" con búsqueda vectorial.
- **Clonación de voz con ElevenLabs**: graba tu voz y crea un clon para que el asistente hable con tu propia voz.
