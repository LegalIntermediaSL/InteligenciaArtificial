# Síntesis de Voz: ElevenLabs, OpenAI TTS y Streaming

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/voz-audio/02-sintesis-voz.ipynb)

Convertir texto a voz de calidad humana es el complemento natural de la transcripción. Este tutorial cubre los dos principales proveedores, streaming de audio y un pipeline texto→Claude→voz listo para producción.

---

## 1. Comparativa de proveedores TTS

| | OpenAI TTS | ElevenLabs |
|---|---|---|
| Modelos | tts-1, tts-1-hd | eleven_multilingual_v2, turbo_v2_5 |
| Voces predefinidas | 6 | 3000+ (biblioteca pública) |
| Clonación de voz | No | Sí (con muestra de 1 min) |
| Streaming | Sí | Sí |
| Precio | $15/1M chars | Desde $5/mes (10k chars) |
| Idiomas | 57 | 32 |
| Latencia | ~500ms | ~300ms (turbo) |

---

## 2. OpenAI TTS

```python
from pathlib import Path
import openai

client = openai.OpenAI()

def texto_a_voz_openai(
    texto: str,
    voz: str = "alloy",       # alloy, echo, fable, onyx, nova, shimmer
    modelo: str = "tts-1-hd", # tts-1 (rápido) o tts-1-hd (calidad)
    formato: str = "mp3",
    ruta_salida: str = "audio.mp3",
) -> str:
    """Convierte texto a voz con OpenAI TTS."""
    respuesta = client.audio.speech.create(
        model=modelo,
        voice=voz,
        input=texto,
        response_format=formato,
    )
    respuesta.stream_to_file(ruta_salida)
    return ruta_salida

# Streaming en tiempo real (para latencia baja)
def tts_streaming_openai(texto: str, voz: str = "nova") -> bytes:
    """Genera audio en streaming para baja latencia."""
    import io
    buffer = io.BytesIO()
    
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voz,
        input=texto,
    ) as respuesta:
        for chunk in respuesta.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
    
    return buffer.getvalue()
```

---

## 3. ElevenLabs

```python
# pip install elevenlabs
from elevenlabs.client import ElevenLabs
from elevenlabs import save, stream

el_client = ElevenLabs(api_key="tu-api-key")

def texto_a_voz_elevenlabs(
    texto: str,
    voz_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel (voz popular)
    modelo: str = "eleven_multilingual_v2",
    ruta_salida: str = "audio.mp3",
) -> str:
    """Convierte texto a voz con ElevenLabs."""
    audio = el_client.generate(
        text=texto,
        voice=voz_id,
        model=modelo,
    )
    save(audio, ruta_salida)
    return ruta_salida

def buscar_voces_elevenlabs(idioma: str = "es") -> list[dict]:
    """Lista voces disponibles filtradas por idioma."""
    voces = el_client.voices.get_all()
    return [
        {
            "id": v.voice_id,
            "nombre": v.name,
            "genero": v.labels.get("gender", "?"),
            "acento": v.labels.get("accent", "?"),
        }
        for v in voces.voices
        if idioma in (v.labels.get("language", "") or "").lower()
    ]

# Streaming para reproducción inmediata
def tts_streaming_elevenlabs(texto: str, voz_id: str = "21m00Tcm4TlvDq8ikWAM"):
    """Genera y reproduce audio en streaming."""
    audio_stream = el_client.generate(
        text=texto,
        voice=voz_id,
        model="eleven_turbo_v2_5",  # modelo más rápido de ElevenLabs
        stream=True,
    )
    stream(audio_stream)
```

---

## 4. Clonación de voz con ElevenLabs

Con una muestra de audio de 1 minuto se puede clonar una voz:

```python
def clonar_voz(nombre: str, ruta_muestra: str, descripcion: str = "") -> str:
    """
    Clona una voz a partir de una muestra de audio.
    La muestra debe ser clara, sin ruido de fondo, mínimo 1 minuto.
    Retorna el voice_id de la voz clonada.
    """
    with open(ruta_muestra, "rb") as f:
        voz = el_client.clone(
            name=nombre,
            description=descripcion,
            files=[f],
        )
    return voz.voice_id

# Uso
# voz_id = clonar_voz("Mi voz", "muestra.mp3", "Voz para asistente de empresa")
# texto_a_voz_elevenlabs("Hola, soy tu asistente de IA", voz_id=voz_id)
```

---

## 5. Pipeline texto → Claude → voz

El caso de uso más común: Claude genera el texto y TTS lo convierte en audio.

```python
import anthropic

anthropic_client = anthropic.Anthropic()

def asistente_vocal(
    pregunta: str,
    historial: list[dict] | None = None,
    voz: str = "nova",
    ruta_salida: str = "respuesta.mp3",
) -> tuple[str, str]:
    """
    Pipeline completo: pregunta → Claude → texto → OpenAI TTS → audio.
    Retorna (texto_respuesta, ruta_audio).
    """
    mensajes = historial or []
    mensajes.append({"role": "user", "content": pregunta})
    
    # 1. Generar respuesta con Claude
    respuesta = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=(
            "Eres un asistente vocal. Responde de forma concisa y natural, "
            "como si estuvieras hablando (evita listas, bullets y markdown). "
            "Máximo 3 frases."
        ),
        messages=mensajes,
    )
    texto = respuesta.content[0].text
    
    # 2. Convertir a audio
    openai_client = openai.OpenAI()
    audio = openai_client.audio.speech.create(
        model="tts-1",
        voice=voz,
        input=texto,
    )
    audio.stream_to_file(ruta_salida)
    
    # Actualizar historial
    mensajes.append({"role": "assistant", "content": texto})
    
    return texto, ruta_salida

# Uso
texto, audio = asistente_vocal("¿Cuáles son las ventajas del prompt caching?")
print(f"Texto: {texto}")
print(f"Audio guardado en: {audio}")
```

---

## 6. SSML para control fino de la voz

OpenAI TTS no admite SSML nativamente, pero se pueden usar técnicas de texto para controlar el ritmo:

```python
def preparar_texto_para_tts(texto: str) -> str:
    """
    Prepara texto para TTS añadiendo pausas y énfasis naturales.
    Útil para contenido generado por IA que necesita sonar natural.
    """
    import re
    
    # Pausas después de puntos
    texto = re.sub(r'\.(\s)', r'. \1', texto)
    
    # Pausas largas después de puntos de lista
    texto = re.sub(r'(\d+\.\s)', r'\1 ', texto)
    
    # Eliminar markdown que no suena bien en audio
    texto = re.sub(r'\*\*(.*?)\*\*', r'\1', texto)  # negrita
    texto = re.sub(r'\*(.*?)\*', r'\1', texto)        # cursiva
    texto = re.sub(r'`(.*?)`', r'\1', texto)          # código
    texto = re.sub(r'#{1,6}\s', '', texto)            # títulos
    
    return texto.strip()
```

---

## Cuándo usar cada proveedor

| Necesitas | Usa |
|-----------|-----|
| Voz estándar, bajo coste | OpenAI TTS (tts-1) |
| Máxima calidad | OpenAI TTS (tts-1-hd) |
| Voz personalizada/marca | ElevenLabs (clonar voz) |
| Baja latencia en tiempo real | ElevenLabs Turbo v2.5 |
| Producción con privacidad | Coqui TTS (open-source local) |
