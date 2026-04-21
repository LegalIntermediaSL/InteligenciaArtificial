# Agentes de Voz en Tiempo Real

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/voz-audio/03-agentes-voz-realtime.ipynb)

Los agentes de voz en tiempo real combinan STT, LLM y TTS en un pipeline con latencia total inferior a 500ms. Este tutorial cubre la arquitectura completa, detección de actividad de voz (VAD) y la OpenAI Realtime API.

---

## 1. Arquitectura de un agente de voz

```
Micrófono → VAD → Buffer de audio → Whisper STT → Claude/GPT → TTS → Altavoz
              ↑                          ↓              ↓
           Silencio               Texto del usuario   Texto respuesta
           detectado
```

**Latencias objetivo**:
- VAD → detección fin de frase: ~100ms
- STT (Whisper turbo local): ~200ms para 5s de audio
- LLM (primera respuesta): ~300ms (streaming)
- TTS (primer chunk de audio): ~200ms
- **Total**: ~800ms — perceptible pero aceptable para asistentes

---

## 2. VAD: detección de actividad de voz

```python
# pip install webrtcvad pyaudio numpy

import webrtcvad
import pyaudio
import numpy as np
from collections import deque

class DetectorVAD:
    """Detecta cuando el usuario empieza y termina de hablar."""
    
    def __init__(
        self,
        agresividad: int = 2,       # 0-3, más alto = más estricto
        sample_rate: int = 16000,
        frame_ms: int = 30,          # frames de 10, 20 o 30ms
        silencio_ms: int = 800,      # ms de silencio para considerar fin de frase
    ):
        self.vad = webrtcvad.Vad(agresividad)
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000) * 2  # bytes
        self.frames_silencio = silencio_ms // frame_ms
        
    def capturar_frase(self) -> bytes:
        """Captura audio hasta detectar fin de frase. Retorna los bytes de audio."""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size // 2,
        )
        
        print("🎤 Escuchando...")
        frames_capturados = []
        frames_silencio_contados = 0
        hablando = False
        
        try:
            while True:
                frame = stream.read(self.frame_size // 2, exception_on_overflow=False)
                es_voz = self.vad.is_speech(frame, self.sample_rate)
                
                if es_voz:
                    if not hablando:
                        print("🔴 Grabando...")
                        hablando = True
                    frames_capturados.append(frame)
                    frames_silencio_contados = 0
                elif hablando:
                    frames_capturados.append(frame)
                    frames_silencio_contados += 1
                    if frames_silencio_contados >= self.frames_silencio:
                        print("⏹ Fin de frase detectado")
                        break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        return b"".join(frames_capturados)
```

---

## 3. Pipeline STT → Claude → TTS

```python
import anthropic
import openai
import io
import wave

anthropic_client = anthropic.Anthropic()
openai_client = openai.OpenAI()

def bytes_a_wav(audio_bytes: bytes, sample_rate: int = 16000) -> io.BytesIO:
    """Convierte bytes de audio PCM a formato WAV en memoria."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_bytes)
    buffer.seek(0)
    buffer.name = "audio.wav"
    return buffer

def pipeline_voz_completo(
    audio_bytes: bytes,
    historial: list[dict],
    voz_tts: str = "nova",
) -> tuple[str, str, bytes]:
    """
    STT → Claude → TTS en un pipeline.
    Retorna (texto_usuario, texto_respuesta, audio_respuesta).
    """
    # 1. STT: audio → texto
    wav_buffer = bytes_a_wav(audio_bytes)
    transcripcion = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_buffer,
        language="es",
    )
    texto_usuario = transcripcion.text
    print(f"👤 Usuario: {texto_usuario}")
    
    # 2. LLM: texto → respuesta
    historial.append({"role": "user", "content": texto_usuario})
    respuesta = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=(
            "Eres un asistente vocal amigable. Responde en 1-2 frases cortas, "
            "sin markdown ni listas. Habla de forma natural y directa."
        ),
        messages=historial,
    )
    texto_respuesta = respuesta.content[0].text
    historial.append({"role": "assistant", "content": texto_respuesta})
    print(f"🤖 Asistente: {texto_respuesta}")
    
    # 3. TTS: respuesta → audio
    audio_respuesta = openai_client.audio.speech.create(
        model="tts-1",
        voice=voz_tts,
        input=texto_respuesta,
        response_format="wav",
    )
    
    return texto_usuario, texto_respuesta, audio_respuesta.content

def agente_voz_interactivo():
    """Bucle principal del agente de voz."""
    detector = DetectorVAD()
    historial = []
    
    print("🎙️ Agente de voz iniciado. Di 'salir' para terminar.")
    
    while True:
        # Capturar audio
        audio_bytes = detector.capturar_frase()
        
        # Pipeline completo
        texto_usuario, texto_respuesta, audio_respuesta = pipeline_voz_completo(
            audio_bytes, historial
        )
        
        # Reproducir respuesta
        reproducir_audio(audio_respuesta)
        
        # Condición de salida
        if "salir" in texto_usuario.lower() or "adiós" in texto_usuario.lower():
            print("👋 Hasta luego")
            break

def reproducir_audio(audio_bytes: bytes):
    """Reproduce bytes de audio WAV."""
    import tempfile, os, subprocess
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    # macOS: afplay, Linux: aplay, Windows: usa pygame
    if os.path.exists("/usr/bin/afplay"):
        subprocess.run(["afplay", tmp_path], check=True)
    else:
        subprocess.run(["aplay", tmp_path], check=True)
    os.unlink(tmp_path)
```

---

## 4. OpenAI Realtime API

La Realtime API de OpenAI permite latencias de ~300ms con conexión WebSocket persistente:

```python
# pip install websockets

import asyncio
import websockets
import json
import base64

async def agente_realtime():
    """
    Agente de voz usando la OpenAI Realtime API.
    Requiere: OPENAI_API_KEY con acceso a la Realtime API.
    """
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {openai_client.api_key}",
        "OpenAI-Beta": "realtime=v1",
    }
    
    async with websockets.connect(uri, extra_headers=headers) as ws:
        # Configurar sesión
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "Eres un asistente útil. Responde brevemente en español.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",  # VAD gestionado por OpenAI
                    "threshold": 0.5,
                    "silence_duration_ms": 600,
                },
            },
        }))
        
        print("✅ Sesión Realtime iniciada. Habla ahora...")
        
        # Capturar y enviar audio en tiempo real
        async def enviar_audio():
            audio_p = pyaudio.PyAudio()
            stream = audio_p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,  # Realtime API requiere 24kHz
                input=True,
                frames_per_buffer=1024,
            )
            try:
                while True:
                    frame = stream.read(1024, exception_on_overflow=False)
                    audio_b64 = base64.b64encode(frame).decode()
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }))
                    await asyncio.sleep(0)
            finally:
                stream.stop_stream()
                stream.close()
                audio_p.terminate()
        
        # Recibir y reproducir respuestas
        async def recibir_audio():
            audio_buffer = bytearray()
            async for mensaje in ws:
                evento = json.loads(mensaje)
                
                if evento["type"] == "response.audio.delta":
                    chunk = base64.b64decode(evento["delta"])
                    audio_buffer.extend(chunk)
                
                elif evento["type"] == "response.audio.done":
                    if audio_buffer:
                        reproducir_audio(bytes(audio_buffer))
                        audio_buffer.clear()
                
                elif evento["type"] == "response.text.delta":
                    print(evento.get("delta", ""), end="", flush=True)
        
        # Ejecutar en paralelo
        await asyncio.gather(enviar_audio(), recibir_audio())

# asyncio.run(agente_realtime())
```

---

## 5. Consideraciones de producción

| Aspecto | Recomendación |
|---------|---------------|
| Cancelación de eco | Usa `pyaudio` con `echoCancellation=True` o hardware dedicado |
| Reducción de ruido | `noisereduce` library antes de enviar a STT |
| Latencia objetivo | < 800ms total para experiencia natural |
| Reconexiones | Implementa backoff exponencial para WebSockets |
| Logs de audio | Nunca guardes audio sin consentimiento explícito |
| Idiomas | Especifica siempre `language=` en Whisper para evitar errores |
