# Generación Musical y Audio Creativo con IA

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/voz-audio/04-generacion-audio-creativo.ipynb)

Más allá de la transcripción y TTS, la IA permite generar música, efectos de sonido y audio creativo completo. Este tutorial cubre los principales servicios y modelos open-source, con foco en casos de uso empresariales.

---

## 1. Panorama de herramientas

| Herramienta | Tipo | Mejor para |
|-------------|------|-----------|
| Suno | API/Web | Canciones completas (letra + música) |
| Udio | Web | Producción musical de alta calidad |
| MusicGen (Meta) | Open-source | Generación local, sin API |
| AudioCraft (Meta) | Open-source | Audio general + efectos |
| ElevenLabs SFX | API | Efectos de sonido a partir de texto |
| Stable Audio | API | Stems, música sin letra |

---

## 2. MusicGen local (Meta, open-source)

MusicGen no requiere API ni coste por uso. Corre en local con GPU (o CPU lento):

```python
# pip install transformers torch scipy

from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile
import numpy as np
import torch

class GeneradorMusica:
    """Genera música a partir de descripciones de texto con MusicGen."""
    
    def __init__(self, modelo: str = "facebook/musicgen-small"):
        # Opciones: musicgen-small (300M), musicgen-medium (1.5B), musicgen-large (3.3B)
        self.processor = AutoProcessor.from_pretrained(modelo)
        self.model = MusicgenForConditionalGeneration.from_pretrained(modelo)
        self.model.eval()
        
    def generar(
        self,
        descripcion: str,
        duracion_segundos: int = 10,
        ruta_salida: str = "musica.wav",
    ) -> str:
        """
        Genera un clip de audio a partir de una descripción.
        Ejemplo: 'melodic electronic music with piano, upbeat, 120bpm'
        """
        inputs = self.processor(
            text=[descripcion],
            padding=True,
            return_tensors="pt",
        )
        
        tokens_por_segundo = 50  # MusicGen usa 50 tokens/seg
        max_tokens = duracion_segundos * tokens_por_segundo
        
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
            )
        
        # Guardar como WAV
        sample_rate = self.model.config.audio_encoder.sampling_rate
        audio_np = audio_values[0, 0].numpy()
        
        scipy.io.wavfile.write(
            ruta_salida,
            rate=sample_rate,
            data=(audio_np * 32767).astype(np.int16),
        )
        
        print(f"Audio generado: {duracion_segundos}s → {ruta_salida}")
        return ruta_salida

# Uso
generador = GeneradorMusica("facebook/musicgen-small")
generador.generar(
    "upbeat corporate background music, piano and strings, professional",
    duracion_segundos=15,
    ruta_salida="fondo_corporativo.wav",
)
```

---

## 3. Efectos de sonido con ElevenLabs SFX

```python
from elevenlabs.client import ElevenLabs

el_client = ElevenLabs(api_key="tu-api-key")

def generar_efecto_sonido(
    descripcion: str,
    duracion_segundos: float = 2.0,
    ruta_salida: str = "efecto.mp3",
) -> str:
    """
    Genera un efecto de sonido a partir de una descripción.
    Ejemplos: 'cash register ding', 'notification bell', 'error buzz'
    """
    resultado = el_client.text_to_sound_effects.convert(
        text=descripcion,
        duration_seconds=duracion_segundos,
        prompt_influence=0.3,  # 0-1, más alto = más literal
    )
    
    with open(ruta_salida, "wb") as f:
        for chunk in resultado:
            f.write(chunk)
    
    return ruta_salida

# Ejemplos
efectos = [
    ("notification bell, soft, professional", 1.5, "notificacion.mp3"),
    ("error buzz, short, digital", 0.5, "error.mp3"),
    ("success chime, warm, pleasant", 1.0, "exito.mp3"),
]

for descripcion, duracion, salida in efectos:
    generar_efecto_sonido(descripcion, duracion, salida)
    print(f"✅ {salida}")
```

---

## 4. Pipeline Claude → MusicGen: música para el contenido

Claude puede generar el prompt musical a partir del contexto del proyecto:

```python
import anthropic

anthropic_client = anthropic.Anthropic()

def generar_prompt_musical(descripcion_proyecto: str) -> str:
    """Usa Claude para generar un prompt optimizado para MusicGen."""
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": f"""Genera un prompt corto (máximo 20 palabras) para MusicGen
que cree música de fondo apropiada para:

{descripcion_proyecto}

El prompt debe describir: género, instrumentos, tempo/mood y estilo.
Solo el prompt, sin explicación.""",
        }],
    )
    return response.content[0].text.strip()

def musica_para_proyecto(descripcion: str, duracion: int = 20) -> str:
    """Pipeline completo: descripción → Claude → prompt → MusicGen → audio."""
    prompt = generar_prompt_musical(descripcion)
    print(f"Prompt musical: {prompt}")
    
    generador = GeneradorMusica()
    return generador.generar(prompt, duracion)

# Ejemplo
musica_para_proyecto(
    "Presentación corporativa de producto tecnológico innovador para inversores",
    duracion=20,
)
```

---

## 5. Casos de uso empresariales

### 5.1 Podcast automatizado

```python
def generar_podcast(tema: str, duracion_min: int = 5) -> dict:
    """
    Genera un episodio de podcast completo:
    guión → narración TTS → música de fondo → mezcla.
    """
    import openai
    openai_client = openai.OpenAI()
    
    # 1. Guión con Claude
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Escribe un guión de podcast de {duracion_min} minutos sobre: {tema}.
Estructura: introducción (30s) → 3 puntos principales (1min cada uno) → conclusión (30s).
Tono: profesional pero accesible. Sin mencionar que es un guión.""",
        }],
    )
    guion = response.content[0].text
    
    # 2. Narración TTS
    audio_narracion = openai_client.audio.speech.create(
        model="tts-1-hd",
        voice="onyx",  # voz grave para podcast
        input=guion,
    )
    with open("narracion.mp3", "wb") as f:
        f.write(audio_narracion.content)
    
    # 3. Música de fondo
    prompt_musica = generar_prompt_musical(f"podcast sobre {tema}")
    generador = GeneradorMusica()
    generador.generar(prompt_musica, duracion_segundos=duracion_min * 60, ruta_salida="musica_fondo.wav")
    
    return {
        "guion": guion,
        "narracion": "narracion.mp3",
        "musica": "musica_fondo.wav",
        "nota": "Mezcla con pydub: AudioSegment.from_mp3('narracion.mp3').overlay(musica, gain_during_overlay=-15)",
    }
```

### 5.2 Audiolibro desde texto

```python
def texto_a_audiolibro(
    texto: str,
    capitulos: bool = True,
    voz: str = "fable",
    directorio: str = "audiolibro",
) -> list[str]:
    """Convierte texto largo en audiolibro con capítulos."""
    import os, re
    openai_client = openai.OpenAI()
    os.makedirs(directorio, exist_ok=True)
    
    # Dividir por capítulos o párrafos
    if capitulos:
        partes = re.split(r'\n#{1,3}\s', texto)
    else:
        # Dividir por párrafos de ~500 chars para TTS
        partes = [texto[i:i+500] for i in range(0, len(texto), 500)]
    
    archivos = []
    for i, parte in enumerate(partes):
        if not parte.strip():
            continue
        ruta = f"{directorio}/capitulo_{i+1:03d}.mp3"
        audio = openai_client.audio.speech.create(
            model="tts-1-hd", voice=voz, input=parte.strip()
        )
        with open(ruta, "wb") as f:
            f.write(audio.content)
        archivos.append(ruta)
        print(f"  Capítulo {i+1} → {ruta}")
    
    return archivos
```

---

## Limitaciones a tener en cuenta

| Aspecto | Detalle |
|---------|---------|
| Derechos musicales | MusicGen (Meta) permite uso comercial bajo CC-BY-NC. Suno/Udio tienen licencias propias |
| Calidad MusicGen small | Suficiente para fondos, no para producción musical profesional |
| Coherencia larga duración | MusicGen pierde coherencia musical pasados ~30 segundos |
| Idioma en TTS | Especifica siempre el idioma; mezclar idiomas da resultados impredecibles |
