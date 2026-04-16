# 10 — Gestión del Contexto y Context Window Management

> **Bloque:** LLMs · **Nivel:** Intermedio · **Tiempo estimado:** 40 min

---

## Índice

1. [La ventana de contexto: qué es y por qué importa](#1-la-ventana-de-contexto-qué-es-y-por-qué-importa)
2. [Estrategias de compresión de contexto](#2-estrategias-de-compresión-de-contexto)
3. [Sliding window y chunking](#3-sliding-window-y-chunking)
4. [Summarization recursiva del contexto](#4-summarization-recursiva-del-contexto)
5. [Gestión de memoria en agentes](#5-gestión-de-memoria-en-agentes)
6. [Prompt Caching para reducir costos](#6-prompt-caching-para-reducir-costos)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. La ventana de contexto: qué es y por qué importa

Todos los LLMs tienen un límite máximo de tokens que pueden procesar en una sola llamada. Este límite incluye el prompt (instrucciones + contexto + historial) y la respuesta generada.

| Modelo | Contexto máximo |
|---|---|
| Claude Opus 4.6 | 200,000 tokens |
| GPT-4o | 128,000 tokens |
| Gemini 1.5 Pro | 1,000,000 tokens |
| Llama 3.1 70B | 128,000 tokens |

### El problema "lost in the middle"

Los LLMs no procesan todos los tokens del contexto con igual atención. Estudios empíricos muestran que la información al principio y al final del contexto se recuerda mejor que la del medio. Para documentos muy largos, la información relevante enterrada en el centro puede ignorarse efectivamente.

```python
# Estimar tokens antes de llamar a la API
import anthropic

client = anthropic.Anthropic()

def estimar_tokens(texto: str) -> int:
    """Estimación rápida: ~4 caracteres = 1 token para texto en inglés/español."""
    return len(texto) // 4

def verificar_limite_contexto(
    messages: list[dict],
    system: str = "",
    max_tokens_respuesta: int = 1024,
    modelo: str = "claude-opus-4-6",
    limite_modelo: int = 200_000
) -> dict:
    texto_total = system + " ".join(m["content"] for m in messages)
    tokens_estimados = estimar_tokens(texto_total) + max_tokens_respuesta

    return {
        "tokens_estimados": tokens_estimados,
        "limite_modelo": limite_modelo,
        "dentro_limite": tokens_estimados < limite_modelo,
        "uso_porcentaje": tokens_estimados / limite_modelo * 100,
        "tokens_disponibles": limite_modelo - tokens_estimados
    }
```

---

## 2. Estrategias de compresión de contexto

```python
# context_compression.py
import anthropic

client = anthropic.Anthropic()


def comprimir_historial(
    historial: list[dict],
    sistema: str,
    max_tokens_comprimido: int = 2000
) -> str:
    """
    Resume el historial de conversación cuando se acerca al límite del contexto.
    Devuelve un resumen compacto que captura la información esencial.
    """
    historial_texto = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in historial
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=max_tokens_comprimido,
        messages=[{
            "role": "user",
            "content": (
                f"Resume esta conversación preservando:\n"
                f"1. Decisiones y acuerdos tomados\n"
                f"2. Información factual importante\n"
                f"3. Preguntas pendientes\n"
                f"4. El tono y contexto de la relación\n\n"
                f"Conversación:\n{historial_texto}"
            )
        }]
    )
    return response.content[0].text


class ConversacionConMemoria:
    """Gestiona una conversación larga con compresión automática del contexto."""

    def __init__(
        self,
        sistema: str = "",
        max_mensajes: int = 20,
        modelo: str = "claude-opus-4-6"
    ):
        self.sistema = sistema
        self.max_mensajes = max_mensajes
        self.modelo = modelo
        self.historial: list[dict] = []
        self.resumen_previo: str = ""

    def chat(self, mensaje: str) -> str:
        self.historial.append({"role": "user", "content": mensaje})

        # Comprimir si superamos el límite de mensajes
        if len(self.historial) > self.max_mensajes:
            self._comprimir()

        # Construir messages con contexto comprimido si existe
        messages = []
        if self.resumen_previo:
            messages.append({
                "role": "user",
                "content": f"[Resumen de conversación anterior]:\n{self.resumen_previo}"
            })
            messages.append({"role": "assistant", "content": "Entendido, continúo con ese contexto."})

        messages.extend(self.historial)

        response = client.messages.create(
            model=self.modelo,
            max_tokens=1024,
            system=self.sistema,
            messages=messages
        )

        respuesta = response.content[0].text
        self.historial.append({"role": "assistant", "content": respuesta})
        return respuesta

    def _comprimir(self):
        # Comprimir los mensajes más antiguos, mantener los últimos 5
        mensajes_a_comprimir = self.historial[:-5]
        self.resumen_previo = comprimir_historial(
            mensajes_a_comprimir,
            self.sistema
        )
        self.historial = self.historial[-5:]
        print(f"Contexto comprimido: {len(mensajes_a_comprimir)} mensajes → resumen")
```

---

## 3. Sliding window y chunking

```python
# chunking.py
from typing import Generator

def chunk_por_tokens(
    texto: str,
    max_tokens: int = 4000,
    overlap_tokens: int = 200
) -> list[str]:
    """
    Divide un texto largo en chunks con solapamiento.
    El solapamiento preserva coherencia entre chunks.
    """
    chars_per_chunk = max_tokens * 4  # ~4 chars por token
    overlap_chars = overlap_tokens * 4

    chunks = []
    inicio = 0

    while inicio < len(texto):
        fin = min(inicio + chars_per_chunk, len(texto))

        # Intentar cortar en un punto natural (párrafo o frase)
        if fin < len(texto):
            # Buscar el último punto o salto de línea antes del límite
            corte = texto.rfind("\n", inicio, fin)
            if corte == -1:
                corte = texto.rfind(". ", inicio, fin)
            if corte != -1:
                fin = corte + 1

        chunks.append(texto[inicio:fin])
        inicio = fin - overlap_chars  # retroceder para el solapamiento

    return chunks


def procesar_documento_largo(
    texto: str,
    tarea: str,
    modelo: str = "claude-opus-4-6"
) -> str:
    """Procesa un documento largo dividiendo en chunks y combinando resultados."""
    chunks = chunk_por_tokens(texto, max_tokens=4000, overlap_tokens=200)
    resultados = []

    for i, chunk in enumerate(chunks):
        print(f"Procesando chunk {i+1}/{len(chunks)}...")
        response = client.messages.create(
            model=modelo,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"{tarea}\n\nTexto (parte {i+1} de {len(chunks)}):\n{chunk}"
            }]
        )
        resultados.append(response.content[0].text)

    # Combinar resultados si hay más de uno
    if len(resultados) == 1:
        return resultados[0]

    combinacion = client.messages.create(
        model=modelo,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Combina estos {len(resultados)} análisis parciales en uno coherente:\n\n"
                + "\n\n---\n\n".join(f"Parte {i+1}:\n{r}" for i, r in enumerate(resultados))
            )
        }]
    )
    return combinacion.content[0].text
```

---

## 4. Summarization recursiva del contexto

```python
# recursive_summarization.py

def resumir_recursivo(
    textos: list[str],
    modelo: str = "claude-opus-4-6",
    max_tokens_por_grupo: int = 4000,
    max_tokens_resumen: int = 512
) -> str:
    """
    Implementa el patrón map-reduce para documentos muy largos:
    1. Resumir cada chunk individualmente (map)
    2. Si los resúmenes combinados siguen siendo largos, resumirlos de nuevo (reduce recursivo)
    """
    if len(textos) == 1:
        return textos[0]

    # Map: resumir cada texto
    resumenes = []
    for texto in textos:
        if estimar_tokens(texto) > max_tokens_por_grupo:
            # Dividir y resumir recursivamente
            sub_chunks = chunk_por_tokens(texto, max_tokens=max_tokens_por_grupo)
            resumen = resumir_recursivo(sub_chunks, modelo, max_tokens_por_grupo, max_tokens_resumen)
        else:
            response = client.messages.create(
                model=modelo,
                max_tokens=max_tokens_resumen,
                messages=[{"role": "user", "content": f"Resume concisamente:\n{texto}"}]
            )
            resumen = response.content[0].text
        resumenes.append(resumen)

    # Reduce: combinar resúmenes
    if estimar_tokens(" ".join(resumenes)) > max_tokens_por_grupo:
        return resumir_recursivo(resumenes, modelo, max_tokens_por_grupo, max_tokens_resumen)

    response = client.messages.create(
        model=modelo,
        max_tokens=max_tokens_resumen * 2,
        messages=[{
            "role": "user",
            "content": "Combina estos resúmenes en uno final coherente:\n\n" +
                       "\n\n".join(resumenes)
        }]
    )
    return response.content[0].text
```

---

## 5. Gestión de memoria en agentes

```python
# agent_memory.py
from datetime import datetime
import json

class MemoriaAgente:
    """
    Sistema de memoria multicapa para agentes:
    - Memoria de trabajo: contexto inmediato (ventana de contexto)
    - Memoria episódica: hechos y eventos de la sesión (JSON en disco)
    - Memoria semántica: conocimiento a largo plazo (vector DB - ver Bloque 9)
    """

    def __init__(self, agente_id: str, max_trabajo: int = 10):
        self.agente_id = agente_id
        self.max_trabajo = max_trabajo
        self.memoria_trabajo: list[dict] = []
        self.memoria_episodica: list[dict] = []

    def recordar(self, hecho: str, categoria: str = "general"):
        """Guarda un hecho en memoria episódica."""
        self.memoria_episodica.append({
            "hecho": hecho,
            "categoria": categoria,
            "timestamp": datetime.now().isoformat()
        })

    def agregar_turno(self, rol: str, contenido: str):
        """Añade un turno a memoria de trabajo, comprimiendo si es necesario."""
        self.memoria_trabajo.append({"role": rol, "content": contenido})
        if len(self.memoria_trabajo) > self.max_trabajo:
            # Archivar los más antiguos en episódica
            antiguos = self.memoria_trabajo[:-self.max_trabajo//2]
            self.recordar(
                f"Resumen de {len(antiguos)} turnos anteriores: " +
                "; ".join(m["content"][:100] for m in antiguos),
                categoria="historial_comprimido"
            )
            self.memoria_trabajo = self.memoria_trabajo[-self.max_trabajo//2:]

    def contexto_para_prompt(self) -> str:
        """Genera el bloque de contexto para incluir en el system prompt."""
        if not self.memoria_episodica:
            return ""
        hechos_relevantes = self.memoria_episodica[-5:]  # los más recientes
        return "Hechos recordados:\n" + "\n".join(
            f"- [{h['categoria']}] {h['hecho']}" for h in hechos_relevantes
        )

    def guardar(self, ruta: str):
        with open(ruta, "w") as f:
            json.dump({
                "agente_id": self.agente_id,
                "episodica": self.memoria_episodica
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def cargar(cls, ruta: str) -> "MemoriaAgente":
        with open(ruta) as f:
            data = json.load(f)
        memoria = cls(data["agente_id"])
        memoria.memoria_episodica = data["episodica"]
        return memoria
```

---

## 6. Prompt Caching para reducir costos

```python
# prompt_caching.py
# Anthropic soporta prompt caching: si el prefijo del prompt no cambia,
# se reutiliza el KV-cache y el costo de los tokens cacheados baja ~90%.

import anthropic

client = anthropic.Anthropic()

# Documento largo que se reutiliza en múltiples consultas
DOCUMENTO_BASE = "..." * 10000  # documento de 40,000+ caracteres


def consultar_con_cache(pregunta: str) -> str:
    """
    Marca el documento base con cache_control para que Anthropic lo cachee.
    La primera llamada paga precio normal; las siguientes pagan ~10% en el documento.
    """
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": "Eres un asistente que responde preguntas sobre el siguiente documento:"
            },
            {
                "type": "text",
                "text": DOCUMENTO_BASE,
                "cache_control": {"type": "ephemeral"}  # marcar para cacheo
            }
        ],
        messages=[{"role": "user", "content": pregunta}]
    )

    # Verificar uso del cache
    usage = response.usage
    print(f"Tokens: input={usage.input_tokens}, "
          f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)}, "
          f"cache_creation={getattr(usage, 'cache_creation_input_tokens', 0)}")

    return response.content[0].text
```

---

## 7. Extensiones sugeridas

- **Context distillation**: entrenar un modelo más pequeño usando las salidas del modelo grande como datos de destilación
- **Selective context**: usar un clasificador ligero para decidir qué partes del historial son relevantes para la pregunta actual
- **Extended thinking**: modelos con razonamiento explícito (Claude 3.7 Sonnet) para tareas complejas sobre documentos largos

---

**Anterior:** [09 — LangGraph](./09-langgraph.md) · **Siguiente:** [11 — Tokenización en profundidad](./11-tokenizacion.md)
