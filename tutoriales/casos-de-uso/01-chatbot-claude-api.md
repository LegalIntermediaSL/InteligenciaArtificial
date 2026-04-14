# 01 — Chatbot con Claude API

> **Bloque:** Casos de uso · **Nivel:** Práctico · **Tiempo estimado:** 30 min

---

## Índice

1. [Objetivo](#1-objetivo)
2. [Requisitos](#2-requisitos)
3. [Versión mínima — chatbot de terminal](#3-versión-mínima--chatbot-de-terminal)
4. [Versión completa — con system prompt y memoria](#4-versión-completa--con-system-prompt-y-memoria)
5. [Guardar y cargar conversaciones](#5-guardar-y-cargar-conversaciones)
6. [Chatbot especializado — ejemplo real](#6-chatbot-especializado--ejemplo-real)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Objetivo

Construir un chatbot conversacional completo usando la API de Anthropic (Claude) que:

- Mantiene el historial de la conversación
- Acepta un system prompt configurable
- Limita el contexto para controlar costes
- Guarda y carga conversaciones desde disco

---

## 2. Requisitos

```bash
pip install anthropic python-dotenv
```

Fichero `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 3. Versión mínima — chatbot de terminal

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
historial = []

print("Chatbot Claude — escribe 'salir' para terminar\n")

while True:
    entrada = input("Tú: ").strip()
    if entrada.lower() in ("salir", "exit", "quit"):
        break
    if not entrada:
        continue

    historial.append({"role": "user", "content": entrada})

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=historial
    )

    texto = respuesta.content[0].text
    historial.append({"role": "assistant", "content": texto})

    print(f"\nClaude: {texto}\n")
```

---

## 4. Versión completa — con system prompt y memoria

```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


class Chatbot:
    """Chatbot conversacional basado en Claude."""

    def __init__(
        self,
        modelo: str = "claude-sonnet-4-6",
        system_prompt: str = "Eres un asistente útil. Responde siempre en español.",
        max_tokens: int = 1024,
        temperatura: float = 0.7,
        max_mensajes_historial: int = 20,
    ):
        self.client = anthropic.Anthropic()
        self.modelo = modelo
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperatura = temperatura
        self.max_mensajes_historial = max_mensajes_historial
        self.historial: list[dict] = []

    def _truncar_historial(self):
        """Mantiene el historial dentro del límite configurado."""
        if len(self.historial) > self.max_mensajes_historial:
            # Conservar siempre número par (user/assistant)
            self.historial = self.historial[-self.max_mensajes_historial:]

    def chat(self, mensaje: str) -> str:
        """Envía un mensaje y devuelve la respuesta."""
        self.historial.append({"role": "user", "content": mensaje})
        self._truncar_historial()

        respuesta = self.client.messages.create(
            model=self.modelo,
            max_tokens=self.max_tokens,
            temperature=self.temperatura,
            system=self.system_prompt,
            messages=self.historial,
        )

        texto = respuesta.content[0].text
        self.historial.append({"role": "assistant", "content": texto})

        # Mostrar uso de tokens
        uso = respuesta.usage
        print(f"  [tokens — entrada: {uso.input_tokens}, salida: {uso.output_tokens}]")

        return texto

    def limpiar(self):
        """Borra el historial de la conversación."""
        self.historial = []
        print("Historial borrado.")

    def mostrar_historial(self):
        """Imprime el historial completo."""
        if not self.historial:
            print("Historial vacío.")
            return
        for i, msg in enumerate(self.historial):
            rol = "Tú" if msg["role"] == "user" else "Claude"
            print(f"\n[{i+1}] {rol}:\n{msg['content']}")


def main():
    bot = Chatbot(
        system_prompt="""Eres un asistente experto en inteligencia artificial.
        - Responde siempre en español
        - Usa ejemplos concretos cuando sea posible
        - Si no sabes algo con certeza, indícalo"""
    )

    print("=" * 50)
    print("Chatbot IA — Claude")
    print("Comandos: 'salir', 'limpiar', 'historial'")
    print("=" * 50)

    while True:
        try:
            entrada = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not entrada:
            continue

        if entrada.lower() == "salir":
            print("Hasta luego!")
            break
        elif entrada.lower() == "limpiar":
            bot.limpiar()
            continue
        elif entrada.lower() == "historial":
            bot.mostrar_historial()
            continue

        respuesta = bot.chat(entrada)
        print(f"\nClaude: {respuesta}")


if __name__ == "__main__":
    main()
```

---

## 5. Guardar y cargar conversaciones

```python
import json
from pathlib import Path
from datetime import datetime


class ChatbotConPersistencia(Chatbot):
    """Extiende Chatbot añadiendo persistencia de conversaciones."""

    def guardar(self, ruta: str = None):
        """Guarda la conversación en un fichero JSON."""
        if ruta is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta = f"conversaciones/conv_{timestamp}.json"

        Path(ruta).parent.mkdir(parents=True, exist_ok=True)

        datos = {
            "modelo": self.modelo,
            "system_prompt": self.system_prompt,
            "fecha": datetime.now().isoformat(),
            "mensajes": self.historial,
        }

        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

        print(f"Conversación guardada en: {ruta}")

    def cargar(self, ruta: str):
        """Carga una conversación desde un fichero JSON."""
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)

        self.historial = datos["mensajes"]
        print(f"Conversación cargada: {len(self.historial)} mensajes")
        print(f"Fecha original: {datos.get('fecha', 'desconocida')}")


# Uso
bot = ChatbotConPersistencia()
bot.chat("¿Qué es el aprendizaje por refuerzo?")
bot.chat("Dame un ejemplo práctico")
bot.guardar()

# Reanudar más tarde
bot2 = ChatbotConPersistencia()
bot2.cargar("conversaciones/conv_20260414_120000.json")
bot2.chat("Continúa explicándome el tema anterior")
```

---

## 6. Chatbot especializado — ejemplo real

Un chatbot de soporte técnico para una empresa de software:

```python
SYSTEM_PROMPT_SOPORTE = """Eres el asistente de soporte técnico de TechCorp.

## Tu rol
Ayudas a los usuarios a resolver problemas con nuestros productos:
- TechApp Desktop (versiones 3.x y 4.x)
- TechCloud (plataforma SaaS)
- TechAPI (API REST para desarrolladores)

## Cómo responder
1. Saluda brevemente y confirma que entiendes el problema
2. Haz preguntas de diagnóstico si necesitas más información
3. Proporciona pasos concretos y numerados para la solución
4. Si el problema requiere escalado, indica: "Voy a escalar este caso a nuestro equipo técnico"

## Límites
- No accedes a datos de cuentas de usuario
- No puedes modificar configuraciones remotamente
- Para problemas de facturación, remite siempre a facturacion@techcorp.com

## Tono
Profesional, empático y conciso. Evita la jerga técnica innecesaria.
"""

bot_soporte = Chatbot(
    system_prompt=SYSTEM_PROMPT_SOPORTE,
    temperatura=0.3,  # Más determinista para soporte técnico
)

# Simular una sesión de soporte
print(bot_soporte.chat("Hola, no puedo iniciar sesión en TechApp, me sale error 403"))
print(bot_soporte.chat("Uso la versión 4.2 en Windows 11"))
print(bot_soporte.chat("Sí, ya intenté borrar la caché"))
```

---

## 7. Extensiones sugeridas

| Extensión | Descripción | Tecnología |
|---|---|---|
| **Interfaz web** | Añadir frontend con Streamlit o Gradio | `pip install streamlit` |
| **Streaming** | Mostrar respuesta token a token | `client.messages.stream()` |
| **Múltiples personalidades** | Cambiar system prompt en tiempo de ejecución | Añadir método `cambiar_rol()` |
| **RAG** | Responder basándose en tus documentos | ChromaDB + embeddings |
| **Voz** | Transcribir audio de entrada con Whisper | OpenAI Whisper |
| **Exportar a PDF/Word** | Guardar conversaciones formateadas | `fpdf2`, `python-docx` |

### Ejemplo con Streamlit (interfaz web en 20 líneas)

```python
# app.py
import streamlit as st
import anthropic

st.title("Chatbot Claude")

if "historial" not in st.session_state:
    st.session_state.historial = []

client = anthropic.Anthropic()

# Mostrar historial
for msg in st.session_state.historial:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input del usuario
if prompt := st.chat_input("Escribe tu mensaje..."):
    st.session_state.historial.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=st.session_state.historial
            )
            texto = respuesta.content[0].text
            st.write(texto)

    st.session_state.historial.append({"role": "assistant", "content": texto})
```

```bash
streamlit run app.py
```

---

**Siguiente:** [02 — Clasificación de texto](./02-clasificacion-texto.md)
