# 04 — Integración con Herramientas de Negocio

> **Bloque:** Automatización con IA · **Nivel:** Intermedio · **Tiempo estimado:** 55 min

---

## Índice

1. [CRM con IA: Salesforce y HubSpot](#1-crm-con-ia-salesforce-y-hubspot)
2. [Google Workspace con Apps Script y LLM](#2-google-workspace-con-apps-script-y-llm)
3. [Microsoft 365: Power Automate y Azure OpenAI](#3-microsoft-365-power-automate-y-azure-openai)
4. [Slack bot con IA](#4-slack-bot-con-ia)
5. [Notion con IA](#5-notion-con-ia)
6. [Patrones de integración](#6-patrones-de-integración)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. CRM con IA: Salesforce y HubSpot

### HubSpot + Claude via API

HubSpot expone una API REST completa. El patrón más común es enriquecer los contactos o notas con análisis de IA después de cada interacción.

```python
# hubspot_ia.py
import httpx
import anthropic
from typing import Optional

HS_TOKEN = "pat-..."           # HubSpot Private App token
ANTHROPIC_KEY = "sk-ant-..."

hs_client = httpx.Client(
    base_url="https://api.hubapi.com",
    headers={"Authorization": f"Bearer {HS_TOKEN}", "Content-Type": "application/json"},
    timeout=30
)
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def obtener_notas_contacto(contact_id: str) -> list[str]:
    """Obtiene las notas de engagement de un contacto en HubSpot."""
    r = hs_client.get(
        f"/crm/v3/objects/contacts/{contact_id}/associations/notes",
    )
    r.raise_for_status()
    note_ids = [a["id"] for a in r.json().get("results", [])]

    notas = []
    for note_id in note_ids[:10]:  # últimas 10 notas
        rn = hs_client.get(f"/crm/v3/objects/notes/{note_id}",
                           params={"properties": "hs_note_body"})
        if rn.is_success:
            body = rn.json()["properties"].get("hs_note_body", "")
            if body:
                notas.append(body)
    return notas


def analizar_sentimiento_y_siguiente_paso(contact_id: str) -> dict:
    """Analiza el historial de un contacto y sugiere el siguiente paso comercial."""
    notas = obtener_notas_contacto(contact_id)
    if not notas:
        return {"error": "Sin notas disponibles"}

    historial = "\n---\n".join(notas)
    message = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Analiza el siguiente historial de interacciones con un prospecto comercial "
                f"y devuelve un JSON con:\n"
                f"- sentimiento: positivo/neutro/negativo\n"
                f"- resumen: 1-2 frases del estado de la relación\n"
                f"- siguiente_paso: acción concreta recomendada\n"
                f"- urgencia: alta/media/baja\n\n"
                f"Historial:\n{historial}"
            )
        }]
    )
    import json
    try:
        return json.loads(message.content[0].text)
    except json.JSONDecodeError:
        return {"raw": message.content[0].text}


def actualizar_propiedad_contacto(contact_id: str, propiedades: dict):
    """Actualiza propiedades personalizadas del contacto en HubSpot."""
    r = hs_client.patch(
        f"/crm/v3/objects/contacts/{contact_id}",
        json={"properties": propiedades}
    )
    r.raise_for_status()
    return r.json()


# Ejemplo de uso en un batch
def enriquecer_contactos(contact_ids: list[str]):
    for cid in contact_ids:
        analisis = analizar_sentimiento_y_siguiente_paso(cid)
        if "error" not in analisis:
            actualizar_propiedad_contacto(cid, {
                "sentiment_ia": analisis.get("sentimiento", ""),
                "next_action_ia": analisis.get("siguiente_paso", ""),
                "ia_urgency": analisis.get("urgencia", "")
            })
            print(f"✅ Contacto {cid} enriquecido: {analisis.get('siguiente_paso', '')}")
```

---

## 2. Google Workspace con Apps Script y LLM

Google Apps Script permite automatizar Docs, Sheets y Gmail directamente desde el navegador, sin servidor externo.

```javascript
// Apps Script — Resumir emails con Claude API
// Pegar en https://script.google.com/ y adjuntar a una hoja de cálculo

const ANTHROPIC_API_KEY = PropertiesService.getScriptProperties().getProperty("ANTHROPIC_KEY");

function resumirEmailsRecientes() {
  const threads = GmailApp.search("is:unread label:inbox newer_than:1d", 0, 20);
  const hoja = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Resúmenes") 
    || SpreadsheetApp.getActiveSpreadsheet().insertSheet("Resúmenes");

  hoja.clearContents();
  hoja.appendRow(["Fecha", "De", "Asunto", "Resumen IA", "Acción sugerida"]);

  threads.forEach(thread => {
    const mensaje = thread.getMessages()[0];
    const cuerpo = mensaje.getPlainBody().substring(0, 2000);
    
    const analisis = llamarClaude(
      `Analiza este email y devuelve JSON con:\n` +
      `- resumen: máximo 2 frases\n` +
      `- accion: qué hacer (responder/archivar/delegar/urgente)\n\n` +
      `Email:\n${cuerpo}`
    );

    let resumen = "Error al procesar";
    let accion = "";
    try {
      const parsed = JSON.parse(analisis);
      resumen = parsed.resumen || analisis;
      accion = parsed.accion || "";
    } catch(e) {
      resumen = analisis.substring(0, 200);
    }

    hoja.appendRow([
      mensaje.getDate(),
      mensaje.getFrom(),
      mensaje.getSubject(),
      resumen,
      accion
    ]);
  });

  SpreadsheetApp.getUi().alert(`Procesados ${threads.length} emails`);
}

function llamarClaude(prompt) {
  const payload = {
    model: "claude-opus-4-6",
    max_tokens: 256,
    messages: [{ role: "user", content: prompt }]
  };

  const options = {
    method: "post",
    contentType: "application/json",
    headers: {
      "x-api-key": ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01"
    },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  const response = UrlFetchApp.fetch("https://api.anthropic.com/v1/messages", options);
  const data = JSON.parse(response.getContentText());
  return data.content[0].text;
}

// Menú personalizado en la hoja de cálculo
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("🤖 IA Asistente")
    .addItem("Resumir emails de hoy", "resumirEmailsRecientes")
    .addToUi();
}
```

---

## 3. Microsoft 365: Power Automate y Azure OpenAI

```python
# azure_openai_365.py — Integración desde Python con Azure OpenAI
from openai import AzureOpenAI
import httpx
import os

# Azure OpenAI (compatible con el SDK de OpenAI)
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  # https://tu-recurso.openai.azure.com/
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2024-02-01"
)


def procesar_documento_word(contenido_texto: str, tarea: str = "resumir") -> str:
    """Procesa el texto de un documento Word con Azure OpenAI."""
    prompts = {
        "resumir": "Resume este documento en 5 puntos clave:",
        "revisar": "Identifica errores gramaticales, incoherencias y mejoras posibles en este texto:",
        "traducir": "Traduce este documento al inglés manteniendo el formato:",
        "extraer_datos": "Extrae todos los datos estructurados (fechas, nombres, cantidades, plazos) en formato JSON:"
    }
    prompt = prompts.get(tarea, tarea)

    response = client.chat.completions.create(
        model="gpt-4o",  # nombre del deployment en Azure
        messages=[
            {"role": "system", "content": "Eres un asistente especializado en documentos corporativos."},
            {"role": "user", "content": f"{prompt}\n\n{contenido_texto[:4000]}"}
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content


# Power Automate webhook — recibir llamadas desde flujos de Power Automate
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()


class PARequest(BaseModel):
    documento_texto: str
    tarea: str = "resumir"
    usuario: str = ""


@app.post("/procesar-documento")
async def procesar(req: PARequest):
    """
    Endpoint llamado desde Power Automate.
    Configurar en PA como HTTP action apuntando a este endpoint.
    """
    resultado = procesar_documento_word(req.documento_texto, req.tarea)
    return {
        "resultado": resultado,
        "tarea": req.tarea,
        "usuario": req.usuario,
        "tokens_estimados": len(req.documento_texto.split()) // 0.75
    }
```

**Configurar Power Automate:**
1. Crear flujo → Trigger: "Cuando se crea un archivo en SharePoint"
2. Acción: "Obtener contenido del archivo" → extraer texto
3. Acción: "HTTP" → POST a tu endpoint FastAPI con el texto
4. Acción: "Actualizar elemento de lista" → guardar el resultado en SharePoint

---

## 4. Slack bot con IA

```python
# slack_bot.py
import os
import re
import anthropic
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ["SLACK_BOT_TOKEN"])
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Memoria de conversación por canal (en prod: usar Redis)
conversaciones: dict[str, list[dict]] = {}
MAX_HISTORIA = 10


def obtener_respuesta(channel_id: str, mensaje_usuario: str) -> str:
    """Genera respuesta de Claude manteniendo contexto por canal."""
    if channel_id not in conversaciones:
        conversaciones[channel_id] = []

    conversaciones[channel_id].append({"role": "user", "content": mensaje_usuario})

    # Mantener solo los últimos N turnos
    historia = conversaciones[channel_id][-MAX_HISTORIA:]

    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "Eres un asistente corporativo en Slack. "
            "Responde de forma concisa y directa. "
            "Usa formato Markdown de Slack (negritas con *texto*, listas con -). "
            "Si te preguntan sobre datos internos que no conoces, dilo claramente."
        ),
        messages=historia
    )

    respuesta = response.content[0].text
    conversaciones[channel_id].append({"role": "assistant", "content": respuesta})
    return respuesta


# Responder a menciones (@bot)
@app.event("app_mention")
def handle_mention(event, say, client):
    channel = event["channel"]
    texto = re.sub(r"<@[A-Z0-9]+>", "", event["text"]).strip()

    if not texto:
        say("¡Hola! ¿En qué puedo ayudarte?")
        return

    # Mostrar indicador de escritura
    client.chat_postEphemeral(
        channel=channel,
        user=event["user"],
        text="🤔 Pensando..."
    )

    respuesta = obtener_respuesta(channel, texto)
    say(respuesta)


# Comando slash /resumir
@app.command("/resumir")
def handle_resumir(ack, respond, command):
    ack()
    texto = command.get("text", "").strip()
    if not texto:
        respond("Uso: `/resumir [texto a resumir]`")
        return

    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": f"Resume esto en 3 puntos:\n{texto}"}]
    )
    respond(response.content[0].text)


# Botón de acción en mensajes (interactividad)
@app.action("analizar_mensaje")
def handle_analizar(ack, body, say):
    ack()
    mensaje_original = body["message"]["text"]
    analisis = obtener_respuesta(body["channel"]["id"], f"Analiza este mensaje: {mensaje_original}")
    say(f"*Análisis:*\n{analisis}")


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
```

```bash
# Variables necesarias
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
export ANTHROPIC_API_KEY=sk-ant-...

pip install slack-bolt anthropic
python slack_bot.py
```

---

## 5. Notion con IA

```python
# notion_ia.py
import httpx
import anthropic
import os
from datetime import datetime

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

notion = httpx.Client(
    base_url="https://api.notion.com/v1",
    headers={
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    },
    timeout=30
)
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def leer_pagina_notion(page_id: str) -> str:
    """Extrae el texto de una página de Notion."""
    r = notion.get(f"/blocks/{page_id}/children")
    r.raise_for_status()
    bloques = r.json().get("results", [])

    textos = []
    for bloque in bloques:
        tipo = bloque["type"]
        if tipo in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item"]:
            rich_text = bloque[tipo].get("rich_text", [])
            texto = "".join(t["plain_text"] for t in rich_text)
            if texto:
                textos.append(texto)

    return "\n".join(textos)


def crear_pagina_en_base_datos(database_id: str, titulo: str, contenido_md: str, tags: list[str] = None):
    """Crea una nueva página en una base de datos de Notion."""
    # Convertir markdown básico a bloques de Notion
    parrafos = contenido_md.split("\n\n")
    bloques = []
    for parrafo in parrafos[:50]:  # límite de Notion: 100 bloques
        if parrafo.startswith("# "):
            bloques.append({
                "object": "block", "type": "heading_1",
                "heading_1": {"rich_text": [{"type": "text", "text": {"content": parrafo[2:]}}]}
            })
        elif parrafo.startswith("## "):
            bloques.append({
                "object": "block", "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": parrafo[3:]}}]}
            })
        elif parrafo.strip():
            bloques.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": parrafo[:2000]}}]}
            })

    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Nombre": {"title": [{"text": {"content": titulo}}]},
            "Fecha": {"date": {"start": datetime.now().isoformat()}},
        },
        "children": bloques
    }

    if tags:
        payload["properties"]["Tags"] = {"multi_select": [{"name": t} for t in tags]}

    r = notion.post("/pages", json=payload)
    r.raise_for_status()
    return r.json()["url"]


def generar_y_guardar_resumen(page_id: str, database_id: str) -> str:
    """Lee una página de Notion, la resume con Claude y guarda el resumen en una BD."""
    contenido = leer_pagina_notion(page_id)
    if not contenido:
        return "Página vacía"

    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Genera un resumen ejecutivo de este documento con:\n"
                f"## Puntos clave\n(3-5 puntos)\n\n"
                f"## Decisiones/Acciones pendientes\n\n"
                f"## Contexto relevante\n\n"
                f"Documento:\n{contenido[:3000]}"
            )
        }]
    )
    resumen = response.content[0].text

    url = crear_pagina_en_base_datos(
        database_id=database_id,
        titulo=f"Resumen — {datetime.now().strftime('%Y-%m-%d')}",
        contenido_md=resumen,
        tags=["IA", "Resumen automático"]
    )
    return url
```

---

## 6. Patrones de integración

```python
# integration_patterns.py — Guía de decisión

PATRONES = {
    "webhook": {
        "descripcion": "El sistema externo llama a tu endpoint cuando ocurre un evento",
        "cuando_usar": "Eventos en tiempo real: nuevo lead, mensaje recibido, formulario enviado",
        "ejemplo": "HubSpot webhook → FastAPI → Claude → actualizar CRM",
        "latencia": "baja (<2s)",
        "complejidad": "media"
    },
    "polling": {
        "descripcion": "Tu sistema consulta periódicamente el sistema externo",
        "cuando_usar": "Sistemas sin webhooks, consolidación de datos periódica",
        "ejemplo": "Cron cada 15min → consultar Notion → procesar con IA → actualizar Sheets",
        "latencia": "alta (hasta interval)",
        "complejidad": "baja"
    },
    "streaming": {
        "descripcion": "Conexión persistente que recibe eventos en tiempo real (SSE, WebSocket)",
        "cuando_usar": "Chat en tiempo real, dashboards live, Slack bots",
        "ejemplo": "Slack Socket Mode → bot escucha mensajes → responde con streaming de Claude",
        "latencia": "muy baja (<500ms)",
        "complejidad": "alta"
    },
    "batch": {
        "descripcion": "Procesar grandes volúmenes de forma asíncrona y eficiente",
        "cuando_usar": "Enriquecer miles de registros, análisis masivo de documentos",
        "ejemplo": "Anthropic Batch API → procesar 10,000 contratos overnight → guardar resultados",
        "latencia": "horas",
        "complejidad": "baja"
    }
}


def seleccionar_patron(
    volumen: str,      # "bajo" (<100/día) | "medio" (100-10k) | "alto" (>10k)
    latencia: str,     # "tiempo_real" | "minutos" | "horas"
    tiene_webhooks: bool
) -> str:
    if latencia == "tiempo_real":
        return "streaming" if not tiene_webhooks else "webhook"
    elif volumen == "alto":
        return "batch"
    elif tiene_webhooks:
        return "webhook"
    else:
        return "polling"


# Ejemplo de decisión
patron = seleccionar_patron(
    volumen="medio",
    latencia="minutos",
    tiene_webhooks=True
)
print(f"Patrón recomendado: {patron}")  # webhook
```

---

## 7. Extensiones sugeridas

- **Zapier + webhooks**: usar Zapier como pegamento entre herramientas SaaS sin código propio
- **Make.com escenarios complejos**: usar iteradores y routers de Make para pipelines multi-paso
- **Autenticación OAuth**: implementar OAuth 2.0 para que múltiples usuarios conecten sus cuentas
- **Rate limiting inteligente**: usar token bucket para respetar los límites de las APIs externas

---

**Anterior:** [03 — Pipelines de negocio](./03-pipelines-negocio.md) · **Siguiente bloque:** [Bloque 14 — Fine-tuning avanzado](../finetuning-avanzado/)
