# 01 — n8n con IA

> **Bloque:** Automatización con IA · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [Qué es n8n y por qué usarlo con IA](#1-qué-es-n8n-y-por-qué-usarlo-con-ia)
2. [Instalación con Docker](#2-instalación-con-docker)
3. [Nodos de IA disponibles](#3-nodos-de-ia-disponibles)
4. [Workflow completo: webhook → LLM → BD → respuesta](#4-workflow-completo-webhook--llm--bd--respuesta)
5. [AI Agent node: tool use y memoria](#5-ai-agent-node-tool-use-y-memoria)
6. [Casos prácticos](#6-casos-prácticos)
7. [Self-hosted vs n8n Cloud](#7-self-hosted-vs-n8n-cloud)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Qué es n8n y por qué usarlo con IA

n8n es una plataforma de automatización de flujos de trabajo (workflow automation) de código abierto con interfaz visual. A diferencia de Make o Zapier, puede alojarse en tu propio servidor —sin límites de ejecuciones ni costes por operación—, y tiene nodos nativos para los principales proveedores de IA.

| Característica | n8n | Make | Zapier |
|---|---|---|---|
| Open-source | Sí (con licencia fair-code) | No | No |
| Self-hosted | Sí | No | No |
| Nodos de IA nativos | Sí (OpenAI, Anthropic, HF…) | Limitado | Limitado |
| Código JavaScript en nodos | Sí | No | No |
| Precio | Gratis (self-hosted) / desde 20 €/mes (cloud) | Desde 9 €/mes | Desde 19,99 $/mes |
| Curva de aprendizaje | Media | Baja | Muy baja |

**Casos de uso ideales para n8n + IA:**
- Clasificación y enrutamiento automático de tickets/emails
- Extracción de datos de documentos (facturas, contratos, formularios)
- Chatbots internos que consultan bases de datos propias
- Pipelines de contenido (generar, revisar, publicar)
- Alertas inteligentes basadas en análisis de logs o métricas

---

## 2. Instalación con Docker

### Instalación mínima

```bash
# Crear volumen persistente
docker volume create n8n_data

# Arrancar n8n (accesible en http://localhost:5678)
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  -e N8N_BASIC_AUTH_ACTIVE=true \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=password_seguro \
  docker.n8n.io/n8nio/n8n
```

### Con Docker Compose (recomendado para producción)

```yaml
# docker-compose.yml
version: "3.8"

services:
  postgres:
    image: postgres:15
    restart: always
    environment:
      POSTGRES_DB: n8n
      POSTGRES_USER: n8n
      POSTGRES_PASSWORD: n8n_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  n8n:
    image: docker.n8n.io/n8nio/n8n:latest
    restart: always
    ports:
      - "5678:5678"
    environment:
      # Base de datos
      DB_TYPE: postgresdb
      DB_POSTGRESDB_HOST: postgres
      DB_POSTGRESDB_PORT: 5432
      DB_POSTGRESDB_DATABASE: n8n
      DB_POSTGRESDB_USER: n8n
      DB_POSTGRESDB_PASSWORD: n8n_password
      # Configuración general
      N8N_HOST: localhost
      N8N_PORT: 5678
      N8N_PROTOCOL: http
      WEBHOOK_URL: http://localhost:5678/
      # Autenticación
      N8N_BASIC_AUTH_ACTIVE: "true"
      N8N_BASIC_AUTH_USER: admin
      N8N_BASIC_AUTH_PASSWORD: password_seguro
      # Claves API (se inyectan como credenciales en la UI, pero
      # también pueden definirse aquí para testing)
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      - postgres

volumes:
  postgres_data:
  n8n_data:
```

```bash
# Arrancar el stack
docker compose up -d

# Ver logs
docker compose logs -f n8n
```

Accede a `http://localhost:5678`, crea la cuenta de administrador y ya puedes empezar a construir workflows.

### Configurar credenciales de IA en la UI

1. Menú lateral → **Credentials** → **Add credential**
2. Busca "Anthropic" o "OpenAI"
3. Pega tu API key
4. Guarda — quedará disponible en todos los nodos de IA del tenant

---

## 3. Nodos de IA disponibles

n8n incluye una sección **"Advanced AI"** en el panel de nodos con los siguientes componentes:

### Nodos de modelos de lenguaje (LLM)

| Nodo | Proveedor | Modelos disponibles |
|---|---|---|
| **Anthropic Chat Model** | Anthropic | claude-opus-4-5, claude-sonnet-4-6, claude-haiku-3-5 |
| **OpenAI Chat Model** | OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| **Groq Chat Model** | Groq | llama3-70b, mixtral-8x7b |
| **Ollama Chat Model** | Ollama (local) | cualquier modelo local |
| **HuggingFace Inference** | HuggingFace | modelos de la Hub por API |
| **Azure OpenAI** | Azure | deployments propios |

### Nodos de embeddings

| Nodo | Uso |
|---|---|
| **Embeddings OpenAI** | Vectorizar texto con text-embedding-3-small/large |
| **Embeddings Anthropic** | Próximamente en roadmap |
| **Embeddings Ollama** | Embeddings locales (nomic-embed-text, etc.) |
| **Embeddings HuggingFace** | sentence-transformers por API |

### Nodos de memoria

| Nodo | Comportamiento |
|---|---|
| **Simple Memory** | Buffer en memoria, se resetea al reiniciar |
| **Window Buffer Memory** | Últimas N conversaciones por sesión |
| **Redis Chat Memory** | Persistencia en Redis por `sessionId` |
| **Postgres Chat Memory** | Persistencia en PostgreSQL |
| **Motorhead Memory** | Servicio externo de memoria semántica |

### Nodos de vectores y documentos

- **Vector Store Retriever** — conecta con Pinecone, Qdrant, Supabase, pgvector
- **Document Loaders** — PDF, CSV, JSON, HTML, Notion, GitHub
- **Text Splitters** — Recursive, Token, Markdown-aware
- **Embeddings + Vector Store** — indexar documentos en una sola operación

### Nodo orquestador

- **AI Agent** — combina LLM + tools + memoria. Es el núcleo de los agentes conversacionales en n8n.

---

## 4. Workflow completo: webhook → LLM → BD → respuesta

Este workflow recibe una petición HTTP con un texto, lo clasifica con Claude, guarda el resultado en PostgreSQL y devuelve la respuesta al cliente.

### Estructura del workflow en JSON

Puedes importar este JSON directamente en n8n: **Workflows → Import from file**.

```json
{
  "name": "Clasificador de texto con Claude",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "clasificar",
        "responseMode": "responseNode",
        "options": {}
      },
      "id": "webhook-entrada",
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "model": "claude-haiku-3-5-20241022",
        "messages": {
          "values": [
            {
              "role": "user",
              "content": "=Clasifica el siguiente texto en una de estas categorías: [consulta, queja, felicitación, solicitud, otro]. Responde ÚNICAMENTE con el nombre de la categoría en minúsculas.\n\nTexto: {{ $json.body.texto }}"
            }
          ]
        },
        "options": {
          "maxTokens": 20,
          "temperature": 0
        }
      },
      "id": "claude-clasificador",
      "name": "Claude - Clasificar",
      "type": "@n8n/n8n-nodes-langchain.lmChat",
      "position": [460, 300]
    },
    {
      "parameters": {
        "operation": "insert",
        "schema": "public",
        "table": "clasificaciones",
        "columns": "texto,categoria,timestamp",
        "values": "={{ $('Webhook').item.json.body.texto }},={{ $json.content[0].text }},={{ $now }}"
      },
      "id": "postgres-guardar",
      "name": "PostgreSQL - Guardar",
      "type": "n8n-nodes-base.postgres",
      "position": [680, 300]
    },
    {
      "parameters": {
        "respondWith": "json",
        "responseBody": "={\n  \"categoria\": \"{{ $('Claude - Clasificar').item.json.content[0].text }}\",\n  \"id\": {{ $json.id }},\n  \"procesado\": true\n}"
      },
      "id": "respuesta",
      "name": "Responder al cliente",
      "type": "n8n-nodes-base.respondToWebhook",
      "position": [900, 300]
    }
  ],
  "connections": {
    "Webhook": {
      "main": [[{"node": "Claude - Clasificar", "type": "main", "index": 0}]]
    },
    "Claude - Clasificar": {
      "main": [[{"node": "PostgreSQL - Guardar", "type": "main", "index": 0}]]
    },
    "PostgreSQL - Guardar": {
      "main": [[{"node": "Responder al cliente", "type": "main", "index": 0}]]
    }
  }
}
```

### Script SQL para crear la tabla

```sql
CREATE TABLE clasificaciones (
    id          SERIAL PRIMARY KEY,
    texto       TEXT NOT NULL,
    categoria   VARCHAR(50) NOT NULL,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_clasificaciones_categoria ON clasificaciones(categoria);
```

### Prueba del webhook

```python
import httpx

# Probar el workflow activado en n8n
respuesta = httpx.post(
    "http://localhost:5678/webhook/clasificar",
    json={"texto": "Quiero cancelar mi suscripción, llevan dos semanas sin responderme"},
    auth=("admin", "password_seguro"),  # solo en entornos de test
)

print(respuesta.json())
# {"categoria": "queja", "id": 1, "procesado": true}
```

### Workflow con lógica condicional (Switch node)

Tras la clasificación, puedes enrutar a distintos flujos según la categoría:

```
Webhook
  └─> Claude (clasificar)
        └─> Switch node
              ├─ "queja"       → Crear ticket en Jira + email urgente
              ├─ "consulta"    → Claude (respuesta automática) → Email cliente
              ├─ "felicitación"→ Guardar en CRM + notificar equipo
              └─ "otro"        → Cola revisión humana
```

El **Switch node** en n8n evalúa la salida del nodo anterior con expresiones:

```
{{ $json.content[0].text === "queja" }}
```

---

## 5. AI Agent node: tool use y memoria

El nodo **AI Agent** de n8n implementa el patrón ReAct: el LLM decide qué herramienta usar, la ejecuta y observa el resultado hasta llegar a la respuesta final.

### Configuración básica del AI Agent

```
AI Agent
  ├── Chat Model: Anthropic Chat Model (claude-sonnet-4-6)
  ├── Memory:     Window Buffer Memory (últimas 10 conversaciones)
  └── Tools:
        ├── HTTP Request tool   → consultar APIs externas
        ├── Code tool           → ejecutar JavaScript
        ├── Postgres tool       → query a base de datos
        └── Send Email tool     → enviar emails
```

### Ejemplo: agente de soporte con acceso a BD

El agente puede responder preguntas sobre pedidos consultando la base de datos directamente.

**System prompt del agente:**

```
Eres un asistente de soporte al cliente de TiendaXYZ.
Tienes acceso a herramientas para consultar el estado de pedidos.
Responde siempre en español, de forma amable y concisa.
Si no encuentras información, díselo al cliente honestamente.
No inventes información sobre pedidos.
```

**Tool: Consultar pedido (nodo PostgreSQL configurado como tool)**

```sql
-- Query parametrizada que el agente genera dinámicamente
SELECT
    p.id,
    p.estado,
    p.fecha_pedido,
    p.fecha_envio,
    p.transportista,
    p.numero_seguimiento,
    c.nombre AS cliente
FROM pedidos p
JOIN clientes c ON p.cliente_id = c.id
WHERE p.id = $1
  AND c.email = $2;
```

**Configuración del tool en n8n:**

```json
{
  "toolName": "consultar_pedido",
  "description": "Consulta el estado de un pedido. Requiere el ID del pedido y el email del cliente para verificar ownership.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "pedido_id": {"type": "integer", "description": "ID numérico del pedido"},
      "email_cliente": {"type": "string", "description": "Email del cliente para verificación"}
    },
    "required": ["pedido_id", "email_cliente"]
  }
}
```

### Memoria de conversación con Redis

Para mantener el contexto entre sesiones distintas (p.ej. un chat web que se reconecta):

```
AI Agent → Redis Chat Memory
  Configuración:
    - Session ID: ={{ $('Webhook').item.json.body.session_id }}
    - Window Size: 20   (últimos 20 intercambios)
    - TTL: 86400        (24 horas en segundos)
```

Esto permite que el agente recuerde conversaciones anteriores del mismo usuario aunque el workflow haya terminado.

### Invocar el agente desde Python (para testing)

```python
import httpx
import json

# El workflow expone el agente via webhook
def chat_con_agente(mensaje: str, session_id: str) -> str:
    """Envía un mensaje al agente n8n y devuelve la respuesta."""
    url = "http://localhost:5678/webhook/agente-soporte"

    respuesta = httpx.post(
        url,
        json={
            "mensaje": mensaje,
            "session_id": session_id,
        },
        timeout=30,
    )
    respuesta.raise_for_status()
    return respuesta.json().get("respuesta", "")


# Prueba de conversación con memoria
session = "usuario-123-sesion-abc"

print(chat_con_agente("Hola, tengo el pedido #5821", session))
# "Hola! Déjame consultar ese pedido. ¿Me puedes confirmar tu email?"

print(chat_con_agente("Mi email es user@ejemplo.com", session))
# "Tu pedido #5821 está en camino con DHL, número de seguimiento ES123456789. 
#  Se enviará el 2025-01-15. ¿Necesitas algo más?"
```

---

## 6. Casos prácticos

### Caso 1: Clasificador de emails con respuesta automática

```
Trigger: Gmail - nuevo email
  └─> Code node (extraer remitente, asunto, cuerpo)
        └─> Claude (clasificar: urgente/normal/spam/oportunidad)
              ├─ "urgente" → Gmail (respuesta inmediata) + Slack (alerta)
              ├─ "oportunidad" → HubSpot (crear lead) + Claude (borrador respuesta)
              ├─ "spam" → Gmail (archivar)
              └─ "normal" → Airtable (registrar) + Claude (respuesta automática)
```

**Prompt de clasificación + extracción:**

```
Analiza el siguiente email y devuelve JSON con esta estructura exacta:
{
  "categoria": "urgente|normal|spam|oportunidad",
  "urgencia": 1-5,
  "resumen": "máximo 2 frases",
  "requiere_respuesta": true|false,
  "sentimiento": "positivo|neutro|negativo",
  "etiquetas": ["tag1", "tag2"]
}

De: {{ $json.from }}
Asunto: {{ $json.subject }}
Cuerpo:
{{ $json.body }}

Devuelve SOLO el JSON, sin texto adicional.
```

### Caso 2: Extractor de datos de facturas (PDF)

```
Trigger: Google Drive - nuevo fichero en carpeta "facturas"
  └─> Google Drive node (descargar PDF como binario)
        └─> Extract from File node (extraer texto del PDF)
              └─> Claude (extraer datos estructurados)
                    └─> Google Sheets (añadir fila)
                          └─> Slack (notificar contabilidad)
```

**Prompt para extracción:**

```
Extrae los datos de la siguiente factura y devuelve JSON con esta estructura:
{
  "numero_factura": "...",
  "fecha": "YYYY-MM-DD",
  "proveedor": {
    "nombre": "...",
    "cif": "...",
    "direccion": "..."
  },
  "cliente": {
    "nombre": "...",
    "cif": "..."
  },
  "lineas": [
    {"descripcion": "...", "cantidad": 0, "precio_unitario": 0.0, "total": 0.0}
  ],
  "subtotal": 0.0,
  "iva_porcentaje": 0,
  "iva_importe": 0.0,
  "total": 0.0,
  "moneda": "EUR"
}

Texto de la factura:
{{ $json.text }}

Si no encuentras un campo, usa null. Devuelve SOLO el JSON.
```

### Caso 3: Generador de respuestas para reseñas

```
Trigger: Typeform webhook (nueva reseña de cliente)
  └─> Claude (generar respuesta personalizada)
        └─> HTTP Request (publicar en Trustpilot via API)
              └─> Airtable (registrar reseña + respuesta + puntuación)
```

**Prompt:**

```
Eres el responsable de atención al cliente de {{ $json.empresa }}.
Escribe una respuesta profesional, cálida y personalizada para la siguiente reseña.

Puntuación: {{ $json.puntuacion }}/5 estrellas
Nombre del cliente: {{ $json.nombre }}
Reseña:
{{ $json.texto_reseña }}

Instrucciones:
- Si la puntuación es 4-5: agradece y refuerza los puntos positivos mencionados
- Si la puntuación es 1-3: disculpa sinceramente, ofrece solución concreta, incluye contacto directo
- Máximo 150 palabras
- No uses frases genéricas como "nos alegra que..."
- Firma como "Equipo de {{ $json.empresa }}"
```

---

## 7. Self-hosted vs n8n Cloud

| Aspecto | Self-hosted (Docker) | n8n Cloud |
|---|---|---|
| **Precio** | Solo infraestructura (~5-20 €/mes en VPS) | Desde 20 €/mes (2.500 ejecuciones) |
| **Control de datos** | Total — datos en tu servidor | n8n almacena logs y datos de ejecución |
| **Mantenimiento** | Tú gestionas actualizaciones y backups | Gestionado por n8n |
| **Escala** | Configuras workers y cola (Redis+Bull) | Auto-scaling incluido |
| **Velocidad de setup** | 15 min con Docker Compose | 2 min (registro + acceso) |
| **Credenciales sensibles** | Bajo tu control total | Cifradas en infraestructura de n8n |
| **Webhooks** | Necesitas dominio público o ngrok | URLs públicas automáticas |
| **Ideal para** | Empresas con requisitos de datos, equipos técnicos | Startups, equipos pequeños, pruebas |

### Configurar n8n self-hosted con HTTPS (Nginx + Certbot)

```nginx
# /etc/nginx/sites-available/n8n
server {
    listen 80;
    server_name n8n.tudominio.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name n8n.tudominio.com;

    ssl_certificate     /etc/letsencrypt/live/n8n.tudominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/n8n.tudominio.com/privkey.pem;

    location / {
        proxy_pass         http://localhost:5678;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 86400;  # Necesario para SSE/streaming
    }
}
```

```bash
# Obtener certificado SSL
certbot --nginx -d n8n.tudominio.com

# Actualizar WEBHOOK_URL en docker-compose.yml
# WEBHOOK_URL: https://n8n.tudominio.com/
docker compose up -d
```

### Escalar con workers (para cargas altas)

```yaml
# docker-compose.yml — configuración multi-worker
services:
  n8n-main:
    image: docker.n8n.io/n8nio/n8n
    environment:
      EXECUTIONS_MODE: queue
      QUEUE_BULL_REDIS_HOST: redis
      N8N_CONCURRENCY_PRODUCTION_LIMIT: 1
    command: n8n start

  n8n-worker:
    image: docker.n8n.io/n8nio/n8n
    environment:
      EXECUTIONS_MODE: queue
      QUEUE_BULL_REDIS_HOST: redis
    command: n8n worker
    deploy:
      replicas: 3  # 3 workers paralelos

  redis:
    image: redis:7-alpine
```

---

## 8. Extensiones sugeridas

| Extensión | Descripción | Nodos involucrados |
|---|---|---|
| **RAG en n8n** | Indexar documentos propios y hacer preguntas | Document Loader + Embeddings + Vector Store + AI Agent |
| **Cron + batch** | Procesar lotes de registros cada hora | Schedule Trigger + Split In Batches + LLM |
| **Error handling** | Capturar fallos y notificar por Slack | Error Trigger + Slack node |
| **Versionado de workflows** | Git sync para backup y CI/CD | n8n Cloud o plugin community `n8n-nodes-github` |
| **Testing de workflows** | Ejecutar en modo dry-run con datos ficticios | Modo manual + fixtures JSON |

---

**Siguiente:** [02 — Make.com y Zapier con IA](./02-make-zapier-ia.md)
