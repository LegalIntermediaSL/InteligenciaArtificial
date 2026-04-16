# 02 — Make.com y Zapier con IA

> **Bloque:** Automatización con IA · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [Make.com: escenarios, módulos y variables](#1-makecom-escenarios-módulos-y-variables)
2. [Integración Make + OpenAI/Claude](#2-integración-make--openaiclaude)
3. [Zapier: zaps, triggers y acciones con IA](#3-zapier-zaps-triggers-y-acciones-con-ia)
4. [Caso práctico Make: Typeform → Claude → Notion → Slack](#4-caso-práctico-make-typeform--claude--notion--slack)
5. [Caso práctico Zapier: Gmail → OpenAI → Google Sheets](#5-caso-práctico-zapier-gmail--openai--google-sheets)
6. [Guía de decisión: Make vs Zapier vs n8n](#6-guía-de-decisión-make-vs-zapier-vs-n8n)

---

## 1. Make.com: escenarios, módulos y variables

Make (anteriormente Integromat) organiza la automatización en **escenarios**: grafos de módulos conectados que se ejecutan de izquierda a derecha cuando se cumple un trigger.

### Conceptos clave

| Concepto | Descripción |
|---|---|
| **Escenario** | Flujo completo de automatización (equivale a un workflow) |
| **Módulo** | Unidad mínima de acción (obtener, crear, actualizar, transformar) |
| **Trigger** | Módulo inicial que activa el escenario (webhook, horario, evento) |
| **Bundle** | Paquete de datos que fluye entre módulos (equivale a un item/registro) |
| **Iterator** | Desglosa un array en bundles individuales para procesarlos uno a uno |
| **Aggregator** | Reúne bundles en un solo array o texto |
| **Router** | Bifurca el flujo según condiciones (como un if/else visual) |
| **Filtro** | Permite continuar el flujo solo si se cumple una condición |

### Expresiones y variables en Make

Make usa doble llave `{{}}` para referencias y tiene funciones propias:

```
# Acceso a datos de módulos anteriores
{{1.nombre}}           → campo "nombre" del módulo 1
{{2.items[].precio}}   → campo "precio" de cada item del módulo 2

# Funciones de texto
{{upper(1.nombre)}}                    → mayúsculas
{{trim(1.descripcion)}}               → eliminar espacios
{{substring(1.texto; 0; 100)}}        → primeros 100 caracteres
{{replace(1.texto; "antiguo"; "nuevo")}} → reemplazar

# Funciones matemáticas
{{round(1.precio * 1.21; 2)}}         → precio con IVA, 2 decimales

# Funciones de fecha
{{formatDate(now; "DD/MM/YYYY")}}     → fecha actual formateada
{{addDays(1.fecha_inicio; 30)}}       → fecha + 30 días

# Funciones de array
{{length(1.items)}}                   → número de elementos
{{first(1.items)}}                    → primer elemento
{{join(1.etiquetas; ", ")}}           → array a string
```

### Iteradores y agregadores

El patrón **iterator → módulos → aggregator** es fundamental para procesar listas con IA:

```
Obtener lista de artículos (módulo 1)
  └─> Iterator (desglosa el array en bundles)
        └─> OpenAI: resumir cada artículo (módulo 3)
              └─> Aggregator (reúne todos los resúmenes en un array)
                    └─> Enviar resumen diario por email
```

**Configuración del Array Aggregator:**

```
Módulo origen: Iterator (módulo 2)
Campo de destino: resúmenes
Expresión: {{3.choices[].message.content}}
Tipo resultado: Array de strings
```

---

## 2. Integración Make + OpenAI/Claude

### Módulo OpenAI en Make

Make tiene un módulo nativo **"OpenAI (DALL-E, ChatGPT & Whisper)"** con las operaciones:

- **Create a completion** — completado de texto (GPT-3.5/4)
- **Create a chat completion** — conversación con historial
- **Create a transcription** — transcripción de audio con Whisper
- **Create an image** — generación con DALL-E

**Configuración de "Create a chat completion":**

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "Eres un asistente que clasifica y resume textos en español."
    },
    {
      "role": "user",
      "content": "{{1.texto}}"
    }
  ],
  "max_tokens": 500,
  "temperature": 0.3
}
```

### Módulo Anthropic (Claude) vía HTTP

Make no tiene módulo nativo de Anthropic, pero puedes usar el módulo **HTTP > Make a request**:

**Configuración del módulo HTTP:**

```
URL: https://api.anthropic.com/v1/messages
Método: POST
Headers:
  x-api-key: {{tu_anthropic_api_key}}
  anthropic-version: 2023-06-01
  content-type: application/json

Body (JSON):
{
  "model": "claude-haiku-3-5-20241022",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "{{1.texto_a_procesar}}"
    }
  ]
}
```

**Extraer la respuesta del módulo HTTP:**

```
# La respuesta de Claude tiene esta estructura:
{{módulo_http.content[].text}}

# En Make, con un subpath:
{{5.content[1].text}}   → primer bloque de texto de la respuesta
```

### Técnica: JSON parsing con Make

Cuando el LLM devuelve JSON embebido en texto:

```
Módulo: Parse JSON (Tools > JSON > Parse JSON)
Input: {{5.content[1].text}}

# Si Claude devuelve ```json {...} ```, primero limpiar con:
# Replace: quitar "```json" y "```"
{{replace(replace(5.content[1].text; "```json"; ""); "```"; "")}}
```

### Manejo de errores en Make

```
Configuración de error handling en cada módulo:
  ├── Break: detiene el escenario y marca como error
  ├── Ignore: continúa con el siguiente bundle
  ├── Resume: continúa con valor por defecto
  └── Rollback: deshace operaciones transaccionales

Para LLMs: usa "Resume" con valor por defecto vacío
y añade un filtro posterior que detecte respuestas vacías.
```

---

## 3. Zapier: zaps, triggers y acciones con IA

Zapier organiza la automatización en **Zaps**: secuencias lineales de **trigger → acciones**. Es más simple que Make (sin bifurcaciones nativas en el plan básico) pero tiene el ecosistema de integraciones más amplio del mercado.

### Módulos de IA en Zapier

**AI by Zapier** — módulo nativo con operaciones:

| Operación | Descripción |
|---|---|
| **Understand a text** | Extrae información estructurada de texto libre |
| **Write from a template** | Genera texto rellenando un template con IA |
| **Summarize text** | Resume texto largo |
| **Translate text** | Traducción automática |
| **Classify text** | Categoriza texto en clases definidas |
| **Extract from text** | Extrae campos concretos (nombres, fechas, emails…) |

**OpenAI en Zapier** — módulo oficial con:
- Send Prompt (completado simple)
- Conversation (con historial)
- Generate Image (DALL-E)
- Create Transcription (Whisper)

### Configuración de "Classify text" con AI by Zapier

```
Trigger: New Email in Gmail

Acción 1: AI by Zapier - Classify text
  Text: {{Body Plain}}
  Categories: urgente, consulta, queja, spam, oferta

Acción 2 (condicional con Paths):
  If clasificación = "urgente" → Send Slack message
  If clasificación = "queja"   → Create Zendesk ticket
  If clasificación = "oferta"  → Add row in Google Sheets
```

### Usar OpenAI en Zapier con prompt personalizado

```
Acción: OpenAI - Send Prompt

Model: gpt-4o-mini
Prompt:
Analiza el siguiente email de soporte y extrae:
1. El problema principal (1 frase)
2. La urgencia (alta/media/baja)
3. Si requiere seguimiento (sí/no)

Devuelve el resultado en formato:
PROBLEMA: ...
URGENCIA: ...
SEGUIMIENTO: ...

Email:
{{Body Plain}}

System: Eres un asistente de soporte técnico especializado.
Temperature: 0.2
Maximum Tokens: 200
```

### Paths en Zapier (bifurcación)

Zapier permite bifurcaciones con **Paths** (plan Team o superior):

```
Trigger: New Form Submission (Typeform)
  └─> OpenAI: Classify sentiment
        ├─ Path A (si sentiment = "positivo"):
        │    └─> Add to Airtable (tabla: feedback_positivo)
        │          └─> Send Slack celebration message
        └─ Path B (si sentiment = "negativo"):
              └─> Create Intercom conversation
                    └─> Notify manager by email
```

---

## 4. Caso práctico Make: Typeform → Claude → Notion → Slack

**Escenario:** cuando alguien rellena un formulario de solicitud de servicio, Claude analiza la solicitud, extrae los requisitos clave, crea una página en Notion con el briefing estructurado y notifica al equipo en Slack.

### Estructura del escenario Make

```
[1] Typeform - Watch Responses
      ↓
[2] HTTP - Anthropic API (analizar solicitud)
      ↓
[3] Tools - Parse JSON (extraer datos de Claude)
      ↓
[4] Notion - Create Page (crear briefing)
      ↓
[5] Slack - Create a Message (notificar equipo)
```

### Módulo 1: Typeform trigger

```
Módulo: Typeform > Watch Responses
Form ID: <ID del formulario>
Webhook: Make genera la URL automáticamente
```

Campos del formulario que usaremos:
- `nombre_empresa` — texto corto
- `descripcion_proyecto` — texto largo
- `presupuesto` — número o desplegable
- `fecha_inicio` — fecha
- `contacto_email` — email

### Módulo 2: HTTP Request a Anthropic

```
URL: https://api.anthropic.com/v1/messages
Método: POST

Headers:
  x-api-key: <ANTHROPIC_API_KEY>
  anthropic-version: 2023-06-01
  content-type: application/json

Body:
{
  "model": "claude-haiku-3-5-20241022",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Analiza la siguiente solicitud de servicio y devuelve un JSON con esta estructura exacta:\n{\n  \"titulo\": \"título conciso del proyecto (max 60 chars)\",\n  \"resumen\": \"resumen ejecutivo en 2-3 frases\",\n  \"requisitos\": [\"requisito 1\", \"requisito 2\", \"...\"],\n  \"tecnologias_sugeridas\": [\"tech1\", \"tech2\"],\n  \"complejidad\": \"baja|media|alta\",\n  \"estimacion_dias\": número,\n  \"preguntas_aclaratorias\": [\"pregunta 1\", \"pregunta 2\"]\n}\n\nEmpresa: {{1.nombre_empresa}}\nProyecto: {{1.descripcion_proyecto}}\nPresupuesto: {{1.presupuesto}}\nFecha inicio deseada: {{1.fecha_inicio}}\n\nDevuelve SOLO el JSON, sin texto adicional."
    }
  ]
}
```

### Módulo 3: Parse JSON

```
Módulo: Tools > JSON > Parse JSON
Input: {{2.content[1].text}}

# Si hay bloques markdown, limpiar primero con un módulo Text > Replace
# buscar: ```json y ```
# reemplazar por: (vacío)
```

### Módulo 4: Notion - Create Page

```
Módulo: Notion > Create a Page
Database ID: <ID de la base de datos de proyectos>

Propiedades de la página:
  Nombre:        {{3.titulo}}
  Estado:        "Nueva solicitud"
  Empresa:       {{1.nombre_empresa}}
  Complejidad:   {{3.complejidad}}
  Estimación:    {{3.estimacion_dias}} días
  Email contacto: {{1.contacto_email}}
  Fecha solicitud: {{formatDate(now; "YYYY-MM-DD")}}

Contenido (bloques):
  Heading 2: "Resumen"
  Paragraph:  {{3.resumen}}

  Heading 2: "Requisitos"
  Bulleted list: [Iterator sobre 3.requisitos]

  Heading 2: "Tecnologías sugeridas"
  Bulleted list: [Iterator sobre 3.tecnologias_sugeridas]

  Heading 2: "Preguntas aclaratorias"
  Bulleted list: [Iterator sobre 3.preguntas_aclaratorias]
```

### Módulo 5: Slack - Create Message

```
Módulo: Slack > Create a Message
Channel: #nuevos-proyectos

Text (con Block Kit):
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": "🆕 Nueva solicitud de proyecto"}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Cliente:*\n{{1.nombre_empresa}}"},
        {"type": "mrkdwn", "text": "*Complejidad:*\n{{3.complejidad}}"},
        {"type": "mrkdwn", "text": "*Estimación:*\n{{3.estimacion_dias}} días"},
        {"type": "mrkdwn", "text": "*Contacto:*\n{{1.contacto_email}}"}
      ]
    },
    {
      "type": "section",
      "text": {"type": "mrkdwn", "text": "*Resumen:*\n{{3.resumen}}"}
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {"type": "plain_text", "text": "Ver en Notion"},
          "url": "{{4.url}}"
        }
      ]
    }
  ]
}
```

---

## 5. Caso práctico Zapier: Gmail → OpenAI → Google Sheets

**Escenario:** monitorizar una bandeja de entrada de soporte, analizar cada email con GPT-4o-mini y registrar los datos estructurados en Google Sheets para reporting.

### Estructura del Zap

```
[Trigger] Gmail - New Email Matching Search
    ↓
[1] Formatter by Zapier - Text - Truncate
    ↓
[2] OpenAI - Send Prompt
    ↓
[3] Formatter by Zapier - Utilities - Line Itemizer
    ↓
[4] Google Sheets - Create Spreadsheet Row
```

### Trigger: Gmail - New Email Matching Search

```
Search string: to:soporte@tuempresa.com -from:noreply
Folder: INBOX
```

### Paso 1: Truncar el email (límite de tokens)

```
Módulo: Formatter by Zapier > Text > Truncate
Input: {{Body Plain}}
Max Length: 3000
Append Ellipsis: true
```

### Paso 2: OpenAI - Analizar email

```
Módulo: OpenAI - Send Prompt
Model: gpt-4o-mini
Temperature: 0

Prompt:
Analiza el siguiente email de soporte y extrae información estructurada.
Devuelve exactamente este formato (sin JSON, una línea por campo):

CATEGORIA: [consulta/queja/solicitud_función/bug/facturación/otro]
URGENCIA: [alta/media/baja]
SENTIMIENTO: [positivo/neutro/negativo/frustrado]
PRODUCTO: [nombre del producto mencionado o "no especificado"]
PROBLEMA_CORTO: [descripción en máximo 10 palabras]
REQUIERE_SEGUIMIENTO: [sí/no]
IDIOMA: [es/en/fr/de/otro]

Email:
De: {{From}}
Asunto: {{Subject}}
Cuerpo: {{output}}

System: Extrae la información con precisión. No añadas explicaciones.
```

### Paso 3: Formatter - Line Itemizer

```
Módulo: Formatter by Zapier > Utilities > Line Itemizer
Input: {{choices__message__content}}
Separator: :
# Esto convierte "CATEGORIA: queja" en pares clave-valor accesibles
```

### Paso 4: Google Sheets - Crear fila

```
Módulo: Google Sheets > Create Spreadsheet Row
Spreadsheet: "Soporte - Análisis IA"
Worksheet: "Emails"

Columnas:
  Fecha:              {{zap_meta_humanized_time}}
  De:                 {{From}}
  Asunto:             {{Subject}}
  Categoría:          {{line_items__CATEGORIA}}
  Urgencia:           {{line_items__URGENCIA}}
  Sentimiento:        {{line_items__SENTIMIENTO}}
  Producto:           {{line_items__PRODUCTO}}
  Problema:           {{line_items__PROBLEMA_CORTO}}
  Requiere seguimiento: {{line_items__REQUIERE_SEGUIMIENTO}}
  Idioma:             {{line_items__IDIOMA}}
  Gmail ID:           {{Message-ID}}
```

### Dashboard automático en Google Sheets

Una vez los datos fluyen a la hoja, puedes crear un dashboard con fórmulas:

```
# En una hoja "Dashboard":

# Distribución por categoría
=COUNTIF('Emails'!D:D, "queja")
=COUNTIF('Emails'!D:D, "consulta")

# Urgencia alta esta semana
=COUNTIFS('Emails'!E:E, "alta",
           'Emails'!A:A, ">="&TODAY()-7)

# % emails negativos
=COUNTIF('Emails'!F:F, "negativo") / COUNTA('Emails'!F:F)

# Tabla dinámica con QUERY
=QUERY('Emails'!A:J,
  "SELECT D, COUNT(D) WHERE A >= date '"&TEXT(TODAY()-30,"YYYY-MM-DD")&"'
   GROUP BY D ORDER BY COUNT(D) DESC",
  1)
```

---

## 6. Guía de decisión: Make vs Zapier vs n8n

### Árbol de decisión

```
¿Tienes requisitos de privacidad de datos o compliance estricto?
  └─ SÍ → n8n self-hosted (control total)
  └─ NO → continúa...

¿El equipo que lo usará tiene perfil técnico?
  └─ Alto (sabe programar) → n8n (más potente, JS nativo)
  └─ Medio → Make (visual pero flexible)
  └─ Bajo → Zapier (más simple, ecosistema más amplio)

¿Necesitas procesar datos complejos (arrays, transformaciones, lógica)?
  └─ SÍ → Make o n8n (Zapier tiene limitaciones sin Paths)
  └─ NO → Zapier (suficiente para flujos lineales)

¿El volumen de operaciones es alto (>10.000/mes)?
  └─ SÍ → n8n self-hosted (sin coste por operación)
  └─ NO → Make o Zapier (precio razonable)

¿Necesitas nodos de IA avanzados (agent, RAG, embeddings)?
  └─ SÍ → n8n (tiene todo el stack de LangChain en nodos visuales)
  └─ NO → Make o Zapier (con módulo HTTP para APIs de IA)
```

### Comparativa detallada

| Criterio | Make | Zapier | n8n (self) |
|---|---|---|---|
| **Precio base** | 9 €/mes (10.000 ops) | 19,99 $/mes (750 tareas) | ~5-15 €/mes (VPS) |
| **Integraciones** | ~1.500 apps | ~7.000 apps | ~400 nodos + HTTP |
| **Bifurcaciones** | Router nativo | Paths (plan Team) | IF node nativo |
| **Iteradores** | Iterator + Aggregator | Loop (plan avanzado) | Split In Batches |
| **Código personalizado** | No | Code by Zapier (JS/Python) | Code node (JS) |
| **Webhooks** | Incluido | Incluido | Incluido |
| **Ejecuciones en tiempo real** | Sí | Depende del plan | Sí |
| **Historial de ejecuciones** | 30 días | 7 días (plan básico) | Configurable |
| **Soporte IA nativo** | OpenAI (módulo) | AI by Zapier + OpenAI | Toda la suite LangChain |
| **API propia de Claude** | HTTP module | HTTP by Zapier | Nodo Anthropic nativo |
| **Mejor para** | Procesos de datos complejos | Máxima cobertura de SaaS | Control + IA avanzada |

### Cuándo combinar herramientas

No es obligatorio elegir solo una. Una arquitectura híbrida habitual:

```
Zapier: conectar los ~100 SaaS que no tienen API directa
  └─> Webhook a n8n: procesamiento con IA, lógica compleja
        └─> Base de datos propia: almacenamiento y estado
              └─> Make: notificaciones y reportes periódicos
```

**Ejemplo concreto:** Zapier detecta nuevo lead en LinkedIn (que n8n no puede hacer nativamente) → envía webhook a n8n → n8n enriquece con IA + CRM propio → Make envía resumen semanal a Slack.

---

**Anterior:** [01 — n8n con IA](./01-n8n-ia.md) · **Siguiente:** [03 — Pipelines de negocio con LLMs](./03-pipelines-negocio.md)
