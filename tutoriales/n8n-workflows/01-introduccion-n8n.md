# Introducción a n8n con Claude

## ¿Qué es n8n y por qué combinarlo con Claude?

n8n es un orquestador de workflows de código abierto (self-hosteable).
Claude aporta el razonamiento. n8n aporta las integraciones: 400+ conectores
con Gmail, Slack, HubSpot, Notion, bases de datos, webhooks y APIs.

```
Sin n8n:             Con n8n + Claude:
┌─────────┐          ┌──────────────────────────────────────┐
│ Código  │          │ Gmail → Claude (clasifica) →          │
│ ad hoc  │    →     │ Notion (guarda) → Slack (notifica) → │
│ frágil  │          │ HubSpot (actualiza CRM)              │
└─────────┘          └──────────────────────────────────────┘
```

**Cuándo usar n8n vs código puro:**

| Caso | n8n | Código puro |
|------|-----|-------------|
| Integrar 3+ servicios SaaS | ✅ | ❌ (mucho boilerplate) |
| Lógica de negocio compleja | ❌ | ✅ |
| Equipo no técnico necesita mantenerlo | ✅ | ❌ |
| Máximo rendimiento/escala | ❌ | ✅ |
| Prototipo rápido | ✅ | ❌ |

## Conceptos clave

```
WORKFLOW     = secuencia de nodos conectados
NODO         = una acción (llamar API, transformar datos, enviar email)
TRIGGER      = nodo de inicio (webhook, cron, evento de app)
CREDENCIAL   = clave de API guardada de forma segura en n8n
RUN          = una ejecución del workflow
EXPRESSION   = código JavaScript mínimo para transformar datos {{ $json.field }}
```

## Primer workflow: Email → Claude → Slack

Este workflow recibe emails, los clasifica con Claude y notifica en Slack.

### Nodo 1: Gmail Trigger (o Webhook para pruebas)

```json
{
  "node": "Gmail Trigger",
  "tipo": "trigger",
  "configuracion": {
    "evento": "messageReceived",
    "filtro": "is:unread label:inbox"
  },
  "output_ejemplo": {
    "from": "cliente@empresa.com",
    "subject": "Urgente: fallo en producción",
    "snippet": "El sistema está caído desde hace 30 minutos..."
  }
}
```

### Nodo 2: HTTP Request → Claude API

```json
{
  "node": "HTTP Request",
  "tipo": "accion",
  "configuracion": {
    "method": "POST",
    "url": "https://api.anthropic.com/v1/messages",
    "headers": {
      "x-api-key": "{{ $credentials.anthropicApiKey }}",
      "anthropic-version": "2023-06-01",
      "content-type": "application/json"
    },
    "body": {
      "model": "claude-haiku-4-5-20251001",
      "max_tokens": 200,
      "messages": [{
        "role": "user",
        "content": "Clasifica este email. Responde SOLO con JSON: {\"categoria\": \"soporte_urgente|soporte_normal|comercial|spam\", \"prioridad\": \"alta|media|baja\", \"resumen\": \"1 frase\"}\n\nAsunto: {{ $json.subject }}\nMensaje: {{ $json.snippet }}"
      }]
    }
  }
}
```

### Nodo 3: Code Node — parsear respuesta de Claude

```javascript
// En el nodo "Code" de n8n (JavaScript)
const respuestaClaude = JSON.parse($input.first().json.body.content[0].text);
const emailOriginal = $('Gmail Trigger').first().json;

return {
  categoria: respuestaClaude.categoria,
  prioridad: respuestaClaude.prioridad,
  resumen: respuestaClaude.resumen,
  from: emailOriginal.from,
  subject: emailOriginal.subject,
  es_urgente: respuestaClaude.prioridad === 'alta'
};
```

### Nodo 4: IF — ¿Es urgente?

```json
{
  "node": "IF",
  "condicion": "{{ $json.es_urgente }} === true",
  "rama_verdadera": "Slack — Canal Urgente",
  "rama_falsa": "Slack — Canal General"
}
```

### Nodo 5: Slack

```json
{
  "node": "Slack",
  "accion": "postMessage",
  "canal": "#soporte-urgente",
  "mensaje": "🚨 Email urgente de {{ $json.from }}\n*{{ $json.subject }}*\n{{ $json.resumen }}"
}
```

## Usar Claude como HTTP Request desde n8n (patrón reutilizable)

```javascript
// Función helper para llamar a Claude desde nodos Code de n8n
// Pegar en cualquier nodo Code que necesite IA

async function llamarClaude(prompt, modelo = 'claude-haiku-4-5-20251001', maxTokens = 500) {
  const response = await $http.request({
    method: 'POST',
    url: 'https://api.anthropic.com/v1/messages',
    headers: {
      'x-api-key': $env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json'
    },
    body: {
      model: modelo,
      max_tokens: maxTokens,
      messages: [{ role: 'user', content: prompt }]
    }
  });
  return response.body.content[0].text;
}

// Uso:
const clasificacion = await llamarClaude(
  `Clasifica este email: ${$json.subject}\n${$json.body}`,
  'claude-haiku-4-5-20251001',
  200
);

return { clasificacion: JSON.parse(clasificacion) };
```

## Credenciales: cómo guardar la API key de Anthropic en n8n

```
En la interfaz de n8n:
1. Settings → Credentials → New Credential
2. Tipo: "HTTP Request Auth" o "Header Auth"
3. Name: "Anthropic API"
4. Header: "x-api-key"
5. Value: tu API key de Anthropic

Alternativa con variable de entorno (recomendado en producción):
N8N_DEFAULT_BINARY_DATA_MODE=default
ANTHROPIC_API_KEY=sk-ant-...

En el nodo HTTP Request:
Header: x-api-key: {{ $env.ANTHROPIC_API_KEY }}
```

## Manejo de errores básico

```json
{
  "configuracion_workflow": {
    "errorWorkflow": "ID_DEL_WORKFLOW_DE_ERRORES",
    "saveDataErrorExecution": "all",
    "saveDataSuccessExecution": "none"
  }
}
```

```javascript
// En el workflow de errores, nodo Code:
const error = $input.first().json;

// Notificar al equipo
await $http.request({
  method: 'POST',
  url: $env.SLACK_WEBHOOK_URL,
  body: {
    text: `❌ Error en workflow "${error.workflow.name}"\nNodo: ${error.execution.lastNodeExecuted}\nError: ${error.execution.error.message}`
  }
});
```

## Recursos

- [Notebook interactivo](../notebooks/n8n-workflows/01-introduccion-n8n.ipynb)
- [n8n Documentación oficial](https://docs.n8n.io)
- [n8n Cloud (sin instalación)](https://n8n.io/cloud/)
- [Templates de n8n para IA](https://n8n.io/workflows/?categories=AI)
