# Workflows Avanzados: Sub-workflows, Webhooks y Producción

## Arquitectura de workflows a escala

Cuando los workflows crecen, la modularidad y la resiliencia son esenciales.

```
EVOLUCIÓN NATURAL DE WORKFLOWS
────────────────────────────────
1 workflow monolítico
        ↓
Separar en workflow principal + sub-workflows
        ↓
Añadir manejo de errores + reintentos
        ↓
Monitorización + alertas
        ↓
Control de versiones + CI/CD
```

## Sub-workflows: modularidad y reutilización

### Sub-workflow: "Llamar a Claude"

Crea un sub-workflow reutilizable que todos los demás workflows invocan:

```json
{
  "nombre": "SW_LlamarClaude",
  "trigger": "Execute Workflow Trigger",
  "inputs_esperados": {
    "prompt": "string",
    "system": "string (opcional)",
    "modelo": "string (opcional, default: claude-haiku-4-5)",
    "max_tokens": "number (opcional, default: 500)"
  },
  "output": {
    "texto": "string",
    "tokens_input": "number",
    "tokens_output": "number",
    "coste_usd": "number"
  }
}
```

```javascript
// Nodo Code dentro del sub-workflow SW_LlamarClaude
const { prompt, system, modelo, max_tokens } = $input.first().json;

const modeloFinal = modelo || 'claude-haiku-4-5-20251001';
const maxTokensFinal = max_tokens || 500;

const bodyPeticion = {
  model: modeloFinal,
  max_tokens: maxTokensFinal,
  messages: [{ role: 'user', content: prompt }]
};

if (system) {
  bodyPeticion.system = system;
}

const resp = await $http.request({
  method: 'POST',
  url: 'https://api.anthropic.com/v1/messages',
  headers: {
    'x-api-key': $env.ANTHROPIC_API_KEY,
    'anthropic-version': '2023-06-01',
    'content-type': 'application/json'
  },
  body: bodyPeticion
});

const precios = {
  'claude-haiku-4-5-20251001': { input: 0.80, output: 4.00 },
  'claude-sonnet-4-6': { input: 3.00, output: 15.00 }
};
const p = precios[modeloFinal] || precios['claude-haiku-4-5-20251001'];
const coste = (resp.body.usage.input_tokens * p.input + resp.body.usage.output_tokens * p.output) / 1_000_000;

return {
  texto: resp.body.content[0].text,
  tokens_input: resp.body.usage.input_tokens,
  tokens_output: resp.body.usage.output_tokens,
  coste_usd: Math.round(coste * 100000) / 100000
};
```

### Invocar el sub-workflow desde otro workflow

```json
{
  "nodo": "Execute Workflow",
  "workflow_id": "ID_DE_SW_LlamarClaude",
  "datos_a_pasar": {
    "prompt": "{{ 'Clasifica este email: ' + $json.subject + ' ' + $json.body }}",
    "max_tokens": 200
  }
}
```

## Webhooks: recibir eventos externos

### Webhook de entrada (n8n como receptor)

```javascript
// n8n expone automáticamente: https://tu-n8n.com/webhook/mi-endpoint
// Configurar en el nodo "Webhook":
// - HTTP Method: POST
// - Path: /procesar-documento
// - Authentication: Header Auth (recomendado)

// Validar que la petición viene de una fuente autorizada
const headers = $input.first().headers;
const body = $input.first().json;

// Verificar secret compartido
if (headers['x-webhook-secret'] !== $env.WEBHOOK_SECRET) {
  throw new Error('Webhook no autorizado');
}

// Validar que tiene los campos requeridos
if (!body.tipo || !body.contenido) {
  throw new Error('Campos requeridos: tipo, contenido');
}

return {
  tipo: body.tipo,
  contenido: body.contenido,
  origen: body.origen || 'desconocido',
  recibido_en: new Date().toISOString(),
  valido: true
};
```

### Respuesta inmediata + procesamiento asíncrono

```javascript
// Patrón: responder 200 inmediatamente, procesar en background

// Nodo 1: Webhook (responde inmediatamente)
// → "Respond to Webhook" nodo con { status: "recibido", id: "..." }

// Nodo 2 (en paralelo): procesamiento pesado con Claude
// → se ejecuta sin bloquear la respuesta al cliente

// En el nodo "Respond to Webhook":
return {
  status: 'recibido',
  id: $json.id || Date.now(),
  mensaje: 'Tu solicitud está siendo procesada. Recibirás notificación cuando esté lista.'
};
```

## Manejo robusto de errores

### Try-catch en nodos Code

```javascript
async function procesarConReintentos(operacion, maxReintentos = 3, espera = 1000) {
  let ultimoError;
  for (let intento = 1; intento <= maxReintentos; intento++) {
    try {
      return await operacion();
    } catch (error) {
      ultimoError = error;

      // No reintentar si es error del cliente (4xx)
      if (error.response?.status >= 400 && error.response?.status < 500) {
        throw error;
      }

      if (intento < maxReintentos) {
        await new Promise(resolve => setTimeout(resolve, espera * intento));
      }
    }
  }
  throw ultimoError;
}

// Uso con la API de Claude
const resultado = await procesarConReintentos(async () => {
  return await $http.request({
    method: 'POST',
    url: 'https://api.anthropic.com/v1/messages',
    headers: {
      'x-api-key': $env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json'
    },
    body: {
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 300,
      messages: [{ role: 'user', content: $json.prompt }]
    }
  });
});

return { respuesta: resultado.body.content[0].text };
```

### Workflow de errores centralizado

```javascript
// Este workflow se activa cuando cualquier otro workflow falla
// Configurar en Settings → Error Workflow de cada workflow

const error = $input.first().json;

const mensaje = `❌ *Fallo en workflow de IA*
*Workflow:* ${error.workflow.name}
*Nodo:* ${error.execution.lastNodeExecuted}
*Error:* ${error.execution.error.message}
*Ejecución:* ${error.execution.id}
*Timestamp:* ${new Date().toISOString()}`;

// Notificar en Slack
await $http.request({
  method: 'POST',
  url: $env.SLACK_WEBHOOK_ALERTAS,
  body: { text: mensaje }
});

// Guardar en log de errores (Google Sheets)
return {
  workflow: error.workflow.name,
  nodo: error.execution.lastNodeExecuted,
  error: error.execution.error.message,
  timestamp: new Date().toISOString(),
  execution_id: error.execution.id
};
```

## Rate limiting para llamadas a Claude desde n8n

```javascript
// Nodo Code: implementar rate limiting con variables de workflow

// n8n permite guardar estado entre ejecuciones con Static Data
const estatico = $getWorkflowStaticData('global');

const ahora = Date.now();
const ventanaMs = 60 * 1000; // 1 minuto
const maxLlamadas = 50; // por minuto

// Limpiar historial antiguo
estatico.historialLlamadas = (estatico.historialLlamadas || [])
  .filter(t => ahora - t < ventanaMs);

if (estatico.historialLlamadas.length >= maxLlamadas) {
  const esperaMs = ventanaMs - (ahora - estatico.historialLlamadas[0]);
  throw new Error(`Rate limit: espera ${Math.ceil(esperaMs / 1000)} segundos`);
}

estatico.historialLlamadas.push(ahora);

// Proceder con la llamada a Claude
const resp = await $http.request({
  method: 'POST',
  url: 'https://api.anthropic.com/v1/messages',
  headers: {
    'x-api-key': $env.ANTHROPIC_API_KEY,
    'anthropic-version': '2023-06-01',
    'content-type': 'application/json'
  },
  body: {
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 300,
    messages: [{ role: 'user', content: $json.prompt }]
  }
});

return {
  respuesta: resp.body.content[0].text,
  llamadas_en_ventana: estatico.historialLlamadas.length
};
```

## Despliegue en producción

### Docker Compose recomendado

```yaml
version: '3.8'
services:
  n8n:
    image: n8nio/n8n:latest
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=${N8N_DOMAIN}
      - N8N_PORT=5678
      - N8N_PROTOCOL=https
      - WEBHOOK_URL=https://${N8N_DOMAIN}/
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - SLACK_WEBHOOK_ALERTAS=${SLACK_WEBHOOK_ALERTAS}
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      - postgres

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=n8n
      - POSTGRES_USER=n8n
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  n8n_data:
  postgres_data:
```

### Variables de entorno (.env)

```bash
N8N_USER=admin
N8N_PASSWORD=contraseña_segura_aqui
N8N_DOMAIN=n8n.tu-empresa.com
POSTGRES_PASSWORD=password_postgres_seguro
ANTHROPIC_API_KEY=sk-ant-...
SLACK_WEBHOOK_ALERTAS=https://hooks.slack.com/services/...
WEBHOOK_SECRET=secreto_compartido_para_webhooks
```

### Backup de workflows

```bash
# Exportar todos los workflows (ejecutar en el servidor)
docker exec n8n n8n export:workflow --all --output=/home/node/.n8n/backup/workflows.json

# Restaurar
docker exec n8n n8n import:workflow --input=/home/node/.n8n/backup/workflows.json
```

## Checklist de producción para workflows de IA

```
Resiliencia:
  ✓ Error workflow configurado en cada workflow
  ✓ Reintentos automáticos para llamadas a API
  ✓ Timeout definido para nodos de larga ejecución
  ✓ Validación de inputs en webhooks

Seguridad:
  ✓ API keys en variables de entorno (nunca hardcodeadas)
  ✓ Autenticación básica activada en n8n
  ✓ Webhook secrets validados
  ✓ HTTPS con certificado válido

Monitorización:
  ✓ Alertas de error en Slack/email
  ✓ Log de ejecuciones en base de datos
  ✓ Coste por workflow registrado

Mantenimiento:
  ✓ Workflows exportados en control de versiones
  ✓ Documentación de cada workflow (descripción + propietario)
  ✓ Entorno de staging para probar cambios
```

## Recursos

- [Notebook interactivo](../notebooks/n8n-workflows/04-workflows-avanzados.ipynb)
- [n8n — Self-hosting guide](https://docs.n8n.io/hosting/)
- [n8n — Error handling](https://docs.n8n.io/flow-logic/error-handling/)
- [n8n — Sub-workflows](https://docs.n8n.io/flow-logic/subworkflows/)
