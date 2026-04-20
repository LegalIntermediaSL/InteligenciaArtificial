# Workflows de Negocio: CRM, Email y Slack Bot

## Los workflows de negocio más rentables

```
IMPACTO VS COMPLEJIDAD
───────────────────────
Alta rentabilidad + Baja complejidad:
  ✓ Triaje automático de emails con respuesta borrador
  ✓ Actualización de CRM desde conversaciones
  ✓ Slack bot para preguntas de FAQs internas

Alta rentabilidad + Media complejidad:
  ✓ Seguimiento automático de leads con personalización IA
  ✓ Informe semanal generado y enviado automáticamente
  ✓ Onboarding de nuevos usuarios con secuencia personalizada
```

## Workflow 1: Triaje de emails con respuesta borrador

### Descripción
Gmail recibe un email → Claude lo clasifica y genera borrador → el borrador se guarda como respuesta en Gmail para que el humano lo revise y envíe.

```javascript
// Nodo Code: clasificar + generar borrador
const email = $input.first().json;

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
    max_tokens: 600,
    system: `Eres el asistente de comunicaciones de [Nombre de la empresa].
Clasificas emails y generas respuestas en tono profesional y cercano.
IMPORTANTE: Los borradores siempre son revisados por un humano antes de enviar.`,
    messages: [{
      role: 'user',
      content: `Analiza este email y genera un borrador de respuesta.

DE: ${email.from}
ASUNTO: ${email.subject}
CUERPO: ${email.body?.substring(0, 1500) || email.snippet}

Responde en JSON:
{
  "categoria": "soporte|comercial|facturacion|partnership|spam|otro",
  "prioridad": "alta|media|baja",
  "sentimiento": "positivo|neutro|negativo|urgente",
  "resumen_3_palabras": "...",
  "requiere_respuesta": true,
  "borrador_respuesta": "texto completo del borrador, en primera persona, con saludo y despedida",
  "acciones_adicionales": ["actualizar CRM", "crear ticket", etc.]
}`
    }]
  }
});

const analisis = JSON.parse(resp.body.content[0].text);
analisis.email_id = email.id;
analisis.from = email.from;
analisis.subject = email.subject;
analisis.procesado_en = new Date().toISOString();

return analisis;
```

### Nodo Switch: enrutar según categoría

```
categoria = "soporte"      → crear ticket en Zendesk/Freshdesk
categoria = "comercial"    → crear oportunidad en HubSpot
categoria = "facturacion"  → notificar a contabilidad en Slack
categoria = "spam"         → marcar como spam + no generar borrador
otros                      → guardar en Notion + notificar al equipo
```

## Workflow 2: Actualización automática de CRM desde emails

```javascript
// Extraer datos de contacto y actualizar HubSpot con IA

const email = $input.first().json;

// Paso 1: extraer datos del contacto
const extraccion = await $http.request({
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
    messages: [{
      role: 'user',
      content: `Extrae datos del contacto de este email. JSON sin markdown:
{
  "nombre": "...",
  "apellidos": "...",
  "email": "...",
  "empresa": "...",
  "cargo": "...",
  "telefono": "... o null",
  "pais": "... o null",
  "interes_principal": "descripción breve de qué quiere",
  "etapa_ciclo": "prospect|lead_calificado|oportunidad|cliente|churned"
}

Email:
De: ${email.from}
Firma y cuerpo: ${email.body?.substring(0, 800) || email.snippet}`
    }]
  }
});

const contacto = JSON.parse(extraccion.body.content[0].text);

// Paso 2: buscar si existe en HubSpot
// (esto se hace con el nodo HubSpot de n8n en el workflow visual)
// Aquí devolvemos los datos para el siguiente nodo

return {
  ...contacto,
  email_origen: email.id,
  actualizar_hubspot: true,
  nota_crm: `[Auto] Email recibido: ${email.subject}. Interés: ${contacto.interes_principal}`
};
```

## Workflow 3: Slack Bot de preguntas internas

### Arquitectura
```
Slack Event (mención al bot @AsiatenteIA o en canal #preguntas)
     ↓
Webhook n8n
     ↓
Buscar en base de conocimiento (Notion / Google Docs / URLs)
     ↓
Claude genera respuesta con el contexto encontrado
     ↓
Responder en el hilo de Slack
```

```javascript
// Nodo Code: procesar pregunta del Slack bot
const slackEvent = $input.first().json;
const pregunta = slackEvent.event?.text?.replace(/<@[^>]+>/g, '').trim();
const canal = slackEvent.event?.channel;
const usuario = slackEvent.event?.user;

// Base de conocimiento interna (en producción: buscar en Notion/Drive)
const BASE_CONOCIMIENTO = `
POLÍTICAS DE EMPRESA:
- Vacaciones: 23 días laborables al año + festivos locales
- Horario: flexible con núcleo 10h-15h
- Home office: hasta 3 días/semana con acuerdo con manager
- Gastos: reembolso con ticket en < 30 días, límite 50€ sin aprobación previa

HERRAMIENTAS:
- CRM: HubSpot (acceso todos los empleados)
- Gestión proyectos: Linear (tech), Notion (resto)
- Comunicación: Slack (trabajo) + email (externo)
- Docs: Google Workspace

PROCESOS:
- Onboarding nuevos empleados: checklist en Notion > RRHH > Onboarding
- Solicitar equipo: formulario en #it-requests
- Baja médica: notificar a manager + RRHH en el mismo día
`;

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
    max_tokens: 400,
    system: `Eres el asistente interno de la empresa. Respondes preguntas de empleados
basándote SOLO en la base de conocimiento proporcionada. Si no sabes la respuesta,
dilo claramente y sugiere a quién preguntar. Responde siempre en el idioma de la pregunta.
Sé conciso (máximo 3 párrafos) y usa formato Slack (negrita con *texto*).`,
    messages: [{
      role: 'user',
      content: `BASE DE CONOCIMIENTO:\n${BASE_CONOCIMIENTO}\n\nPREGUNTA DEL EMPLEADO:\n${pregunta}`
    }]
  }
});

return {
  canal: canal,
  usuario: usuario,
  pregunta: pregunta,
  respuesta: resp.body.content[0].text,
  thread_ts: slackEvent.event?.ts
};
```

### Responder en Slack

```json
{
  "nodo": "Slack",
  "accion": "postMessage",
  "canal": "{{ $json.canal }}",
  "texto": "{{ $json.respuesta }}",
  "thread_ts": "{{ $json.thread_ts }}"
}
```

## Workflow 4: Informe semanal automático

```javascript
// Trigger: cada lunes a las 8:00 (Cron trigger en n8n)
// Recopila datos de HubSpot + Linear + Google Analytics
// Claude genera el informe + lo envía por email

// Nodo Code final: generar informe con los datos recopilados
const datosHubSpot = $('HubSpot').first().json;   // leads, oportunidades, cierres
const datosLinear = $('Linear').first().json;       // tareas completadas, incidencias
const datosAnalytics = $('Analytics').first().json; // visitas, conversiones

const resumenDatos = `
VENTAS (semana):
- Nuevos leads: ${datosHubSpot.nuevos_leads}
- Oportunidades activas: ${datosHubSpot.oportunidades}
- Cierres: ${datosHubSpot.cierres} (${datosHubSpot.mrr_nuevo}€ MRR nuevo)

PRODUCTO (semana):
- Tareas completadas: ${datosLinear.completadas}
- Bugs cerrados: ${datosLinear.bugs_cerrados}
- Incidencias críticas: ${datosLinear.incidencias_criticas}

WEB (semana):
- Visitas: ${datosAnalytics.sesiones}
- Demos solicitadas: ${datosAnalytics.demos}
- Tasa de conversión: ${datosAnalytics.conversion_pct}%
`;

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
    max_tokens: 600,
    messages: [{
      role: 'user',
      content: `Genera el informe semanal ejecutivo para el equipo directivo.

DATOS DE LA SEMANA:
${resumenDatos}

El informe debe incluir:
1. Resumen en 2 frases (qué fue bien, qué no)
2. 3 highlights positivos con datos
3. 2-3 áreas de atención o riesgo
4. 1 foco recomendado para esta semana

Tono: directo, sin florituras, orientado a acción.`
    }]
  }
});

return {
  informe: resp.body.content[0].text,
  semana: new Date().toISOString().substring(0, 10),
  datos_raw: resumenDatos
};
```

## Recursos

- [Notebook interactivo](../notebooks/n8n-workflows/03-workflows-negocio.ipynb)
- [n8n — HubSpot node](https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.hubspot/)
- [n8n — Slack node](https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.slack/)
