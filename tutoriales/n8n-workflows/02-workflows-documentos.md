# Workflows de Documentos con n8n y Claude

## Casos de uso de documentos con IA

Procesar documentos es uno de los casos más comunes y rentables para empresas:
facturas, contratos, albaranes, emails, informes y formularios.

```
FLUJO TÍPICO DE DOCUMENTO
──────────────────────────
Fuente del documento
  (Email adjunto / Google Drive / Dropbox / formulario web / API)
        ↓
Extracción de texto
  (PDF → texto con n8n / AWS Textract / Google Document AI)
        ↓
Procesamiento con Claude
  (clasificar / extraer datos / analizar / resumir)
        ↓
Acción de negocio
  (guardar en BD / actualizar CRM / notificar / enviar respuesta)
```

## Workflow 1: Procesamiento automático de facturas

### Arquitectura del workflow

```
[Email Trigger]
     → adjunto PDF detectado
     ↓
[Extract from File]  (nodo n8n nativo)
     → texto del PDF
     ↓
[HTTP Request → Claude API]
     → extrae campos estructurados
     ↓
[Code Node]
     → parsea JSON de Claude
     ↓
[Google Sheets / Airtable]
     → guarda la factura
     ↓
[Slack / Email]
     → notifica si necesita aprobación
```

### Nodo Claude: extraer datos de factura

```javascript
// En nodo "Code" — llamada a Claude con texto del PDF
const textoPDF = $input.first().json.text; // texto extraído del PDF

const response = await $http.request({
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
      content: `Extrae los datos de esta factura. Responde SOLO con JSON válido, sin markdown.

FACTURA:
${textoPDF.substring(0, 3000)}

JSON requerido:
{
  "numero_factura": "...",
  "fecha_emision": "YYYY-MM-DD",
  "fecha_vencimiento": "YYYY-MM-DD",
  "proveedor": {
    "nombre": "...",
    "nif": "...",
    "direccion": "..."
  },
  "cliente": {
    "nombre": "...",
    "nif": "..."
  },
  "lineas": [
    {"descripcion": "...", "cantidad": 1, "precio_unitario": 0.00, "total": 0.00}
  ],
  "subtotal": 0.00,
  "iva_pct": 21,
  "iva_importe": 0.00,
  "total": 0.00,
  "moneda": "EUR",
  "numero_pedido": "...",
  "metodo_pago": "...",
  "confianza": "alta|media|baja"
}`
    }]
  }
});

const datos = JSON.parse(response.body.content[0].text);

// Añadir metadatos del procesamiento
datos.procesado_en = new Date().toISOString();
datos.requiere_revision = datos.confianza !== 'alta' || !datos.numero_factura;

return datos;
```

### Nodo IF: ¿requiere revisión humana?

```javascript
// Condición del nodo IF
{{ $json.requiere_revision === true || $json.total > 5000 }}

// Rama TRUE → Slack con botón de aprobación
// Rama FALSE → guardar directamente en Sheets
```

### Nodo Google Sheets: guardar factura

```json
{
  "hoja": "Facturas_2026",
  "operacion": "appendRow",
  "columnas": {
    "A": "{{ $json.numero_factura }}",
    "B": "{{ $json.fecha_emision }}",
    "C": "{{ $json.proveedor.nombre }}",
    "D": "{{ $json.proveedor.nif }}",
    "E": "{{ $json.total }}",
    "F": "{{ $json.iva_pct }}",
    "G": "{{ $json.confianza }}",
    "H": "{{ $json.procesado_en }}"
  }
}
```

## Workflow 2: Clasificación y archivo de contratos

```javascript
// Nodo Code: clasificar tipo de contrato con Claude
const textoContrato = $input.first().json.text;

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
    messages: [{
      role: 'user',
      content: `Analiza este contrato. JSON sin markdown:
{
  "tipo": "prestacion_servicios|compraventa|arrendamiento|laboral|confidencialidad|otro",
  "partes": ["Empresa A", "Empresa B"],
  "fecha_inicio": "YYYY-MM-DD o null",
  "fecha_fin": "YYYY-MM-DD o null",
  "valor_contrato_eur": 0,
  "renovacion_automatica": true,
  "clausulas_criticas": ["lista de 1-3 cláusulas importantes"],
  "carpeta_archivo": "Proveedores|Clientes|RRHH|Legal|Otro"
}

CONTRATO (primeras 2000 palabras):
${textoContrato.substring(0, 4000)}`
    }]
  }
});

const clasificacion = JSON.parse(resp.body.content[0].text);

// Determinar carpeta en Google Drive
clasificacion.ruta_drive = `Contratos/${clasificacion.tipo}/${new Date().getFullYear()}`;
clasificacion.nombre_archivo = `${clasificacion.partes[0]}_${clasificacion.fecha_inicio || 'sin_fecha'}.pdf`;

return clasificacion;
```

## Workflow 3: Resumen automático de informes largos

```javascript
// Para documentos largos: dividir en chunks y resumir por partes

async function resumirDocumentoLargo(texto, maxChunkChars = 8000) {
  const chunks = [];
  for (let i = 0; i < texto.length; i += maxChunkChars) {
    chunks.push(texto.substring(i, i + maxChunkChars));
  }

  // Resumir cada chunk
  const resumenes = [];
  for (const [idx, chunk] of chunks.entries()) {
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
        messages: [{
          role: 'user',
          content: `Resume este fragmento (${idx + 1}/${chunks.length}) en 3-4 bullets:\n\n${chunk}`
        }]
      }
    });
    resumenes.push(resp.body.content[0].text);
  }

  // Síntesis final
  const resp_final = await $http.request({
    method: 'POST',
    url: 'https://api.anthropic.com/v1/messages',
    headers: {
      'x-api-key': $env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json'
    },
    body: {
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 500,
      messages: [{
        role: 'user',
        content: `Sintetiza estos resúmenes parciales en un resumen ejecutivo final de máximo 200 palabras:\n\n${resumenes.join('\n\n---\n\n')}`
      }]
    }
  });

  return {
    resumen_ejecutivo: resp_final.body.content[0].text,
    num_partes: chunks.length,
    chars_totales: texto.length
  };
}

const texto = $input.first().json.text;
const resultado = await resumirDocumentoLargo(texto);
return resultado;
```

## Workflow 4: Extracción de datos de formularios web

```javascript
// Webhook trigger → formulario web envía datos en JSON
// Enriquecer con Claude antes de guardar en CRM

const datosFormulario = $input.first().json;

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
    messages: [{
      role: 'user',
      content: `Enriquece este lead de formulario web. JSON sin markdown:
{
  "industria_detectada": "tecnología|retail|legal|salud|educacion|otro",
  "tamaño_empresa_estimado": "1-10|11-50|51-200|201-1000|1000+",
  "urgencia": "alta|media|baja",
  "caso_uso_principal": "descripción breve",
  "siguiente_accion": "llamada_inmediata|email_nurturing|demo_producto|descalificar",
  "score_lead": 0-100
}

DATOS DEL FORMULARIO:
Nombre: ${datosFormulario.nombre}
Email: ${datosFormulario.email}
Empresa: ${datosFormulario.empresa || 'N/A'}
Mensaje: ${datosFormulario.mensaje || 'N/A'}
Cargo: ${datosFormulario.cargo || 'N/A'}`
    }]
  }
});

const enriquecimiento = JSON.parse(resp.body.content[0].text);

return {
  ...datosFormulario,
  ...enriquecimiento,
  procesado_en: new Date().toISOString()
};
```

## Recursos

- [Notebook interactivo](../notebooks/n8n-workflows/02-workflows-documentos.ipynb)
- [n8n — Extract from File node](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.extractfromfile/)
- [AWS Textract para OCR avanzado](https://aws.amazon.com/textract/)
