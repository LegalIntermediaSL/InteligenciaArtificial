# 01 — LangChain.js con Claude

> **Bloque:** JavaScript | **Nivel:** Intermedio

LangChain es el framework más popular para construir aplicaciones con LLMs. Disponible tanto en Python como en JavaScript/TypeScript, permite construir pipelines complejos: cadenas, agentes, RAG y más.

---

## Instalación

```bash
npm install @langchain/anthropic @langchain/core langchain dotenv
```

Archivo `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 1. Llamada simple

El punto de entrada es `ChatAnthropic`, el wrapper de LangChain sobre la API de Anthropic:

```javascript
import 'dotenv/config';
import { ChatAnthropic } from '@langchain/anthropic';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

const modelo = new ChatAnthropic({
  model: 'claude-haiku-4-5-20251001',
  maxTokens: 256,
  temperature: 0.7,
});

const respuesta = await modelo.invoke([
  new SystemMessage('Eres un asistente experto en IA.'),
  new HumanMessage('¿Qué es un transformer?'),
]);

console.log(respuesta.content);
console.log('Tokens:', respuesta.usage_metadata);
```

**Tipos de mensaje:**
| Clase | Equivalente API | Uso |
|-------|-----------------|-----|
| `SystemMessage` | `role: "system"` | Instrucciones al modelo |
| `HumanMessage` | `role: "user"` | Mensaje del usuario |
| `AIMessage` | `role: "assistant"` | Respuesta previa del modelo |

---

## 2. PromptTemplate y cadenas (LCEL)

LangChain Expression Language (LCEL) permite componer componentes con `|` (pipe) o `RunnableSequence`:

```javascript
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

const prompt = ChatPromptTemplate.fromMessages([
  ['system', 'Eres un experto en {dominio}.'],
  ['human', 'Explica "{concepto}" en {palabras} palabras para alguien sin conocimientos técnicos.'],
]);

const cadena = RunnableSequence.from([
  prompt,
  new ChatAnthropic({ model: 'claude-haiku-4-5-20251001', maxTokens: 300 }),
  new StringOutputParser(),
]);

// Ejecutar con variables
const resultado = await cadena.invoke({
  dominio: 'inteligencia artificial',
  concepto: 'embeddings',
  palabras: 80,
});
console.log(resultado);
```

El flujo es:
```
Input → PromptTemplate → ChatAnthropic → StringOutputParser → String
```

---

## 3. Clasificación con salida estructurada

Combinar un prompt específico con un parser de JSON es el patrón más sencillo para obtener datos estructurados:

```javascript
const prompt = ChatPromptTemplate.fromMessages([
  ['system', `Clasifica el ticket. Responde SOLO con JSON:
{"categoria": "BUG|FEATURE_REQUEST|PREGUNTA|QUEJA|OTRO",
 "confianza": 0.0-1.0, "razon": "..."}`],
  ['human', '{ticket}'],
]);

const cadena = prompt
  .pipe(new ChatAnthropic({ model: 'claude-haiku-4-5-20251001' }))
  .pipe(new StringOutputParser());

const raw = await cadena.invoke({ ticket: 'La app se cierra sola al exportar a PDF' });
const resultado = JSON.parse(raw);
console.log(resultado.categoria); // "BUG"
```

> Para schemas complejos, usa `withStructuredOutput()` o el Vercel AI SDK con Zod (ver `02-vercel-ai-sdk.md`).

---

## 4. Streaming

```javascript
const modelo = new ChatAnthropic({
  model: 'claude-haiku-4-5-20251001',
  streaming: true,
});

const stream = await modelo.stream([
  new HumanMessage('Lista 5 casos de uso de los LLMs en empresas.'),
]);

for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

---

## 5. Conversación con historial

```javascript
const mensajes = [
  new SystemMessage('Eres un tutor de IA. Responde en español.'),
  new HumanMessage('¿Qué es un transformer?'),
];

const r1 = await modelo.invoke(mensajes);
mensajes.push(new AIMessage(r1.content));
mensajes.push(new HumanMessage('¿Y cómo difiere de un RNN?'));

const r2 = await modelo.invoke(mensajes);
console.log(r2.content);
```

Para gestionar el historial automáticamente en apps más complejas, usa `ChatMessageHistory` y `RunnableWithMessageHistory` de LangChain.

---

## Conceptos clave

| Concepto | Descripción |
|----------|-------------|
| **Runnable** | Unidad básica composable en LCEL (`invoke`, `stream`, `batch`) |
| **Chain** | Secuencia de runnables |
| **PromptTemplate** | Plantilla reutilizable con variables |
| **OutputParser** | Convierte la salida del modelo (string, JSON, etc.) |
| **Memory** | Gestión de historial en conversaciones |

---

## Ejecutar los ejemplos

```bash
node 01-langchain-js.js
```

**Ver también:** [02-vercel-ai-sdk.md](./02-vercel-ai-sdk.md) para structured output con Zod y tool use.
