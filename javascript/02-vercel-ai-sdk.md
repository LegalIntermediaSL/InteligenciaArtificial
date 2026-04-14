# 02 — Vercel AI SDK con Claude

> **Bloque:** JavaScript | **Nivel:** Intermedio

El [AI SDK de Vercel](https://sdk.vercel.ai/) es una librería open source diseñada para construir aplicaciones de IA en JavaScript/TypeScript. Es la opción natural para proyectos **Next.js** o **React**, pero funciona igual de bien en Node.js puro.

Destaca por su soporte de primera clase para streaming, su integración con Zod para salidas estructuradas y su API limpia para tool use.

---

## Instalación

```bash
npm install ai @ai-sdk/anthropic zod dotenv
```

Archivo `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 1. generateText — respuesta completa

```javascript
import 'dotenv/config';
import { anthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';

const { text, usage } = await generateText({
  model: anthropic('claude-haiku-4-5-20251001'),
  maxTokens: 256,
  system: 'Responde siempre en español.',
  prompt: '¿Cuál es la diferencia entre ML y Deep Learning?',
});

console.log(text);
console.log(`Tokens — prompt: ${usage.promptTokens}, completion: ${usage.completionTokens}`);
```

---

## 2. streamText — streaming

```javascript
import { streamText } from 'ai';

const resultado = streamText({
  model: anthropic('claude-haiku-4-5-20251001'),
  prompt: 'Explica el concepto RAG en 3 puntos.',
});

for await (const delta of resultado.textStream) {
  process.stdout.write(delta);
}

// Acceder a uso después del stream
const { totalTokens } = await resultado.usage;
console.log(`\nTotal tokens: ${totalTokens}`);
```

### En Next.js (App Router)

```typescript
// app/api/chat/route.ts
import { anthropic } from '@ai-sdk/anthropic';
import { streamText } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: anthropic('claude-haiku-4-5-20251001'),
    messages,
  });

  return result.toDataStreamResponse();
}
```

```typescript
// app/page.tsx
'use client';
import { useChat } from 'ai/react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();

  return (
    <div>
      {messages.map(m => (
        <div key={m.id}><b>{m.role}:</b> {m.content}</div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit">Enviar</button>
      </form>
    </div>
  );
}
```

---

## 3. generateObject — salida estructurada con Zod

Esta es una de las características más potentes del AI SDK: definir el schema con Zod y obtener el objeto validado directamente.

```javascript
import { generateObject } from 'ai';
import { z } from 'zod';

const schema = z.object({
  categoria: z.enum(['BUG', 'FEATURE_REQUEST', 'PREGUNTA', 'QUEJA', 'OTRO']),
  prioridad: z.enum(['ALTA', 'MEDIA', 'BAJA']),
  confianza: z.number().min(0).max(1),
  resumen: z.string().describe('Resumen en una frase'),
  equipo: z.enum(['backend', 'frontend', 'soporte', 'producto']),
});

const { object } = await generateObject({
  model: anthropic('claude-haiku-4-5-20251001'),
  schema,
  system: 'Eres un sistema de clasificación de tickets de soporte.',
  prompt: 'Clasifica: "La app se cuelga al pagar con tarjeta en el checkout"',
});

console.log(object.categoria);   // BUG
console.log(object.prioridad);   // ALTA
console.log(object.confianza);   // 0.95
console.log(object.equipo);      // backend
```

El AI SDK se encarga de la validación — si el modelo no respeta el schema, reintenta automáticamente.

---

## 4. Tool use con maxSteps

Con `maxSteps > 1`, el modelo puede hacer múltiples llamadas a herramientas en un solo `generateText`:

```javascript
import { generateText, tool } from 'ai';
import { z } from 'zod';

const { text, steps } = await generateText({
  model: anthropic('claude-sonnet-4-6'),
  maxSteps: 5,
  system: 'Usa las herramientas disponibles para responder con precisión.',
  prompt: '¿Qué fecha es hoy? ¿Cuántos segundos hay en una semana?',

  tools: {
    obtener_fecha: tool({
      description: 'Devuelve la fecha actual',
      parameters: z.object({}),
      execute: async () => ({
        fecha: new Date().toLocaleDateString('es-ES'),
      }),
    }),

    calcular: tool({
      description: 'Evalúa una expresión matemática',
      parameters: z.object({
        expresion: z.string(),
      }),
      execute: async ({ expresion }) => {
        const resultado = new Function('Math', `return ${expresion}`)(Math);
        return { resultado };
      },
    }),
  },
});

console.log('Respuesta:', text);
console.log('Pasos ejecutados:', steps.length);
```

**Flujo de tool use con maxSteps:**
```
Prompt
  ↓
Modelo decide usar herramienta → llama a execute()
  ↓
Resultado de la herramienta → vuelve al modelo
  ↓
Modelo responde (o usa otra herramienta)
  ↓
Respuesta final
```

---

## 5. Conversación multi-turno

```javascript
const historial = [];

const turnos = [
  '¿Qué es un embedding?',
  '¿Para qué se usan en aplicaciones reales?',
];

for (const pregunta of turnos) {
  historial.push({ role: 'user', content: pregunta });

  const { text } = await generateText({
    model: anthropic('claude-haiku-4-5-20251001'),
    maxTokens: 300,
    system: 'Eres un tutor de IA. Responde de forma concisa.',
    messages: historial,
  });

  historial.push({ role: 'assistant', content: text });
  console.log(`Usuario: ${pregunta}\nIA: ${text}\n`);
}
```

---

## Comparativa rápida con LangChain.js

| Feature | Vercel AI SDK | LangChain.js |
|---------|---------------|--------------|
| **Structured output** | Zod nativo (`generateObject`) | Manual o `withStructuredOutput` |
| **Streaming** | Primera clase | Sí, con iteradores |
| **React/Next.js** | `useChat`, `useCompletion` hooks | Manual |
| **Agentes complejos** | `maxSteps` | Agentes con tools completos |
| **RAG** | No incluido | Sí (vectorstores, retrievers) |
| **Multi-proveedor** | Sí | Sí |

---

## Ejecutar los ejemplos

```bash
node 02-vercel-ai-sdk.js
```

**Ver también:** [01-langchain-js.md](./01-langchain-js.md) para cadenas y pipelines complejos.
