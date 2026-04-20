# Mastra.ai — Framework TypeScript para agentes

Mastra es el equivalente TypeScript de PydanticAI: agentes tipados con Zod,
workflows con estado, memoria integrada y herramientas de evaluación.

---

## Instalación

```bash
npm create mastra@latest mi-proyecto-ia
cd mi-proyecto-ia
npm install
```

O sobre un proyecto existente:

```bash
npm install @mastra/core @mastra/memory
npm install @ai-sdk/anthropic  # o @ai-sdk/openai, @ai-sdk/google
```

---

## 1. Agente básico con Mastra

```typescript
// src/mastra/agents/contrato-agent.ts
import { Agent } from '@mastra/core/agent';
import { anthropic } from '@ai-sdk/anthropic';

export const contratoAgent = new Agent({
  name: 'Analizador de Contratos',
  instructions: `
    Eres un experto en análisis de contratos legales.
    Identifica cláusulas de riesgo, obligaciones y plazos.
    Responde siempre en español.
  `,
  model: anthropic('claude-haiku-4-5-20251001'),
});

// Uso
const respuesta = await contratoAgent.generate(
  'Analiza esta cláusula de responsabilidad...'
);
console.log(respuesta.text);
```

---

## 2. Herramientas con Zod

```typescript
import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

const calcularPreavisoTool = createTool({
  id: 'calcular-preaviso',
  description: 'Calcula la fecha límite para enviar un preaviso de contrato',
  inputSchema: z.object({
    fechaVencimiento: z.string().describe('Fecha de vencimiento ISO 8601'),
    diasPrevio: z.number().min(1).max(365).describe('Días de preaviso requeridos'),
  }),
  outputSchema: z.object({
    fechaLimite: z.string(),
    diasRestantes: z.number(),
    urgente: z.boolean(),
  }),
  execute: async ({ context }) => {
    const vencimiento = new Date(context.fechaVencimiento);
    const limite = new Date(vencimiento);
    limite.setDate(limite.getDate() - context.diasPrevio);

    const hoy = new Date();
    const diasRestantes = Math.floor(
      (limite.getTime() - hoy.getTime()) / (1000 * 60 * 60 * 24)
    );

    return {
      fechaLimite: limite.toISOString().split('T')[0],
      diasRestantes: Math.max(0, diasRestantes),
      urgente: diasRestantes < 30,
    };
  },
});

// Agente con herramienta
export const agentePlazos = new Agent({
  name: 'Gestión de Plazos',
  instructions: 'Gestiona plazos y vencimientos de contratos.',
  model: anthropic('claude-haiku-4-5-20251001'),
  tools: { calcularPreaviso: calcularPreavisoTool },
});
```

---

## 3. Workflows con estado

Los workflows de Mastra son grafos de pasos con tipado estricto. Cada paso
declara su input y output con Zod, lo que hace el flujo introspectable y testeable.

```typescript
import { createWorkflow, createStep } from '@mastra/core/workflows';
import { z } from 'zod';

// Pasos del workflow
const clasificarContratoStep = createStep({
  id: 'clasificar-contrato',
  description: 'Clasifica el tipo y riesgo del contrato',
  inputSchema: z.object({
    textoContrato: z.string(),
  }),
  outputSchema: z.object({
    tipoContrato: z.enum(['arrendamiento', 'servicio', 'compraventa', 'laboral', 'otro']),
    nivelRiesgo: z.enum(['alto', 'medio', 'bajo']),
    clausulasIdentificadas: z.number(),
  }),
  execute: async ({ inputData, mastra }) => {
    const agente = mastra.getAgent('Analizador de Contratos');
    const resp = await agente.generate(
      `Clasifica este contrato en JSON:
      {"tipoContrato": "...", "nivelRiesgo": "...", "clausulasIdentificadas": 0}
      
      Contrato: ${inputData.textoContrato}`
    );
    return JSON.parse(resp.text);
  },
});

const extraerPlazosStep = createStep({
  id: 'extraer-plazos',
  description: 'Extrae fechas y plazos relevantes',
  inputSchema: z.object({
    textoContrato: z.string(),
    tipoContrato: z.string(),
  }),
  outputSchema: z.object({
    fechaInicio: z.string().nullable(),
    fechaVencimiento: z.string().nullable(),
    diasPriorAviso: z.number(),
    renovacionAutomatica: z.boolean(),
  }),
  execute: async ({ inputData, mastra }) => {
    const agente = mastra.getAgent('Analizador de Contratos');
    const resp = await agente.generate(
      `Extrae plazos de este ${inputData.tipoContrato} en JSON.
      Contrato: ${inputData.textoContrato}`
    );
    return JSON.parse(resp.text);
  },
});

const generarAlertasStep = createStep({
  id: 'generar-alertas',
  inputSchema: z.object({
    fechaVencimiento: z.string().nullable(),
    diasPriorAviso: z.number(),
    nivelRiesgo: z.string(),
  }),
  outputSchema: z.object({
    alertas: z.array(z.object({
      tipo: z.string(),
      fecha: z.string(),
      mensaje: z.string(),
    })),
  }),
  execute: async ({ inputData }) => {
    if (!inputData.fechaVencimiento) return { alertas: [] };

    const vencimiento = new Date(inputData.fechaVencimiento);
    const alertas = [
      { tipo: 'preaviso', fecha: offsetDate(vencimiento, -inputData.diasPriorAviso), mensaje: 'Fecha límite de preaviso' },
      { tipo: 'revision', fecha: offsetDate(vencimiento, -60), mensaje: 'Revisión 60 días antes' },
      { tipo: 'vencimiento', fecha: inputData.fechaVencimiento, mensaje: 'Vencimiento del contrato' },
    ];

    return { alertas };
  },
});

// Componer el workflow
export const analisisContratoWorkflow = createWorkflow({
  id: 'analisis-contrato',
  inputSchema: z.object({ textoContrato: z.string() }),
  outputSchema: z.object({
    clasificacion: z.object({ tipoContrato: z.string(), nivelRiesgo: z.string() }),
    alertas: z.array(z.object({ tipo: z.string(), fecha: z.string(), mensaje: z.string() })),
  }),
})
  .then(clasificarContratoStep)
  .then(extraerPlazosStep)
  .then(generarAlertasStep)
  .commit();

function offsetDate(fecha: Date, dias: number): string {
  const nueva = new Date(fecha);
  nueva.setDate(nueva.getDate() + dias);
  return nueva.toISOString().split('T')[0];
}
```

---

## 4. Memoria integrada

```typescript
import { Memory } from '@mastra/memory';
import { Agent } from '@mastra/core/agent';

const memory = new Memory({
  // Por defecto usa almacenamiento en memoria; en producción usa PostgreSQL:
  // storage: new PostgresStore({ connectionString: process.env.DATABASE_URL }),
});

const agenteConMemoria = new Agent({
  name: 'Asistente Legal Persistente',
  instructions: 'Recuerdas el historial completo de contratos de cada cliente.',
  model: anthropic('claude-haiku-4-5-20251001'),
  memory,
});

// Thread por cliente
const threadId = 'cliente-acme-corp';

const resp1 = await agenteConMemoria.generate(
  'El contrato con Proveedor X vence el 1 de septiembre de 2026.',
  { threadId, resourceId: 'acme-corp' }
);

// En una llamada posterior, el agente recuerda el contexto
const resp2 = await agenteConMemoria.generate(
  '¿Cuándo debería enviar el preaviso?',
  { threadId, resourceId: 'acme-corp' }
);
// El agente "sabe" que se refiere al contrato mencionado antes
```

---

## 5. Evaluación de agentes

```typescript
import { evaluate } from '@mastra/evals';
import { AnswerRelevancyMetric, FaithfulnessMetric } from '@mastra/evals/metrics';

const resultadosEval = await evaluate({
  agent: contratoAgent,
  dataset: [
    {
      input: '¿Qué es una cláusula de responsabilidad limitada?',
      expectedOutput: 'Limita la responsabilidad máxima del proveedor...',
    },
    {
      input: '¿Cuándo se puede rescindir un contrato anticipadamente?',
      expectedOutput: 'Generalmente cuando existe incumplimiento grave...',
    },
  ],
  metrics: [
    new AnswerRelevancyMetric(),
    new FaithfulnessMetric(),
  ],
});

console.table(resultadosEval.results.map(r => ({
  pregunta: r.input.slice(0, 40),
  relevancia: r.metrics.answerRelevancy.toFixed(2),
  fidelidad: r.metrics.faithfulness.toFixed(2),
})));
```

---

## 6. Integración con Next.js

```typescript
// app/api/contrato/route.ts
import { mastra } from '@/mastra';
import { streamText } from 'ai';

export async function POST(req: Request) {
  const { textoContrato } = await req.json();
  const agente = mastra.getAgent('Analizador de Contratos');

  const stream = await agente.stream(
    `Analiza este contrato y da tu evaluación: ${textoContrato}`
  );

  return stream.toDataStreamResponse();
}
```

```tsx
// components/ContratoChat.tsx
'use client';
import { useChat } from '@ai-sdk/react';

export function ContratoChat() {
  const { messages, input, handleSubmit, handleInputChange, isLoading } = useChat({
    api: '/api/contrato',
  });

  return (
    <div>
      {messages.map(m => (
        <div key={m.id} className={m.role === 'user' ? 'text-right' : 'text-left'}>
          {m.content}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <textarea value={input} onChange={handleInputChange} />
        <button type="submit" disabled={isLoading}>Analizar</button>
      </form>
    </div>
  );
}
```

---

→ Siguiente: [Comparativa de frameworks](04-comparativa-frameworks.md)
