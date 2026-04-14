/**
 * 02 — Vercel AI SDK con Claude
 *
 * El AI SDK de Vercel es ideal para aplicaciones React/Next.js pero también
 * funciona perfectamente en Node.js puro para backends y scripts.
 *
 * Instalación:
 *   npm install ai @ai-sdk/anthropic dotenv
 *
 * Uso:
 *   node 02-vercel-ai-sdk.js
 */

import 'dotenv/config';
import { anthropic } from '@ai-sdk/anthropic';
import {
  generateText,
  streamText,
  generateObject,
  tool,
} from 'ai';
import { z } from 'zod';

// ─── 1. generateText — respuesta completa ───────────────────────────────────

async function textoCompleto() {
  console.log('\n=== 1. generateText ===');

  const { text, usage } = await generateText({
    model: anthropic('claude-haiku-4-5-20251001'),
    maxTokens: 256,
    temperature: 0.7,
    system: 'Responde siempre en español y de forma concisa.',
    prompt: '¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?',
  });

  console.log('Respuesta:', text);
  console.log(`Tokens → prompt: ${usage.promptTokens}, completion: ${usage.completionTokens}`);
}

// ─── 2. streamText — streaming en Node.js ───────────────────────────────────

async function textoStreaming() {
  console.log('\n=== 2. streamText ===');

  const resultado = streamText({
    model: anthropic('claude-haiku-4-5-20251001'),
    maxTokens: 300,
    prompt: 'Explica el concepto de RAG (Retrieval-Augmented Generation) en 3 puntos clave.',
  });

  process.stdout.write('Streaming: ');
  for await (const delta of resultado.textStream) {
    process.stdout.write(delta);
  }

  const usage = await resultado.usage;
  console.log(`\n(${usage.totalTokens} tokens)`);
}

// ─── 3. generateObject — salida estructurada con Zod ────────────────────────

async function objetoEstructurado() {
  console.log('\n=== 3. generateObject (structured output) ===');

  // Definir el schema con Zod
  const schemaTicket = z.object({
    categoria: z.enum(['BUG', 'FEATURE_REQUEST', 'PREGUNTA', 'QUEJA', 'OTRO']),
    prioridad: z.enum(['ALTA', 'MEDIA', 'BAJA']),
    confianza: z.number().min(0).max(1),
    resumen: z.string().describe('Resumen del ticket en una frase'),
    equipo_responsable: z.enum(['backend', 'frontend', 'soporte', 'producto']),
  });

  const tickets = [
    'La app se cuelga al intentar pagar con tarjeta de crédito en el checkout',
    'Estaría bien poder exportar los datos a Excel con un clic',
    '¿Cómo puedo añadir a otro usuario como administrador?',
  ];

  for (const ticket of tickets) {
    const { object } = await generateObject({
      model: anthropic('claude-haiku-4-5-20251001'),
      schema: schemaTicket,
      system: 'Eres un sistema de clasificación de tickets de soporte técnico.',
      prompt: `Clasifica este ticket: "${ticket}"`,
    });

    console.log(`\nTicket: "${ticket.substring(0, 60)}..."`);
    console.log(`  → ${object.categoria} | ${object.prioridad} | confianza: ${object.confianza}`);
    console.log(`  → Equipo: ${object.equipo_responsable}`);
    console.log(`  → ${object.resumen}`);
  }
}

// ─── 4. Tool use — herramientas con Zod ─────────────────────────────────────

async function toolUse() {
  console.log('\n=== 4. Tool use ===');

  // Definir herramientas
  const herramientas = {
    calcular: tool({
      description: 'Evalúa una expresión matemática y devuelve el resultado',
      parameters: z.object({
        expresion: z.string().describe('Expresión matemática (ej: 2**10, Math.sqrt(144))'),
      }),
      execute: async ({ expresion }) => {
        try {
          // eslint-disable-next-line no-new-func
          const resultado = new Function('Math', `return ${expresion}`)(Math);
          return { resultado, expresion };
        } catch (error) {
          return { error: error.message };
        }
      },
    }),

    obtener_fecha: tool({
      description: 'Devuelve la fecha y hora actuales',
      parameters: z.object({}),
      execute: async () => ({
        fecha: new Date().toLocaleDateString('es-ES'),
        hora: new Date().toLocaleTimeString('es-ES'),
        timestamp: Date.now(),
      }),
    }),

    buscar_info: tool({
      description: 'Busca información en la base de conocimiento',
      parameters: z.object({
        tema: z.string().describe('Tema a buscar'),
      }),
      execute: async ({ tema }) => {
        const base = {
          anthropic: 'Anthropic es una empresa de seguridad en IA fundada en 2021.',
          claude: 'Claude es la familia de modelos de IA de Anthropic.',
          openai: 'OpenAI es la empresa creadora de ChatGPT y GPT-4.',
        };
        const resultado = base[tema.toLowerCase()];
        return resultado
          ? { resultado }
          : { resultado: `No se encontró información sobre: ${tema}` };
      },
    }),
  };

  // Bucle agéntico con maxSteps
  const { text, steps, usage } = await generateText({
    model: anthropic('claude-sonnet-4-6'),
    maxTokens: 1024,
    maxSteps: 5, // Permite múltiples pasos tool use → respuesta
    system: 'Eres un asistente que usa herramientas para responder preguntas con precisión.',
    prompt: '¿Qué fecha es hoy? Y calcula cuántos días tiene un año bisiesto (366) multiplicado por 24 horas.',
    tools: herramientas,
  });

  console.log('\nRespuesta final:', text);
  console.log(`Pasos ejecutados: ${steps.length}`);
  console.log(`Total tokens: ${usage.totalTokens}`);

  // Mostrar herramientas usadas
  for (const paso of steps) {
    for (const llamada of paso.toolCalls ?? []) {
      console.log(`  🔧 ${llamada.toolName}(${JSON.stringify(llamada.args)})`);
    }
  }
}

// ─── 5. Conversación multi-turno ─────────────────────────────────────────────

async function conversacionMultiturno() {
  console.log('\n=== 5. Conversación multi-turno ===');

  const historial = [];

  const turnos = [
    '¿Qué es un embedding en IA?',
    '¿Y para qué se usan en aplicaciones reales?',
    '¿Puedes darme un ejemplo concreto con código?',
  ];

  for (const pregunta of turnos) {
    historial.push({ role: 'user', content: pregunta });

    const { text } = await generateText({
      model: anthropic('claude-haiku-4-5-20251001'),
      maxTokens: 300,
      system: 'Eres un tutor de IA. Responde de forma concisa y pedagógica.',
      messages: historial,
    });

    console.log(`\nUsuario: ${pregunta}`);
    console.log(`IA: ${text.substring(0, 250)}...`);

    historial.push({ role: 'assistant', content: text });
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  console.log('Vercel AI SDK con Claude — Ejemplos');
  console.log('=====================================');

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('Error: define ANTHROPIC_API_KEY en tu archivo .env');
    process.exit(1);
  }

  try {
    await textoCompleto();
    await textoStreaming();
    await objetoEstructurado();
    await toolUse();
    await conversacionMultiturno();

    console.log('\n✓ Todos los ejemplos completados');
  } catch (error) {
    console.error('Error:', error.message);
    if (error.cause) console.error('Causa:', error.cause);
  }
}

main();
