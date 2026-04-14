/**
 * 01 — LangChain.js con Claude
 *
 * Ejemplos progresivos: llamada simple → cadena → RAG básico → agente con herramientas
 *
 * Instalación:
 *   npm install @langchain/anthropic @langchain/core langchain dotenv
 *
 * Uso:
 *   node 01-langchain-js.js
 */

import 'dotenv/config';
import { ChatAnthropic } from '@langchain/anthropic';
import { HumanMessage, SystemMessage, AIMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

// ─── 1. Llamada simple ───────────────────────────────────────────────────────

async function llamadaSimple() {
  console.log('\n=== 1. Llamada simple ===');

  const modelo = new ChatAnthropic({
    model: 'claude-haiku-4-5-20251001',
    maxTokens: 256,
    temperature: 0.7,
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const respuesta = await modelo.invoke([
    new HumanMessage('¿Cuáles son los 3 principales beneficios de los LLMs para las empresas?'),
  ]);

  console.log('Respuesta:', respuesta.content);
  console.log('Tokens usados:', respuesta.usage_metadata);
}

// ─── 2. Conversación con historial ──────────────────────────────────────────

async function conversacion() {
  console.log('\n=== 2. Conversación con historial ===');

  const modelo = new ChatAnthropic({
    model: 'claude-haiku-4-5-20251001',
    maxTokens: 256,
  });

  const mensajes = [
    new SystemMessage('Eres un asistente experto en inteligencia artificial. Responde de forma concisa.'),
    new HumanMessage('¿Qué es un transformer?'),
  ];

  const r1 = await modelo.invoke(mensajes);
  console.log('Usuario: ¿Qué es un transformer?');
  console.log('IA:', r1.content.substring(0, 200) + '...');

  // Añadir la respuesta al historial y hacer seguimiento
  mensajes.push(new AIMessage(r1.content));
  mensajes.push(new HumanMessage('¿Y cómo se diferencia de un RNN?'));

  const r2 = await modelo.invoke(mensajes);
  console.log('\nUsuario: ¿Y cómo se diferencia de un RNN?');
  console.log('IA:', r2.content.substring(0, 200) + '...');
}

// ─── 3. Cadena con prompt template ──────────────────────────────────────────

async function cadenaConTemplate() {
  console.log('\n=== 3. Cadena con PromptTemplate ===');

  const modelo = new ChatAnthropic({
    model: 'claude-haiku-4-5-20251001',
    maxTokens: 400,
  });

  // Crear un prompt reutilizable
  const promptResumen = ChatPromptTemplate.fromMessages([
    ['system', 'Eres un experto en {dominio}. Responde siempre en español.'],
    ['human', 'Explica "{concepto}" en máximo {palabras} palabras para alguien sin conocimientos técnicos.'],
  ]);

  // Construir la cadena: prompt → modelo → parser
  const cadena = RunnableSequence.from([
    promptResumen,
    modelo,
    new StringOutputParser(),
  ]);

  // Ejecutar con diferentes inputs
  const conceptos = [
    { dominio: 'inteligencia artificial', concepto: 'embeddings', palabras: 80 },
    { dominio: 'machine learning', concepto: 'overfitting', palabras: 60 },
  ];

  for (const entrada of conceptos) {
    const resultado = await cadena.invoke(entrada);
    console.log(`\n[${entrada.concepto}] → ${resultado.substring(0, 200)}...`);
  }
}

// ─── 4. Cadena de clasificación con structured output ───────────────────────

async function clasificacionEstructurada() {
  console.log('\n=== 4. Clasificación con salida estructurada ===');

  const modelo = new ChatAnthropic({
    model: 'claude-haiku-4-5-20251001',
    maxTokens: 200,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    ['system', `Clasifica el ticket de soporte en una de estas categorías: BUG, FEATURE_REQUEST, PREGUNTA, QUEJA, OTRO.
Responde ÚNICAMENTE con un JSON válido: {"categoria": "...", "confianza": 0.0-1.0, "razon": "..."}`],
    ['human', '{ticket}'],
  ]);

  const cadena = prompt.pipe(modelo).pipe(new StringOutputParser());

  const tickets = [
    'La aplicación se cierra sola cuando intento exportar a PDF',
    'Sería genial poder filtrar por fecha en los reportes',
    '¿Cómo puedo cambiar mi contraseña?',
  ];

  for (const ticket of tickets) {
    const resultado = await cadena.invoke({ ticket });
    try {
      const parsed = JSON.parse(resultado);
      console.log(`\nTicket: "${ticket.substring(0, 50)}..."`);
      console.log(`  Categoría: ${parsed.categoria} (confianza: ${parsed.confianza})`);
      console.log(`  Razón: ${parsed.razon}`);
    } catch {
      console.log('Respuesta:', resultado);
    }
  }
}

// ─── 5. Streaming ──────────────────────────────────────────────────────────

async function streaming() {
  console.log('\n=== 5. Streaming de respuestas ===');

  const modelo = new ChatAnthropic({
    model: 'claude-haiku-4-5-20251001',
    maxTokens: 400,
    streaming: true,
  });

  process.stdout.write('Respuesta (streaming): ');

  const stream = await modelo.stream([
    new HumanMessage('Enumera 5 casos de uso prácticos de los LLMs en empresas. Sé conciso.'),
  ]);

  for await (const chunk of stream) {
    process.stdout.write(chunk.content);
  }
  console.log('\n');
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  console.log('LangChain.js con Claude — Ejemplos');
  console.log('====================================');

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('Error: define ANTHROPIC_API_KEY en tu archivo .env');
    process.exit(1);
  }

  try {
    await llamadaSimple();
    await conversacion();
    await cadenaConTemplate();
    await clasificacionEstructurada();
    await streaming();

    console.log('\n✓ Todos los ejemplos completados');
  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
