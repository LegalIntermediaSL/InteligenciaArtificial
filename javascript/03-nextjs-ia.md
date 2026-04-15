# 03 — Chatbot de producción con Next.js 15 y Claude API

> **Bloque:** JavaScript · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. [Arquitectura](#1-arquitectura)
2. [Setup del proyecto](#2-setup-del-proyecto)
3. [Route Handler con streaming](#3-route-handler-con-streaming)
4. [Hook en el cliente](#4-hook-en-el-cliente)
5. [UI con Tailwind](#5-ui-con-tailwind)
6. [System prompt configurable](#6-system-prompt-configurable)
7. [Rate limiting con Upstash Redis](#7-rate-limiting-con-upstash-redis)
8. [Despliegue en Vercel](#8-despliegue-en-vercel)
9. [Extensiones sugeridas](#9-extensiones-sugeridas)

---

## 1. Arquitectura

La arquitectura aprovecha el **App Router** de Next.js 15 junto al **Vercel AI SDK v4** para construir un chatbot con streaming de extremo a extremo.

```
┌─────────────────────────────────────────────────────────────────┐
│                     ARQUITECTURA GENERAL                        │
│                                                                 │
│  Navegador (React)          Servidor (Next.js 15)               │
│  ─────────────────          ──────────────────────              │
│                                                                 │
│  useChat (ai/react)  ──POST──▶  app/api/chat/route.ts          │
│       │                              │                          │
│  Renderiza mensajes  ◀──stream──  streamText()                  │
│       │                              │                          │
│  localStorage        ◀──           @ai-sdk/anthropic           │
│  (system prompt)                     │                          │
│                                      ▼                          │
│                             Claude API (Anthropic)              │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  Capa de protección (middleware.ts)                             │
│  Upstash Redis Rate Limiter: 10 req/min por IP                  │
└─────────────────────────────────────────────────────────────────┘
```

**Flujo de datos:**

```
Usuario escribe → useChat → POST /api/chat
                                   │
                            streamText() ──▶ Claude API
                                   │
                          toDataStreamResponse() ──▶ useChat
                                                       │
                                               renderizar tokens
```

**Tecnologías:**

| Capa | Tecnología |
|------|-----------|
| Framework | Next.js 15 (App Router) |
| IA SDK | Vercel AI SDK v4 (`ai`, `@ai-sdk/anthropic`) |
| Modelo | Claude claude-sonnet-4-6 |
| Estilos | Tailwind CSS v4 |
| Rate limiting | Upstash Redis + `@upstash/ratelimit` |
| Despliegue | Vercel |

---

## 2. Setup del proyecto

### Crear el proyecto

```bash
npx create-next-app@latest chatbot-claude \
  --typescript \
  --tailwind \
  --app \
  --src-dir \
  --import-alias "@/*"

cd chatbot-claude
```

### Instalar dependencias

```bash
npm install ai @ai-sdk/anthropic @upstash/ratelimit @upstash/redis
```

### Variables de entorno

Crea el archivo `.env.local` en la raíz del proyecto:

```env
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Upstash Redis (obtener en console.upstash.com)
UPSTASH_REDIS_REST_URL=https://...upstash.io
UPSTASH_REDIS_REST_TOKEN=...
```

### Estructura de carpetas

```
chatbot-claude/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── chat/
│   │   │       └── route.ts        ← Route Handler
│   │   ├── layout.tsx
│   │   └── page.tsx                ← Página principal
│   ├── components/
│   │   ├── ChatMessage.tsx
│   │   ├── ChatInput.tsx
│   │   └── ChatWindow.tsx
│   └── middleware.ts               ← Rate limiting
├── .env.local
└── package.json
```

---

## 3. Route Handler con streaming

El Route Handler recibe el historial de mensajes y el system prompt desde el cliente, y devuelve la respuesta de Claude como un stream.

```typescript
// src/app/api/chat/route.ts
import { anthropic } from '@ai-sdk/anthropic';
import { streamText } from 'ai';

// Tiempo máximo de ejecución en Vercel (segundos)
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages, system } = await req.json();

  const result = streamText({
    model: anthropic('claude-sonnet-4-6'),
    system: system || 'Eres un asistente útil y conciso. Responde siempre en español.',
    messages,
    maxTokens: 1024,
    temperature: 0.7,
    // Callback opcional para logging
    onFinish({ usage, finishReason }) {
      console.log('[chat] finish_reason:', finishReason);
      console.log('[chat] tokens:', usage);
    },
  });

  return result.toDataStreamResponse();
}
```

**Detalles importantes:**

- `streamText` del Vercel AI SDK gestiona automáticamente el formato del stream.
- `toDataStreamResponse()` devuelve una `Response` compatible con `useChat` del cliente.
- El campo `system` viene del frontend (sección 6), permitiendo personalizarlo por usuario.
- `maxDuration = 30` evita timeouts en Vercel para streams largos.

---

## 4. Hook en el cliente

`useChat` de Vercel AI SDK maneja el estado de los mensajes, el input y el envío de peticiones al Route Handler.

```typescript
// src/app/page.tsx
'use client';

import { useChat } from 'ai/react';
import { useEffect, useState } from 'react';
import ChatWindow from '@/components/ChatWindow';
import ChatInput from '@/components/ChatInput';

const SYSTEM_STORAGE_KEY = 'chatbot_system_prompt';
const DEFAULT_SYSTEM = 'Eres un asistente útil y conciso. Responde siempre en español.';

export default function Page() {
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM);

  // Cargar system prompt guardado
  useEffect(() => {
    const saved = localStorage.getItem(SYSTEM_STORAGE_KEY);
    if (saved) setSystemPrompt(saved);
  }, []);

  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    error,
    reload,
    stop,
  } = useChat({
    api: '/api/chat',
    // Enviar el system prompt con cada petición
    body: { system: systemPrompt },
    onError(err) {
      console.error('[useChat] error:', err);
    },
  });

  return (
    <main className="flex flex-col h-screen bg-gray-50">
      <header className="flex items-center justify-between px-6 py-4 bg-white border-b shadow-sm">
        <h1 className="text-xl font-semibold text-gray-800">Chatbot Claude</h1>
        <span className="text-xs text-gray-400">claude-sonnet-4-6</span>
      </header>

      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        error={error}
        onReload={reload}
      />

      <ChatInput
        input={input}
        isLoading={isLoading}
        onInputChange={handleInputChange}
        onSubmit={handleSubmit}
        onStop={stop}
        systemPrompt={systemPrompt}
        onSystemPromptChange={(value) => {
          setSystemPrompt(value);
          localStorage.setItem(SYSTEM_STORAGE_KEY, value);
        }}
      />
    </main>
  );
}
```

---

## 5. UI con Tailwind

### ChatMessage

```typescript
// src/components/ChatMessage.tsx
import type { Message } from 'ai';

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex w-full mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {/* Avatar del asistente */}
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center mr-3 mt-1">
          <span className="text-white text-xs font-bold">IA</span>
        </div>
      )}

      <div
        className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-indigo-600 text-white rounded-tr-none'
            : 'bg-white text-gray-800 border border-gray-200 rounded-tl-none shadow-sm'
        }`}
      >
        {message.content}
      </div>

      {/* Avatar del usuario */}
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center ml-3 mt-1">
          <span className="text-gray-600 text-xs font-bold">Tú</span>
        </div>
      )}
    </div>
  );
}
```

### ChatWindow

```typescript
// src/components/ChatWindow.tsx
'use client';

import { useEffect, useRef } from 'react';
import type { Message } from 'ai';
import ChatMessage from './ChatMessage';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  error: Error | undefined;
  onReload: () => void;
}

export default function ChatWindow({
  messages,
  isLoading,
  error,
  onReload,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll al último mensaje
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-gray-400">
          <p className="text-lg font-medium mb-2">Hola, soy Claude</p>
          <p className="text-sm">Escribe un mensaje para comenzar la conversación.</p>
        </div>
      )}

      {messages.map((message) => (
        <ChatMessage key={message.id} message={message} />
      ))}

      {/* Indicador de carga */}
      {isLoading && (
        <div className="flex justify-start mb-4">
          <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
            <div className="flex gap-1 items-center">
              <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
              <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
              <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" />
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex justify-center mb-4">
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-xl flex items-center gap-3">
            <span>Error al conectar con el servidor.</span>
            <button
              onClick={onReload}
              className="underline font-medium hover:text-red-900"
            >
              Reintentar
            </button>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
```

### ChatInput

```typescript
// src/components/ChatInput.tsx
'use client';

import { useState, type FormEvent, type ChangeEvent } from 'react';

interface ChatInputProps {
  input: string;
  isLoading: boolean;
  onInputChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  onStop: () => void;
  systemPrompt: string;
  onSystemPromptChange: (value: string) => void;
}

export default function ChatInput({
  input,
  isLoading,
  onInputChange,
  onSubmit,
  onStop,
  systemPrompt,
  onSystemPromptChange,
}: ChatInputProps) {
  const [showSystem, setShowSystem] = useState(false);

  return (
    <div className="border-t bg-white px-4 py-3">
      {/* System prompt expandible */}
      {showSystem && (
        <div className="mb-3">
          <label className="block text-xs font-medium text-gray-500 mb-1">
            System Prompt
          </label>
          <textarea
            value={systemPrompt}
            onChange={(e) => onSystemPromptChange(e.target.value)}
            rows={3}
            className="w-full text-sm border border-gray-200 rounded-xl px-3 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
      )}

      <form onSubmit={onSubmit} className="flex items-center gap-2">
        {/* Botón para mostrar/ocultar system prompt */}
        <button
          type="button"
          onClick={() => setShowSystem((v) => !v)}
          title="Configurar system prompt"
          className="flex-shrink-0 w-9 h-9 flex items-center justify-center rounded-xl border border-gray-200 text-gray-500 hover:bg-gray-50 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>

        <input
          type="text"
          value={input}
          onChange={onInputChange}
          placeholder="Escribe un mensaje..."
          disabled={isLoading}
          className="flex-1 border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
        />

        {isLoading ? (
          <button
            type="button"
            onClick={onStop}
            className="flex-shrink-0 w-10 h-10 bg-red-500 hover:bg-red-600 text-white rounded-xl flex items-center justify-center transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="1" />
            </svg>
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="flex-shrink-0 w-10 h-10 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white rounded-xl flex items-center justify-center transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        )}
      </form>
    </div>
  );
}
```

---

## 6. System prompt configurable

El system prompt se pasa desde el cliente en el cuerpo de la petición y se persiste en `localStorage`. Esto permite personalizar el comportamiento del chatbot sin desplegar cambios de código.

**En el cliente (`page.tsx`):**

```typescript
// El system prompt viaja en el body de cada petición
const { messages, ... } = useChat({
  api: '/api/chat',
  body: { system: systemPrompt },   // enviado en cada POST
});
```

**En el servidor (`route.ts`):**

```typescript
export async function POST(req: Request) {
  const { messages, system } = await req.json();

  const result = streamText({
    model: anthropic('claude-sonnet-4-6'),
    system: system || 'Responde siempre en español.',
    messages,
  });

  return result.toDataStreamResponse();
}
```

**Persistencia en localStorage:**

```typescript
// Guardar al cambiar
localStorage.setItem('chatbot_system_prompt', nuevoPrompt);

// Recuperar al cargar
const saved = localStorage.getItem('chatbot_system_prompt');
if (saved) setSystemPrompt(saved);
```

Ejemplos de system prompts útiles:

```
Eres un asistente de código experto en TypeScript y Next.js. Sé conciso y muestra siempre ejemplos de código.

Eres un tutor de español para hablantes de inglés. Corrige errores con amabilidad.

Eres un analista de datos. Cuando te den datos, responde con tablas Markdown y estadísticas clave.
```

---

## 7. Rate limiting con Upstash Redis

El middleware de Next.js intercepta todas las peticiones a `/api/chat` antes de que lleguen al Route Handler.

### Middleware

```typescript
// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

// Instanciar el cliente Redis y el rate limiter una sola vez (fuera del handler)
const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

const ratelimit = new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(10, '1 m'), // 10 peticiones por minuto
  analytics: true,
  prefix: 'chatbot_rl',
});

export async function middleware(request: NextRequest) {
  // Solo aplicar rate limiting a la API del chat
  if (!request.nextUrl.pathname.startsWith('/api/chat')) {
    return NextResponse.next();
  }

  // Obtener la IP del cliente
  const ip =
    request.headers.get('x-forwarded-for')?.split(',')[0].trim() ??
    request.headers.get('x-real-ip') ??
    '127.0.0.1';

  const { success, limit, remaining, reset } = await ratelimit.limit(ip);

  if (!success) {
    return new NextResponse(
      JSON.stringify({
        error: 'Demasiadas peticiones. Espera un momento antes de continuar.',
        limit,
        remaining: 0,
        reset: new Date(reset).toISOString(),
      }),
      {
        status: 429,
        headers: {
          'Content-Type': 'application/json',
          'X-RateLimit-Limit': String(limit),
          'X-RateLimit-Remaining': '0',
          'X-RateLimit-Reset': String(reset),
          'Retry-After': String(Math.ceil((reset - Date.now()) / 1000)),
        },
      }
    );
  }

  const response = NextResponse.next();
  response.headers.set('X-RateLimit-Limit', String(limit));
  response.headers.set('X-RateLimit-Remaining', String(remaining));
  return response;
}

export const config = {
  matcher: '/api/chat',
};
```

**Manejar el error 429 en el cliente:**

```typescript
// En page.tsx — useChat maneja errores automáticamente
const { error } = useChat({
  onError(err) {
    // err.message contiene el JSON de error del servidor
    if (err.message.includes('429') || err.message.includes('Demasiadas')) {
      alert('Has superado el límite de mensajes. Espera un minuto.');
    }
  },
});
```

---

## 8. Despliegue en Vercel

### vercel.json

```json
{
  "functions": {
    "src/app/api/chat/route.ts": {
      "maxDuration": 30
    }
  }
}
```

### Pasos de deploy

**1. Inicializar repositorio Git:**

```bash
git init
git add .
git commit -m "feat: chatbot Next.js 15 con Claude y rate limiting"
```

**2. Crear proyecto en Vercel:**

```bash
# Instalar Vercel CLI si no lo tienes
npm install -g vercel

# Conectar y desplegar
vercel
```

O desde la interfaz web:
- Ve a [vercel.com](https://vercel.com) → "New Project"
- Importa el repositorio de GitHub
- Vercel detecta automáticamente Next.js

**3. Configurar variables de entorno en Vercel:**

En el dashboard de tu proyecto → Settings → Environment Variables:

```
ANTHROPIC_API_KEY          = sk-ant-...
UPSTASH_REDIS_REST_URL     = https://...upstash.io
UPSTASH_REDIS_REST_TOKEN   = ...
```

**4. Desplegar:**

```bash
vercel --prod
```

### Verificar el deploy

```bash
# Comprobar que el endpoint responde
curl -X POST https://tu-proyecto.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hola"}]}'
```

---

## 9. Extensiones sugeridas

| Extensión | Descripción | Dificultad |
|-----------|-------------|-----------|
| **Historial persistente** | Guardar conversaciones en una base de datos (Supabase, PlanetScale) | Media |
| **Autenticación** | Añadir NextAuth.js para que cada usuario tenga su historial | Media |
| **Upload de archivos** | Permitir adjuntar PDFs o imágenes usando la API multimodal de Claude | Alta |
| **Tool use en el servidor** | Añadir herramientas (búsqueda web, calculadora) en `streamText` | Alta |
| **Múltiples agentes** | Selector de "personalidad" con system prompts distintos | Baja |
| **Temas** | Dark mode con `next-themes` | Baja |
| **Tests E2E** | Playwright para probar el flujo completo de chat | Media |

---

**Anterior:** [02 — Vercel AI SDK con Claude](./02-vercel-ai-sdk.md) · Fin del bloque JavaScript
