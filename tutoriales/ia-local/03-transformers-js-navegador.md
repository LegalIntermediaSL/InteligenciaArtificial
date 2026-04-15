# 03 — IA en el navegador con Transformers.js

> **Bloque:** IA local · **Nivel:** Avanzado · **Tiempo estimado:** 55 min

---

## Índice

1. [Qué es Transformers.js y casos de uso](#1-qué-es-transformersjs-y-casos-de-uso)
2. [Instalación](#2-instalación)
3. [Inferencia básica en Node.js](#3-inferencia-básica-en-nodejs)
4. [En el navegador con bundler](#4-en-el-navegador-con-bundler)
5. [Embeddings en el navegador](#5-embeddings-en-el-navegador)
6. [Web Worker para no bloquear el UI](#6-web-worker-para-no-bloquear-el-ui)
7. [Caso práctico: buscador semántico offline](#7-caso-práctico-buscador-semántico-offline)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Qué es Transformers.js y casos de uso

Transformers.js es el port oficial de Hugging Face de la librería `transformers` de Python para JavaScript. Utiliza el runtime ONNX (Open Neural Network Exchange) para ejecutar modelos de ML directamente en el navegador o en Node.js, sin necesidad de ningún servidor backend.

### Por qué es importante

| Característica | Transformers.js | API cloud |
|---------------|-----------------|-----------|
| **Privacidad** | Los datos nunca salen del dispositivo | Datos enviados a servidores externos |
| **Backend** | No requiere ninguno | Requiere servidor y API key |
| **Offline** | Funciona completamente offline tras la primera carga | Requiere conexión constante |
| **Coste** | Cero (solo el ancho de banda inicial) | Pago por petición |
| **Latencia** | Depende del dispositivo del usuario | Red + tiempo de servidor |
| **Escalabilidad** | Infinita (la inferencia ocurre en el cliente) | Limitada por la cuota de la API |

### Casos de uso ideales

- **Análisis de sentimientos en tiempo real** sin enviar texto a terceros.
- **Corrección gramatical** o **autocompletado** offline en aplicaciones de escritura.
- **Búsqueda semántica** en una base de datos local del usuario.
- **Clasificación de imágenes** en la cámara del móvil sin latencia de red.
- **Traducción offline** en aplicaciones de idiomas.
- **Accesibilidad**: descripción de imágenes para lectores de pantalla sin conexión.

---

## 2. Instalación

### En Node.js

```bash
npm install @huggingface/transformers
```

### En un proyecto con bundler (Vite, webpack, Parcel)

```bash
npm install @huggingface/transformers
```

No se requiere configuración adicional: los bundlers modernos gestionan automáticamente la carga de los archivos ONNX.

### Verificar la instalación

```javascript
// test-install.mjs
import { pipeline } from "@huggingface/transformers";

console.log("Transformers.js importado correctamente.");

const clasificador = await pipeline("sentiment-analysis");
const resultado = await clasificador("I love working with local AI models!");
console.log(resultado);
// [{ label: 'POSITIVE', score: 0.9998 }]
```

```bash
node test-install.mjs
```

---

## 3. Inferencia básica en Node.js

```javascript
// inferencia-basica.mjs
import { pipeline } from "@huggingface/transformers";

// ─── Análisis de sentimientos ──────────────────────────────────────────────
console.log("Cargando modelo de análisis de sentimientos...");
const analizadorSentimiento = await pipeline(
  "sentiment-analysis",
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
);

const textos = [
  "This product is absolutely amazing and exceeded all my expectations!",
  "The service was terrible and the staff were very rude.",
  "It was okay, nothing special but not bad either.",
];

console.log("\n=== Análisis de sentimientos ===");
for (const texto of textos) {
  const resultado = await analizadorSentimiento(texto);
  const { label, score } = resultado[0];
  const emoji = label === "POSITIVE" ? "✓" : "✗";
  console.log(`${emoji} [${(score * 100).toFixed(1)}%] ${label}: "${texto}"`);
}

// ─── Traducción ────────────────────────────────────────────────────────────
console.log("\nCargando modelo de traducción (español → inglés)...");
const traductor = await pipeline(
  "translation",
  "Xenova/opus-mt-es-en"
);

const frases = [
  "La inteligencia artificial está transformando la industria tecnológica.",
  "Los modelos de lenguaje local protegen la privacidad del usuario.",
  "El aprendizaje automático requiere grandes cantidades de datos.",
];

console.log("\n=== Traducción español → inglés ===");
for (const frase of frases) {
  const traduccion = await traductor(frase);
  console.log(`ES: ${frase}`);
  console.log(`EN: ${traduccion[0].translation_text}\n`);
}

// ─── Resumen de texto ──────────────────────────────────────────────────────
console.log("Cargando modelo de resumen...");
const resumidor = await pipeline(
  "summarization",
  "Xenova/distilbart-cnn-6-6"
);

const textoLargo = `
  Artificial intelligence has evolved dramatically over the past decade. 
  From simple rule-based systems to complex neural networks capable of 
  understanding and generating human language, the field has made remarkable 
  progress. Today, large language models like GPT and Claude can write code, 
  analyze documents, answer questions, and even engage in creative tasks. 
  However, running these models locally presents unique challenges around 
  hardware requirements and model size optimization.
`.trim();

const resumen = await resumidor(textoLargo, { max_length: 60, min_length: 20 });
console.log("\n=== Resumen ===");
console.log("Original:", textoLargo.substring(0, 100) + "...");
console.log("Resumen:", resumen[0].summary_text);
```

---

## 4. En el navegador con bundler

Este ejemplo muestra una página HTML completa con inferencia de IA ejecutándose en el cliente, usando JavaScript vanilla y cargando los módulos a través de un CDN (o un bundler en producción).

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>IA en el navegador — Análisis de sentimientos</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, sans-serif;
      max-width: 600px;
      margin: 40px auto;
      padding: 20px;
      background: #f5f5f5;
    }
    h1 { margin-bottom: 8px; font-size: 1.4rem; }
    .subtitulo { color: #666; font-size: 0.9rem; margin-bottom: 24px; }
    textarea {
      width: 100%;
      height: 120px;
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: 8px;
      font-size: 1rem;
      resize: vertical;
    }
    button {
      margin-top: 12px;
      padding: 10px 24px;
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
    }
    button:disabled { background: #94a3b8; cursor: not-allowed; }
    #estado { margin-top: 16px; font-size: 0.85rem; color: #555; }
    #resultado {
      margin-top: 16px;
      padding: 16px;
      background: white;
      border-radius: 8px;
      border-left: 4px solid #2563eb;
      display: none;
    }
    .etiqueta { font-weight: bold; font-size: 1.1rem; }
    .positivo { color: #16a34a; }
    .negativo { color: #dc2626; }
    .confianza { color: #666; font-size: 0.9rem; margin-top: 4px; }
  </style>
</head>
<body>
  <h1>Análisis de sentimientos local</h1>
  <p class="subtitulo">El modelo se ejecuta en tu navegador. Ningún dato sale de tu dispositivo.</p>

  <textarea
    id="texto"
    placeholder="Escribe aquí el texto a analizar (en inglés)..."
  >I absolutely love this new feature, it makes everything so much easier!</textarea>

  <button id="boton" onclick="analizar()">Analizar sentimiento</button>
  <div id="estado">Cargando modelo (primera vez puede tardar ~30s)...</div>
  <div id="resultado">
    <div class="etiqueta" id="etiqueta"></div>
    <div class="confianza" id="confianza"></div>
  </div>

  <!-- Usar ESM desde CDN para prototipado rápido sin bundler -->
  <script type="module">
    import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js";

    // Configurar para usar modelos cuantizados (más ligeros para el navegador)
    env.allowLocalModels = false;

    let clasificador = null;

    // Cargar el modelo al iniciar la página
    async function inicializar() {
      try {
        clasificador = await pipeline(
          "sentiment-analysis",
          "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
          {
            // Usar modelo cuantizado (INT8): 65 MB en lugar de 250 MB
            quantized: true,
            // Progreso de descarga
            progress_callback: (info) => {
              if (info.status === "downloading") {
                const pct = ((info.loaded / info.total) * 100).toFixed(0);
                document.getElementById("estado").textContent =
                  `Descargando modelo: ${pct}%`;
              }
            },
          }
        );

        document.getElementById("estado").textContent =
          "Modelo listo. Procesamiento 100% local.";
        document.getElementById("boton").disabled = false;
      } catch (error) {
        document.getElementById("estado").textContent =
          "Error al cargar el modelo: " + error.message;
      }
    }

    // Exponer la función al ámbito global para el onclick
    window.analizar = async function () {
      if (!clasificador) return;

      const texto = document.getElementById("texto").value.trim();
      if (!texto) return;

      const boton = document.getElementById("boton");
      boton.disabled = true;
      boton.textContent = "Analizando...";

      try {
        const resultado = await clasificador(texto);
        const { label, score } = resultado[0];
        const esPositivo = label === "POSITIVE";

        const divEtiqueta = document.getElementById("etiqueta");
        divEtiqueta.textContent = esPositivo ? "Positivo" : "Negativo";
        divEtiqueta.className = `etiqueta ${esPositivo ? "positivo" : "negativo"}`;

        document.getElementById("confianza").textContent =
          `Confianza: ${(score * 100).toFixed(1)}%`;

        document.getElementById("resultado").style.display = "block";
      } finally {
        boton.disabled = false;
        boton.textContent = "Analizar sentimiento";
      }
    };

    // Arrancar al cargar la página
    document.getElementById("boton").disabled = true;
    inicializar();
  </script>
</body>
</html>
```

---

## 5. Embeddings en el navegador

```javascript
// embeddings-navegador.mjs
import { pipeline, cos_sim } from "@huggingface/transformers";

// ─── Cargar modelo de embeddings ───────────────────────────────────────────
// all-MiniLM-L6-v2 cuantizado: ~23 MB, 384 dimensiones
console.log("Cargando modelo de embeddings...");
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
  { quantized: true }
);
console.log("Modelo listo.\n");


// ─── Generar embeddings ────────────────────────────────────────────────────
async function generarEmbedding(texto) {
  const salida = await extractor(texto, {
    pooling: "mean",      // Media de todos los tokens: embedding de la oración completa
    normalize: true,      // Normalizar a longitud 1 (necesario para similitud coseno)
  });
  // salida.data es un Float32Array con 384 valores
  return Array.from(salida.data);
}


// ─── Calcular similitud coseno ─────────────────────────────────────────────
function similitudCoseno(a, b) {
  // cos_sim de Transformers.js trabaja con tensores; aquí lo hacemos manual
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}


// ─── Ejemplo: comparar pares de oraciones ─────────────────────────────────
const pares = [
  ["El aprendizaje automático es una rama de la IA", "Machine learning is a branch of AI"],
  ["El fútbol es el deporte más popular", "El aprendizaje automático usa datos"],
  ["Python es ideal para ciencia de datos", "R también se usa en análisis estadístico"],
];

console.log("=== Similitud entre pares de oraciones ===");
for (const [texto1, texto2] of pares) {
  const emb1 = await generarEmbedding(texto1);
  const emb2 = await generarEmbedding(texto2);
  const similitud = similitudCoseno(emb1, emb2);

  console.log(`Texto 1: "${texto1}"`);
  console.log(`Texto 2: "${texto2}"`);
  console.log(`Similitud: ${similitud.toFixed(4)} ${similitud > 0.7 ? "(alta)" : similitud > 0.4 ? "(media)" : "(baja)"}\n`);
}
```

---

## 6. Web Worker para no bloquear el UI

La inferencia con modelos de IA puede tardar varios segundos. Si se ejecuta en el hilo principal, el navegador se congela. La solución es mover el modelo a un Web Worker.

```javascript
// worker.js — Este archivo se ejecuta en un hilo separado
import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;

let modelo = null;

// Escuchar mensajes del hilo principal
self.addEventListener("message", async (evento) => {
  const { tipo, payload, id } = evento.data;

  if (tipo === "INICIALIZAR") {
    try {
      modelo = await pipeline(
        "sentiment-analysis",
        "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
        {
          quantized: true,
          progress_callback: (info) => {
            // Reenviar progreso al hilo principal
            self.postMessage({ tipo: "PROGRESO", payload: info });
          },
        }
      );
      self.postMessage({ tipo: "LISTO", id });
    } catch (error) {
      self.postMessage({ tipo: "ERROR", payload: error.message, id });
    }
  }

  if (tipo === "INFERENCIA") {
    if (!modelo) {
      self.postMessage({ tipo: "ERROR", payload: "Modelo no inicializado", id });
      return;
    }

    try {
      const resultado = await modelo(payload.texto);
      self.postMessage({ tipo: "RESULTADO", payload: resultado, id });
    } catch (error) {
      self.postMessage({ tipo: "ERROR", payload: error.message, id });
    }
  }
});
```

```javascript
// main.js — Hilo principal: nunca se bloquea
class ClienteIA {
  constructor(rutaWorker) {
    this.worker = new Worker(rutaWorker, { type: "module" });
    this.pendientes = new Map(); // id → { resolve, reject }
    this.contadorId = 0;

    this.worker.addEventListener("message", (evento) => {
      this._manejarMensaje(evento.data);
    });
  }

  _manejarMensaje({ tipo, payload, id }) {
    if (tipo === "PROGRESO") {
      // Emitir evento personalizado para que la UI muestre el progreso
      window.dispatchEvent(new CustomEvent("ia:progreso", { detail: payload }));
      return;
    }

    const pendiente = this.pendientes.get(id);
    if (!pendiente) return;

    if (tipo === "ERROR") {
      pendiente.reject(new Error(payload));
    } else {
      pendiente.resolve(payload);
    }
    this.pendientes.delete(id);
  }

  _enviar(tipo, payload = null) {
    return new Promise((resolve, reject) => {
      const id = ++this.contadorId;
      this.pendientes.set(id, { resolve, reject });
      this.worker.postMessage({ tipo, payload, id });
    });
  }

  inicializar() {
    return this._enviar("INICIALIZAR");
  }

  analizar(texto) {
    return this._enviar("INFERENCIA", { texto });
  }

  terminar() {
    this.worker.terminate();
  }
}


// ─── Uso en la página ──────────────────────────────────────────────────────
const ia = new ClienteIA("./worker.js");

// Escuchar progreso de descarga
window.addEventListener("ia:progreso", (evento) => {
  const info = evento.detail;
  if (info.status === "downloading") {
    const pct = ((info.loaded / info.total) * 100).toFixed(0);
    document.getElementById("progreso").textContent = `Descargando: ${pct}%`;
  }
});

// Inicializar sin bloquear el UI
await ia.inicializar();
document.getElementById("progreso").textContent = "Modelo listo";

// El usuario puede seguir interactuando con la página mientras se hace la inferencia
document.getElementById("btn-analizar").addEventListener("click", async () => {
  const texto = document.getElementById("entrada").value;
  const resultado = await ia.analizar(texto);
  document.getElementById("resultado").textContent = JSON.stringify(resultado, null, 2);
});
```

---

## 7. Caso práctico: buscador semántico offline

Un buscador semántico completo que funciona íntegramente en el navegador: indexa una lista de textos y permite buscar por significado, no solo por palabras clave. No requiere ningún servidor.

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Buscador Semántico Offline</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; }
    h1 { margin-bottom: 4px; }
    .badge {
      display: inline-block; background: #dcfce7; color: #166534;
      padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-bottom: 20px;
    }
    #barra-busqueda {
      display: flex; gap: 8px; margin-bottom: 20px;
    }
    input {
      flex: 1; padding: 10px 14px; border: 1px solid #ddd;
      border-radius: 8px; font-size: 1rem;
    }
    button {
      padding: 10px 20px; background: #2563eb; color: white;
      border: none; border-radius: 8px; cursor: pointer; font-size: 1rem;
    }
    button:disabled { background: #94a3b8; }
    #estado { font-size: 0.85rem; color: #666; margin-bottom: 16px; }
    .resultado {
      padding: 14px; background: #f8fafc; border-radius: 8px;
      margin-bottom: 8px; border-left: 3px solid #2563eb;
    }
    .resultado-texto { font-size: 0.95rem; }
    .resultado-score {
      font-size: 0.8rem; color: #888; margin-top: 4px;
    }
    .barra-similitud {
      height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: 6px;
    }
    .barra-similitud-fill {
      height: 100%; background: #2563eb; border-radius: 2px;
      transition: width 0.3s ease;
    }
  </style>
</head>
<body>
  <h1>Buscador Semántico Offline</h1>
  <span class="badge">100% local · Sin servidor · Sin cookies</span>

  <div id="barra-busqueda">
    <input
      type="text"
      id="consulta"
      placeholder="Busca por significado, no por palabras exactas..."
      disabled
    />
    <button id="btn-buscar" disabled onclick="buscar()">Buscar</button>
  </div>

  <div id="estado">Cargando modelo de embeddings...</div>
  <div id="resultados"></div>

  <script type="module">
    import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js";

    // ─── Base de conocimiento local ────────────────────────────────────────
    const BASE_CONOCIMIENTO = [
      "Python es un lenguaje de programación de alto nivel conocido por su legibilidad.",
      "JavaScript es el lenguaje de programación principal para desarrollo web frontend.",
      "El aprendizaje automático permite a los sistemas aprender de los datos.",
      "Las redes neuronales profundas imitan el funcionamiento del cerebro humano.",
      "Docker permite empaquetar aplicaciones en contenedores portables.",
      "Git es un sistema de control de versiones distribuido creado por Linus Torvalds.",
      "React es una librería de JavaScript para construir interfaces de usuario.",
      "PostgreSQL es un sistema de base de datos relacional de código abierto.",
      "La criptografía protege la información mediante técnicas matemáticas.",
      "El cloud computing permite acceder a recursos informáticos bajo demanda.",
      "Los algoritmos de búsqueda encuentran elementos en estructuras de datos.",
      "La inteligencia artificial busca crear sistemas que realizan tareas inteligentes.",
    ];

    let extractor = null;
    let embeddingsBase = null;

    // ─── Generar embedding de un texto ────────────────────────────────────
    async function generarEmbedding(texto) {
      const salida = await extractor(texto, { pooling: "mean", normalize: true });
      return Array.from(salida.data);
    }

    // ─── Similitud coseno ──────────────────────────────────────────────────
    function similitudCoseno(a, b) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    // ─── Inicializar modelo e indexar la base ──────────────────────────────
    async function inicializar() {
      const estado = document.getElementById("estado");

      try {
        extractor = await pipeline(
          "feature-extraction",
          "Xenova/all-MiniLM-L6-v2",
          {
            quantized: true,
            progress_callback: (info) => {
              if (info.status === "downloading") {
                const pct = ((info.loaded / info.total) * 100).toFixed(0);
                estado.textContent = `Descargando modelo: ${pct}%`;
              }
            },
          }
        );

        // Indexar todos los documentos
        estado.textContent = "Indexando documentos...";
        embeddingsBase = await Promise.all(
          BASE_CONOCIMIENTO.map((texto) => generarEmbedding(texto))
        );

        estado.textContent = `Listo. ${BASE_CONOCIMIENTO.length} documentos indexados. Búsqueda 100% local.`;
        document.getElementById("consulta").disabled = false;
        document.getElementById("btn-buscar").disabled = false;
        document.getElementById("consulta").focus();
      } catch (error) {
        estado.textContent = "Error: " + error.message;
      }
    }

    // ─── Realizar búsqueda ─────────────────────────────────────────────────
    window.buscar = async function () {
      const consulta = document.getElementById("consulta").value.trim();
      if (!consulta || !extractor) return;

      const btn = document.getElementById("btn-buscar");
      btn.disabled = true;
      btn.textContent = "Buscando...";
      document.getElementById("estado").textContent = "Calculando similitudes...";

      try {
        const embeddingConsulta = await generarEmbedding(consulta);

        // Calcular similitud con todos los documentos
        const puntuaciones = BASE_CONOCIMIENTO.map((texto, i) => ({
          texto,
          similitud: similitudCoseno(embeddingConsulta, embeddingsBase[i]),
        }));

        // Ordenar por similitud descendente
        puntuaciones.sort((a, b) => b.similitud - a.similitud);
        const top5 = puntuaciones.slice(0, 5);

        // Renderizar resultados
        const divResultados = document.getElementById("resultados");
        divResultados.innerHTML = top5.map((r, idx) => `
          <div class="resultado">
            <div class="resultado-texto">${r.texto}</div>
            <div class="resultado-score">
              Similitud: ${(r.similitud * 100).toFixed(1)}%
            </div>
            <div class="barra-similitud">
              <div class="barra-similitud-fill" style="width: ${(r.similitud * 100).toFixed(1)}%"></div>
            </div>
          </div>
        `).join("");

        document.getElementById("estado").textContent =
          `${top5.length} resultados para "${consulta}"`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Buscar";
      }
    };

    // Permitir buscar con Enter
    document.getElementById("consulta").addEventListener("keydown", (e) => {
      if (e.key === "Enter") window.buscar();
    });

    inicializar();
  </script>
</body>
</html>
```

---

## 8. Extensiones sugeridas

- **Persistencia con IndexedDB**: guarda los embeddings ya calculados en el almacenamiento local del navegador para evitar recalcularlos en cada visita.
- **Inferencia de imágenes**: usa `pipeline("image-classification", "Xenova/vit-base-patch16-224")` para clasificar imágenes directamente desde `<input type="file">`.
- **Modelos de voz**: `pipeline("automatic-speech-recognition", "Xenova/whisper-tiny")` transcribe audio del micrófono del usuario sin ningún servidor.
- **Cache de modelos con Service Worker**: intercepta las peticiones de descarga del modelo ONNX y las sirve desde la caché del navegador para funcionar completamente offline tras la primera carga.
- **WebGPU**: Transformers.js soporta WebGPU para aceleración hardware en Chrome 113+, con mejoras de velocidad de 3-5x frente a WebAssembly.

---

**Siguiente:** [04 — Local vs cloud: cuándo usar cada opción](./04-comparativa-local-cloud.md)
