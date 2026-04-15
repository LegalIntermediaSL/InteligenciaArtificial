# 02 — Inferencia local con Hugging Face Transformers

> **Bloque:** IA local · **Nivel:** Intermedio · **Tiempo estimado:** 50 min

---

## Índice

1. [Cuándo usar Transformers vs Ollama](#1-cuándo-usar-transformers-vs-ollama)
2. [Instalación y configuración](#2-instalación-y-configuración)
3. [Pipeline de inferencia](#3-pipeline-de-inferencia)
4. [Optimización de memoria con cuantización](#4-optimización-de-memoria-con-cuantización)
5. [Modelos de embeddings locales](#5-modelos-de-embeddings-locales)
6. [Clasificación de texto sin fine-tuning](#6-clasificación-de-texto-sin-fine-tuning)
7. [Guardar y reutilizar modelos descargados](#7-guardar-y-reutilizar-modelos-descargados)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Cuándo usar Transformers vs Ollama

Ambas herramientas permiten ejecutar modelos localmente, pero apuntan a casos de uso diferentes.

| Dimensión | Hugging Face Transformers | Ollama |
|-----------|--------------------------|--------|
| **Facilidad de uso** | Requiere más código | CLI lista para usar en minutos |
| **Control del modelo** | Total: acceso a logits, hidden states, capas | API de alto nivel únicamente |
| **Fine-tuning** | Soporte nativo (Trainer, PEFT, LoRA) | No soportado |
| **Cuantización** | bitsandbytes, GPTQ, AWQ | GGUF integrado |
| **Ecosistema** | 500k+ modelos en el Hub | ~200 modelos curados |
| **Integración Python** | Profunda (PyTorch, JAX) | REST API / SDK ligero |
| **Casos de uso** | Investigación, producción ML, pipelines complejos | Prototipos, chatbots, RAG sencillo |

**Elige Transformers cuando** necesitas fine-tuning, acceso a representaciones internas del modelo, máximo control sobre el proceso de inferencia o integración con pipelines PyTorch existentes.

**Elige Ollama cuando** quieres empezar rápido, no necesitas modificar el modelo y priorizas la simplicidad operativa.

---

## 2. Instalación y configuración

```bash
pip install transformers torch sentence-transformers
```

Para soporte de cuantización (requiere GPU NVIDIA con CUDA):

```bash
pip install bitsandbytes accelerate
```

### Detectar y configurar el dispositivo

```python
import torch

def obtener_dispositivo() -> torch.device:
    """Selecciona automáticamente el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        nombre_gpu = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU CUDA: {nombre_gpu} ({vram_gb:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        dispositivo = torch.device("mps")
        print("GPU Apple Silicon (MPS)")
    else:
        dispositivo = torch.device("cpu")
        print("CPU (la inferencia será más lenta)")

    return dispositivo


dispositivo = obtener_dispositivo()
print(f"Dispositivo seleccionado: {dispositivo}")
```

---

## 3. Pipeline de inferencia

La clase `pipeline` de Transformers ofrece la forma más sencilla de cargar un modelo y realizar inferencia. Aquí usamos `microsoft/phi-3-mini-4k-instruct`, un modelo de 3.8B parámetros con excelente relación tamaño/calidad.

```python
from transformers import pipeline
import torch

# ─── Cargar el modelo ──────────────────────────────────────────────────────
# La primera ejecución descarga el modelo (~2.4 GB)
generador = pipeline(
    "text-generation",
    model="microsoft/phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,       # FP16 usa la mitad de VRAM
    device_map="auto",               # Distribuye automáticamente entre GPU/CPU
    trust_remote_code=True,          # Requerido por algunos modelos del Hub
)

print("Modelo cargado correctamente.")


# ─── Conversación completa ─────────────────────────────────────────────────
def chat(
    mensajes: list[dict],
    max_tokens: int = 512,
    temperatura: float = 0.7,
) -> str:
    """Genera una respuesta dado el historial de mensajes."""
    resultado = generador(
        mensajes,
        max_new_tokens=max_tokens,
        temperature=temperatura,
        do_sample=True,
        return_full_text=False,      # Solo devuelve la respuesta, no el prompt
    )
    return resultado[0]["generated_text"]


# ─── Ejemplo: conversación de varios turnos ───────────────────────────────
historial = [
    {
        "role": "system",
        "content": "Eres un experto en ciencia de datos. Responde en español, de forma clara y concisa.",
    }
]

preguntas = [
    "¿Qué es el sobreajuste (overfitting) en machine learning?",
    "¿Cómo puedo detectarlo en mi modelo?",
    "Dame un ejemplo de código en Python para visualizarlo",
]

for pregunta in preguntas:
    historial.append({"role": "user", "content": pregunta})

    respuesta = chat(historial)
    historial.append({"role": "assistant", "content": respuesta})

    print(f"\nUsuario: {pregunta}")
    print(f"Asistente: {respuesta}")
    print("─" * 60)
```

---

## 4. Optimización de memoria con cuantización

La cuantización reduce la precisión numérica de los pesos del modelo (de FP32 a INT4 o INT8), lo que disminuye drásticamente el uso de memoria con una pérdida mínima de calidad.

```bash
pip install bitsandbytes accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ─── Configuración de cuantización a 4 bits ────────────────────────────────
# Solo disponible con GPU NVIDIA (CUDA). No funciona en MPS ni CPU.
config_cuantizacion = BitsAndBytesConfig(
    load_in_4bit=True,                          # Activar cuantización 4-bit
    bnb_4bit_compute_dtype=torch.float16,       # Precisión para cálculos intermedios
    bnb_4bit_use_double_quant=True,             # Cuantización doble: ahorra más memoria
    bnb_4bit_quant_type="nf4",                  # NF4: mejor para modelos de lenguaje
)

nombre_modelo = "microsoft/phi-3-mini-4k-instruct"

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

# Cargar modelo con cuantización 4-bit (~1.2 GB VRAM vs ~7 GB en FP32)
modelo = AutoModelForCausalLM.from_pretrained(
    nombre_modelo,
    quantization_config=config_cuantizacion,
    device_map="auto",
    trust_remote_code=True,
)

print("Modelo cuantizado cargado.")
print(f"Memoria GPU usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ─── Inferencia con modelo cuantizado ─────────────────────────────────────
def generar_respuesta(prompt: str, max_tokens: int = 256) -> str:
    # Formatear el prompt en el formato de chat del modelo
    mensajes = [{"role": "user", "content": prompt}]
    texto_formateado = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenizar
    inputs = tokenizer(texto_formateado, return_tensors="pt").to(modelo.device)

    # Generar
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decodificar solo los tokens nuevos (excluir el prompt)
    tokens_nuevos = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(tokens_nuevos, skip_special_tokens=True)


respuesta = generar_respuesta("Explica la diferencia entre RAM y VRAM en 3 líneas")
print(f"Respuesta: {respuesta}")
```

### Comparativa de uso de memoria

| Precisión | Memoria (7B params) | Calidad relativa |
|-----------|---------------------|------------------|
| FP32 (completo) | ~28 GB | 100% |
| FP16 | ~14 GB | ~99% |
| INT8 | ~7 GB | ~97% |
| INT4 (NF4) | ~3.5 GB | ~95% |

---

## 5. Modelos de embeddings locales

Los embeddings son representaciones vectoriales del texto que capturan su significado semántico. `sentence-transformers` proporciona modelos preentrenados excelentes para esta tarea.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# ─── Cargar modelo de embeddings ───────────────────────────────────────────
# all-MiniLM-L6-v2: 80 MB, 384 dimensiones, muy rápido y preciso
modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Modelo de embeddings cargado. Dimensiones: {modelo_emb.get_sentence_embedding_dimension()}")


# ─── Generar embeddings ────────────────────────────────────────────────────
textos = [
    "El aprendizaje profundo usa redes neuronales multicapa",
    "Las redes neuronales profundas tienen múltiples capas ocultas",
    "Python es el lenguaje más popular para la ciencia de datos",
    "El cielo es azul por la dispersión de Rayleigh",
    "La ciencia de datos usa Python con frecuencia",
]

# Los embeddings se calculan en lote (más eficiente)
embeddings = modelo_emb.encode(textos, convert_to_numpy=True, show_progress_bar=True)
print(f"\nShape de embeddings: {embeddings.shape}")  # (5, 384)


# ─── Similitud coseno ──────────────────────────────────────────────────────
def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\nMatriz de similitudes semánticas:")
print(f"{'':45}", end="")
for i in range(len(textos)):
    print(f"T{i+1:>4}", end="")
print()

for i, (texto_i, emb_i) in enumerate(zip(textos, embeddings)):
    print(f"T{i+1} {texto_i[:42]:<42}", end="")
    for j, emb_j in enumerate(embeddings):
        sim = similitud_coseno(emb_i, emb_j)
        print(f"{sim:>5.2f}", end="")
    print()


# ─── Búsqueda semántica ────────────────────────────────────────────────────
def buscar(consulta: str, corpus: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """Busca los textos más similares a la consulta."""
    emb_consulta = modelo_emb.encode([consulta], convert_to_numpy=True)[0]
    emb_corpus = modelo_emb.encode(corpus, convert_to_numpy=True)

    similitudes = [similitud_coseno(emb_consulta, emb) for emb in emb_corpus]
    indices_ordenados = np.argsort(similitudes)[::-1][:top_k]

    return [(corpus[i], similitudes[i]) for i in indices_ordenados]


consulta = "lenguajes de programación para machine learning"
resultados = buscar(consulta, textos)

print(f"\nBúsqueda: '{consulta}'")
for texto, sim in resultados:
    print(f"  [{sim:.3f}] {texto}")
```

---

## 6. Clasificación de texto sin fine-tuning

La clasificación zero-shot permite clasificar texto en categorías arbitrarias sin haber entrenado el modelo para esa tarea específica.

```python
from transformers import pipeline

# ─── Cargar pipeline de zero-shot classification ───────────────────────────
# facebook/bart-large-mnli: 400 MB, excelente para zero-shot en inglés
# Para español: PlanTL-GOB-ES/roberta-large-bne-capitel-ner o traducir las etiquetas
clasificador = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if __import__("torch").cuda.is_available() else -1,
)

print("Clasificador zero-shot cargado.\n")


# ─── Clasificación de sentimiento y tema ───────────────────────────────────
textos_a_clasificar = [
    "The quarterly results exceeded all analyst expectations with 40% revenue growth.",
    "The new medication shows severe side effects in 30% of patients during trials.",
    "Scientists discover a new species of deep-sea fish near the Mariana Trench.",
    "The football team won the championship after a dramatic penalty shootout.",
]

# Clasificar por tema
etiquetas_tema = ["business", "health", "science", "sports", "technology", "politics"]

print("=== Clasificación por tema ===")
for texto in textos_a_clasificar:
    resultado = clasificador(texto, candidate_labels=etiquetas_tema)
    etiqueta_top = resultado["labels"][0]
    confianza = resultado["scores"][0]
    print(f"  Texto: {texto[:60]}...")
    print(f"  Tema: {etiqueta_top} (confianza: {confianza:.2%})\n")

# Clasificación multi-etiqueta (un texto puede tener múltiples categorías)
print("=== Clasificación multi-etiqueta ===")
texto_complejo = "The tech startup raised $50M to develop AI-powered medical diagnostics."
etiquetas_multiples = ["technology", "healthcare", "finance", "artificial intelligence", "startup"]

resultado = clasificador(
    texto_complejo,
    candidate_labels=etiquetas_multiples,
    multi_label=True,  # Permite múltiples etiquetas simultáneas
)

print(f"Texto: {texto_complejo}")
print("Etiquetas detectadas:")
for etiqueta, puntuacion in zip(resultado["labels"], resultado["scores"]):
    if puntuacion > 0.5:
        print(f"  [{'X' if puntuacion > 0.5 else ' '}] {etiqueta}: {puntuacion:.2%}")
```

---

## 7. Guardar y reutilizar modelos descargados

Por defecto, Transformers guarda los modelos en `~/.cache/huggingface/hub`. Puedes cambiar esta ruta y cargar los modelos desde disco sin necesidad de internet.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os

# ─── Configurar directorio de caché personalizado ─────────────────────────
DIRECTORIO_MODELOS = Path("/ruta/a/mis/modelos")
DIRECTORIO_MODELOS.mkdir(parents=True, exist_ok=True)

# Opción 1: variable de entorno (afecta a toda la sesión)
os.environ["HF_HOME"] = str(DIRECTORIO_MODELOS)

# Opción 2: especificar cache_dir en cada llamada (más explícito)
nombre_modelo = "microsoft/phi-3-mini-4k-instruct"
ruta_local = DIRECTORIO_MODELOS / "phi-3-mini-4k-instruct"


def descargar_modelo(nombre: str, ruta: Path) -> None:
    """Descarga un modelo del Hub y lo guarda en disco."""
    if ruta.exists():
        print(f"El modelo ya está en disco: {ruta}")
        return

    print(f"Descargando {nombre}...")
    tokenizer = AutoTokenizer.from_pretrained(nombre, trust_remote_code=True)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, trust_remote_code=True)

    # Guardar en disco
    tokenizer.save_pretrained(str(ruta))
    modelo.save_pretrained(str(ruta))
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo_local(ruta: Path):
    """Carga un modelo desde disco sin acceso a internet."""
    print(f"Cargando modelo desde: {ruta}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ruta),
        local_files_only=True,      # Prohibir cualquier descarga
        trust_remote_code=True,
    )
    modelo = AutoModelForCausalLM.from_pretrained(
        str(ruta),
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
    )

    print("Modelo cargado desde disco.")
    return tokenizer, modelo


# ─── Flujo completo ────────────────────────────────────────────────────────
# Paso 1: descargar una vez (requiere internet)
descargar_modelo(nombre_modelo, ruta_local)

# Paso 2: cargar desde disco (funciona offline)
tokenizer, modelo = cargar_modelo_local(ruta_local)

# Paso 3: usar el modelo
def inferencia(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)
    with __import__("torch").no_grad():
        outputs = modelo.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nPrueba offline:")
respuesta = inferencia("Hola, ¿qué es Python en una oración?")
print(respuesta)


# ─── Ver todos los modelos en caché ───────────────────────────────────────
from huggingface_hub import scan_cache_dir

info_cache = scan_cache_dir()
print(f"\nModelos en caché: {len(list(info_cache.repos))}")
print(f"Tamaño total: {info_cache.size_on_disk / 1e9:.2f} GB")

for repo in info_cache.repos:
    tamano_gb = repo.size_on_disk / 1e9
    print(f"  {repo.repo_id}: {tamano_gb:.2f} GB")
```

---

## 8. Extensiones sugeridas

- **Fine-tuning con LoRA**: usa `peft` y `trl` para adaptar modelos preentrenados a tareas específicas con pocos recursos (`LoraConfig`, `SFTTrainer`).
- **Evaluate**: mide el rendimiento de tus modelos con métricas estándar (`pip install evaluate`, BLEU, ROUGE, perplexity).
- **Optimum**: exporta modelos a ONNX para inferencia más rápida en CPU con `pip install optimum`.
- **Text Generation Inference (TGI)**: servidor de producción de Hugging Face para modelos grandes, con batching continuo y soporte multi-GPU.
- **Modelos recomendados para explorar**: `mistralai/Mistral-7B-Instruct-v0.3`, `google/gemma-2-9b-it`, `Qwen/Qwen2-7B-Instruct`.

---

**Siguiente:** [03 — IA en el navegador con Transformers.js](./03-transformers-js-navegador.md)
