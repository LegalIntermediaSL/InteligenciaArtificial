# 11 — Tokenización en Profundidad

> **Bloque:** LLMs · **Nivel:** Intermedio · **Tiempo estimado:** 35 min

---

## Índice

1. [Qué es un token](#1-qué-es-un-token)
2. [Algoritmos de tokenización](#2-algoritmos-de-tokenización)
3. [Tokenizadores en práctica](#3-tokenizadores-en-práctica)
4. [Impacto en costos y latencia](#4-impacto-en-costos-y-latencia)
5. [Casos especiales: código, idiomas, emojis](#5-casos-especiales-código-idiomas-emojis)
6. [Extensiones sugeridas](#6-extensiones-sugeridas)

---

## 1. Qué es un token

Un **token** es la unidad mínima de texto que procesa un LLM. No equivale a una palabra: puede ser una sílaba, un carácter, una palabra completa o incluso varias palabras juntas dependiendo del vocabulario del tokenizador.

```
"Hola mundo" → ["Hola", " mundo"]   → 2 tokens
"tokenización" → ["token", "ización"] → 2 tokens  
"AI"          → ["AI"]               → 1 token
```

Los modelos trabajan con IDs numéricos, no con texto:
```
["Hola", " mundo"] → [29636, 13978]
```

---

## 2. Algoritmos de tokenización

### Byte-Pair Encoding (BPE)

BPE es el algoritmo más común (GPT, LLaMA, Claude). Parte de caracteres individuales y fusiona los pares más frecuentes iterativamente.

```python
# Demostración simplificada de BPE
def bpe_simplificado(corpus: list[str], n_fusiones: int = 10) -> dict:
    """Versión simplificada de BPE para ilustrar el algoritmo."""
    from collections import Counter

    # Inicializar: dividir en caracteres + marcador de fin de palabra
    vocab = Counter()
    for palabra in corpus:
        vocab[" ".join(list(palabra)) + " </w>"] += 1

    fusiones = []
    for _ in range(n_fusiones):
        pares = Counter()
        for seq, freq in vocab.items():
            simbolos = seq.split()
            for i in range(len(simbolos) - 1):
                pares[(simbolos[i], simbolos[i+1])] += freq

        if not pares:
            break

        mejor_par = max(pares, key=pares.get)
        fusiones.append(mejor_par)

        # Aplicar fusión
        nuevo_vocab = {}
        bigrama = " ".join(mejor_par)
        reemplazo = "".join(mejor_par)
        for seq in vocab:
            nuevo_seq = seq.replace(bigrama, reemplazo)
            nuevo_vocab[nuevo_seq] = vocab[seq]
        vocab = nuevo_vocab

    return {"vocab": dict(vocab), "fusiones": fusiones}


corpus = ["bajo", "bajada", "bajar", "base", "basa", "básico"]
resultado = bpe_simplificado(corpus, n_fusiones=5)
print("Fusiones aprendidas:", resultado["fusiones"])
```

### WordPiece (BERT)

Similar a BPE pero maximiza la probabilidad del corpus en lugar de la frecuencia de pares. Usa el prefijo `##` para subpalabras que no inician una palabra.

### SentencePiece (LLaMA, Mistral)

Trabaja directamente sobre bytes Unicode, tratando el texto como una secuencia de caracteres. No requiere pretokenización por espacios — funciona bien con cualquier idioma.

---

## 3. Tokenizadores en práctica

```python
# tokenizers_practica.py
from transformers import AutoTokenizer
import tiktoken  # para modelos de OpenAI

# ── Tokenizador de LLaMA 3 (SentencePiece / BPE) ──────────────────────────
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

texto = "La tokenización es fundamental para entender los LLMs."
tokens = tokenizer_llama.encode(texto)
tokens_str = tokenizer_llama.convert_ids_to_tokens(tokens)

print(f"Texto: {texto}")
print(f"Tokens IDs: {tokens}")
print(f"Tokens string: {tokens_str}")
print(f"Número de tokens: {len(tokens)}")


# ── Tokenizador de OpenAI (tiktoken) ───────────────────────────────────────
enc_gpt4 = tiktoken.encoding_for_model("gpt-4o")

tokens_gpt4 = enc_gpt4.encode(texto)
print(f"\nGPT-4o tokens: {len(tokens_gpt4)}")
print(f"Decoded: {[enc_gpt4.decode([t]) for t in tokens_gpt4]}")


# ── Comparar tokenización entre modelos ────────────────────────────────────
def comparar_tokenizadores(texto: str) -> dict:
    """Compara el número de tokens para diferentes modelos."""
    resultados = {}

    # tiktoken para modelos OpenAI
    for modelo in ["gpt-4o", "gpt-3.5-turbo"]:
        enc = tiktoken.encoding_for_model(modelo)
        resultados[modelo] = len(enc.encode(texto))

    # transformers para modelos open-source
    for modelo_hf in [
        "meta-llama/Meta-Llama-3-8B",
        "mistralai/Mistral-7B-v0.1"
    ]:
        try:
            tok = AutoTokenizer.from_pretrained(modelo_hf)
            resultados[modelo_hf.split("/")[-1]] = len(tok.encode(texto))
        except Exception:
            pass

    return resultados


# Ejemplo
textos_prueba = [
    "Hello world",
    "Hola mundo",
    "こんにちは世界",  # japonés
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]

for t in textos_prueba:
    comp = comparar_tokenizadores(t)
    print(f"\n'{t[:50]}'")
    for modelo, n in comp.items():
        print(f"  {modelo}: {n} tokens")
```

---

## 4. Impacto en costos y latencia

```python
# cost_calculator.py

PRECIOS_POR_MILLON = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def calcular_costo(
    tokens_input: int,
    tokens_output: int,
    modelo: str
) -> dict:
    if modelo not in PRECIOS_POR_MILLON:
        raise ValueError(f"Modelo {modelo} no en tabla de precios")

    precios = PRECIOS_POR_MILLON[modelo]
    costo_input = (tokens_input / 1_000_000) * precios["input"]
    costo_output = (tokens_output / 1_000_000) * precios["output"]

    return {
        "modelo": modelo,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "costo_input_usd": round(costo_input, 6),
        "costo_output_usd": round(costo_output, 6),
        "costo_total_usd": round(costo_input + costo_output, 6),
    }


def optimizar_prompt(prompt: str, max_tokens: int = 2000) -> str:
    """Sugiere cómo reducir el número de tokens de un prompt."""
    tokens_actuales = len(prompt) // 4  # estimación

    sugerencias = []
    if "Por favor," in prompt or "Podrías" in prompt:
        sugerencias.append("Eliminar cortesías ('Por favor', 'Podrías')")
    if "\n\n\n" in prompt:
        sugerencias.append("Reducir saltos de línea múltiples")
    if len([s for s in prompt.split(".") if len(s) > 200]) > 2:
        sugerencias.append("Dividir frases largas")

    if sugerencias:
        print("Sugerencias para reducir tokens:")
        for s in sugerencias:
            print(f"  - {s}")

    return prompt  # devolver prompt original (las optimizaciones son manuales)


# Comparativa de costos para 1 millón de llamadas
print("Costo por 1M llamadas (1K tokens input, 200 tokens output):\n")
for modelo in PRECIOS_POR_MILLON:
    costo = calcular_costo(1_000 * 1_000_000, 200 * 1_000_000, modelo)
    print(f"{modelo}: ${costo['costo_total_usd']:,.0f}")
```

---

## 5. Casos especiales: código, idiomas, emojis

```python
# special_cases.py
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")


def analizar_densidad_tokens(texto: str) -> dict:
    """Analiza cuántos caracteres hay por token en un texto dado."""
    tokens = enc.encode(texto)
    return {
        "texto": texto[:50] + "..." if len(texto) > 50 else texto,
        "n_chars": len(texto),
        "n_tokens": len(tokens),
        "chars_por_token": round(len(texto) / len(tokens), 2)
    }


casos = {
    "Inglés común": "The quick brown fox jumps over the lazy dog",
    "Español": "El veloz zorro marrón salta sobre el perro perezoso",
    "Chino": "快速的棕色狐狸跳过懒狗",
    "Árabe": "الثعلب البني السريع يقفز فوق الكلب الكسول",
    "Código Python": "def hello(name: str) -> str:\n    return f'Hello, {name}!'",
    "JSON": '{"nombre": "Juan", "edad": 30, "activo": true}',
    "Emojis": "🚀🤖💡🔥✨🌟💪🎯",
    "URLs": "https://www.example.com/path/to/resource?param=value&other=123",
    "Números": "3.14159265358979 1000000 0.001 1e-10",
}

print(f"{'Tipo':<20} {'Chars':>8} {'Tokens':>8} {'Chars/Token':>12}")
print("-" * 52)
for tipo, texto in casos.items():
    r = analizar_densidad_tokens(texto)
    print(f"{tipo:<20} {r['n_chars']:>8} {r['n_tokens']:>8} {r['chars_por_token']:>12.2f}")
```

**Observaciones clave:**
- El inglés es el idioma más eficiente (~4 chars/token)
- El español es ligeramente menos eficiente (~3.5 chars/token) por tildes y ñ
- El chino/japonés/coreano puede ser muy ineficiente (1-2 chars/token)
- El código suele tokenizarse eficientemente porque las palabras clave son comunes
- Los emojis pueden ser 1-3 tokens por carácter

---

## 6. Extensiones sugeridas

- **Token healing**: algunos frameworks corrigen los "cortes" de tokenización en el boundary del prompt
- **Tokenización con vocabulario personalizado**: fine-tuning del tokenizador para dominios específicos (médico, legal)
- **Análisis de vocabulario**: visualizar qué tokens son más frecuentes en tu dataset con `Counter`

---

**Anterior:** [10 — Gestión del contexto](./10-gestion-contexto.md) · **Siguiente:** [12 — Function Calling avanzado](./12-function-calling-avanzado.md)
