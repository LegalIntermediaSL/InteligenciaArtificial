# 02 — Prompt Engineering

> **Bloque:** LLMs · **Nivel:** Introductorio-Intermedio · **Tiempo estimado:** 30 min

---

## Índice

1. [¿Qué es el prompt engineering?](#1-qué-es-el-prompt-engineering)
2. [Anatomía de un prompt](#2-anatomía-de-un-prompt)
3. [Técnicas fundamentales](#3-técnicas-fundamentales)
4. [Técnicas avanzadas](#4-técnicas-avanzadas)
5. [Errores comunes](#5-errores-comunes)
6. [Buenas prácticas](#6-buenas-prácticas)
7. [Diferencias entre modelos](#7-diferencias-entre-modelos)
8. [Cheatsheet de referencia](#8-cheatsheet-de-referencia)

---

## 1. ¿Qué es el prompt engineering?

El **prompt engineering** es la práctica de diseñar y optimizar las instrucciones (prompts) que se envían a un LLM para obtener los resultados deseados de forma consistente.

No es programación tradicional: no se escriben reglas explícitas, sino que se guía al modelo a través del lenguaje natural. Sin embargo, pequeñas diferencias en la formulación pueden tener un impacto enorme en la calidad del resultado.

> Un mismo modelo puede dar respuestas radicalmente distintas ante prompts que parecen equivalentes.

**¿Por qué importa?**

- Ahorra iteraciones y tiempo
- Reduce alucinaciones y errores
- Permite extraer capacidades avanzadas del modelo
- Es la base de cualquier aplicación de IA basada en LLMs

---

## 2. Anatomía de un prompt

Un prompt bien estructurado suele tener algunos o todos estos componentes:

```
┌──────────────────────────────────────────────┐
│  SYSTEM PROMPT (instrucciones del sistema)   │
│  Define el rol, tono y restricciones         │
├──────────────────────────────────────────────┤
│  CONTEXTO                                    │
│  Información de fondo relevante              │
├──────────────────────────────────────────────┤
│  INSTRUCCIÓN                                 │
│  La tarea concreta que debe realizar         │
├──────────────────────────────────────────────┤
│  EJEMPLOS (opcionales)                       │
│  Demuestran el formato o estilo esperado     │
├──────────────────────────────────────────────┤
│  DATOS DE ENTRADA                            │
│  El material sobre el que operar             │
├──────────────────────────────────────────────┤
│  FORMATO DE SALIDA                           │
│  Cómo debe estructurarse la respuesta        │
└──────────────────────────────────────────────┘
```

### Ejemplo completo

```
SYSTEM: Eres un asistente legal especializado en derecho mercantil español.
        Responde siempre en español. Sé conciso y preciso.
        Si no conoces algo con certeza, indícalo explícitamente.

CONTEXTO: Estoy redactando los estatutos de una SL unipersonal.

INSTRUCCIÓN: Revisa la siguiente cláusula y señala cualquier problema legal:

DATOS: "El administrador podrá realizar cualquier acto en nombre de la sociedad
        sin necesidad de autorización de la junta."

FORMATO: Responde con: 1) Problemas detectados, 2) Riesgo legal, 3) Sugerencia de redacción.
```

---

## 3. Técnicas fundamentales

### 3.1 Zero-shot prompting

Pedir la tarea directamente sin dar ejemplos. Funciona bien para tareas sencillas y bien definidas.

```
Traduce al inglés: "La inteligencia artificial cambiará el mundo"
```

```
Clasifica el sentimiento de este texto como POSITIVO, NEGATIVO o NEUTRO:
"El servicio fue aceptable pero nada especial"
```

---

### 3.2 Few-shot prompting

Proporcionar 2-5 ejemplos del formato entrada→salida esperado antes de la pregunta real. Muy eficaz para tareas con formato específico.

```
Clasifica el sentimiento:

Texto: "El producto llegó roto" → NEGATIVO
Texto: "Superó todas mis expectativas" → POSITIVO
Texto: "El envío tardó lo habitual" → NEUTRO

Texto: "La interfaz es confusa pero funciona" → ?
```

**Cuándo usarlo:**
- Cuando el modelo no entiende bien el formato deseado
- Para tareas muy específicas del dominio
- Para mantener consistencia en el estilo de respuesta

---

### 3.3 Chain-of-Thought (CoT)

Pedir al modelo que "piense en voz alta" antes de dar la respuesta final. Mejora drásticamente el razonamiento en problemas complejos.

**Sin CoT:**
```
¿Cuántos días hay entre el 15 de marzo y el 10 de junio?
→ 86 días  ← probablemente incorrecto
```

**Con CoT:**
```
¿Cuántos días hay entre el 15 de marzo y el 10 de junio?
Piensa paso a paso antes de responder.

→ Del 15 al 31 de marzo: 16 días
  Todo abril: 30 días
  Todo mayo: 31 días
  Del 1 al 10 de junio: 10 días
  Total: 16 + 30 + 31 + 10 = 87 días
```

**Variantes:**
- `"Piensa paso a paso"` — lo más simple y efectivo
- `"Razona antes de responder"` — similar
- `"Primero analiza el problema, luego da tu respuesta"` — más estructurado

---

### 3.4 Asignación de rol (Role prompting)

Indicar al modelo que adopte un rol o persona específica para orientar su estilo y conocimiento.

```
Eres un senior developer con 15 años de experiencia en Python.
Revisa el siguiente código y señala los problemas de rendimiento:
[código]
```

```
Actúa como un profesor de secundaria explicando este concepto
a un alumno de 14 años que no tiene conocimientos previos:
[concepto]
```

**Importante:** El rol define el *tono y enfoque*, pero no le da al modelo conocimiento que no tiene.

---

### 3.5 Instrucciones negativas y positivas

Las instrucciones **positivas** (qué hacer) son más efectivas que las **negativas** (qué no hacer).

| Menos efectivo | Más efectivo |
|---|---|
| "No uses jerga técnica" | "Usa lenguaje sencillo apto para no técnicos" |
| "No seas demasiado largo" | "Responde en máximo 3 párrafos" |
| "No repitas la pregunta" | "Ve directo a la respuesta" |

---

## 4. Técnicas avanzadas

### 4.1 Self-consistency

Generar múltiples respuestas con temperatura > 0 y tomar la respuesta mayoritaria. Útil para problemas con respuesta única (matemáticas, lógica).

```python
respuestas = []
for _ in range(5):
    r = llm.generate(prompt, temperature=0.7)
    respuestas.append(r)

respuesta_final = max(set(respuestas), key=respuestas.count)
```

---

### 4.2 Least-to-Most prompting

Descomponer un problema complejo en subproblemas más simples y resolverlos en orden.

```
Problema: ¿Cuánto cuesta alquilar un coche en Madrid durante 2 semanas
          si el precio base es 45€/día y hay un descuento del 15% a partir
          de los 7 días?

Paso 1: ¿Cuántos días son 2 semanas? → 14 días
Paso 2: ¿Se aplica el descuento? → Sí, a partir del día 7
Paso 3: ¿A cuántos días se aplica el descuento? → A 14 - 7 = 7 días
Paso 4: Coste sin descuento: 14 × 45 = 630€
Paso 5: Descuento en los últimos 7 días: 7 × 45 × 0.15 = 47.25€
Paso 6: Total: 630 - 47.25 = 582.75€
```

---

### 4.3 ReAct (Reason + Act)

Combinar razonamiento (Thought) con acciones (Action) y observaciones (Observation). Es la base de los agentes de IA.

```
Thought: Necesito saber el CEO actual de Anthropic.
Action: buscar("CEO Anthropic 2024")
Observation: Dario Amodei es el CEO de Anthropic.
Thought: Ahora tengo la información necesaria.
Answer: El CEO de Anthropic es Dario Amodei.
```

---

### 4.4 Prompt con delimitadores

Usar delimitadores claros para separar secciones del prompt, especialmente cuando se incluyen datos de entrada del usuario (ayuda a prevenir prompt injection).

```
Analiza el siguiente contrato y extrae las fechas clave.

<contrato>
{{texto_del_contrato}}
</contrato>

Devuelve las fechas en formato JSON: {"fechas": [{"descripcion": "...", "fecha": "YYYY-MM-DD"}]}
```

Delimitadores comunes: `<etiquetas>`, `"""triple comillas"""`, `---separadores---`, `###secciones###`

---

### 4.5 Output estructurado

Pedir explícitamente la respuesta en un formato concreto para facilitar el procesamiento posterior.

```
Analiza este currículum y extrae la información en el siguiente formato JSON:
{
  "nombre": "",
  "email": "",
  "años_experiencia": 0,
  "tecnologías": [],
  "ultimo_puesto": ""
}

Currículum:
[texto del cv]
```

---

## 5. Errores comunes

### Instrucciones ambiguas

```
❌ "Escribe algo sobre machine learning"
✅ "Escribe una introducción de 200 palabras sobre machine learning
    dirigida a directivos no técnicos, enfocada en beneficios empresariales"
```

### Demasiadas instrucciones a la vez

```
❌ "Analiza este texto, resume los puntos clave, identifica el tono,
    tradúcelo al inglés, y valora si es adecuado para publicar en LinkedIn"

✅ Dividir en múltiples llamadas: primero analizar, luego resumir, etc.
```

### Asumir que el modelo recuerda

Cada llamada a la API es independiente (salvo el historial de conversación que se envíe explícitamente).

```
❌ Primera llamada: "Analiza este contrato: [contrato]"
   Segunda llamada: "¿Cuál es la fecha de vencimiento?" ← El modelo no recuerda el contrato

✅ Segunda llamada: "Del siguiente contrato, ¿cuál es la fecha de vencimiento? [contrato]"
```

### Prompts demasiado cortos para tareas complejas

Para tareas simples, menos es más. Para tareas complejas, más contexto e instrucciones = mejor resultado.

---

## 6. Buenas prácticas

| Práctica | Descripción |
|---|---|
| **Sé específico** | Indica el formato, extensión, tono y audiencia |
| **Una tarea a la vez** | Separa tareas complejas en múltiples llamadas |
| **Usa ejemplos** | Few-shot si el formato es importante |
| **Pide razonamiento** | "Piensa paso a paso" para problemas complejos |
| **Itera** | El primer prompt rara vez es el óptimo |
| **Versiona tus prompts** | Guarda las versiones como código |
| **Prueba con casos límite** | Inputs vacíos, ambiguos, en otro idioma... |
| **Usa delimitadores** | Para separar instrucciones de datos de usuario |
| **Valida outputs** | No asumas que la respuesta es correcta |
| **Temperature según tarea** | 0 para código/datos; 0.7 para texto creativo |

---

## 7. Diferencias entre modelos

Los LLMs tienen personalidades y capacidades distintas. Lo que funciona bien en uno puede no funcionar igual en otro.

| Modelo | Fortalezas | Notas de prompting |
|---|---|---|
| **Claude** | Razonamiento largo, seguimiento de instrucciones, seguridad | Responde bien a instrucciones detalladas y estructuradas |
| **GPT-4** | Multimodalidad, función calling | Muy versátil; admite system prompts elaborados |
| **Gemini** | Contexto muy largo, integración Google | Bueno con documentos largos |
| **Llama 3** | Open source, eficiente | Puede requerir prompts más explícitos que los modelos propietarios |
| **Mistral** | Velocidad y eficiencia | Menor ventana de contexto en versiones pequeñas |

**Recomendación:** Prueba siempre tus prompts en el modelo que vayas a usar en producción. No des por hecho que lo que funciona en uno funciona igual en otro.

---

## 8. Cheatsheet de referencia

```
TÉCNICA              CUÁNDO USARLA
─────────────────────────────────────────────────────
Zero-shot            Tareas simples y bien definidas
Few-shot             Cuando el formato importa mucho
Chain-of-Thought     Razonamiento, matemáticas, lógica
Role prompting       Necesitas tono/enfoque específico
Output estructurado  Necesitas parsear la respuesta
Delimitadores        Datos de usuario en el prompt
Self-consistency     Respuestas únicas de alta precisión
ReAct                Agentes con acceso a herramientas

PARÁMETROS           CUÁNDO AJUSTARLOS
─────────────────────────────────────────────────────
temperature=0        Código, datos, respuestas exactas
temperature=0.3-0.7  Redacción, análisis
temperature=0.8-1.2  Creatividad, brainstorming
max_tokens           Limitar longitud de respuesta
top_p                Alternativa a temperature
```

---

**Anterior:** [01 — ¿Qué es un LLM?](./01-que-es-un-llm.md) · **Siguiente:** [03 — Fine-tuning vs RAG](./03-finetuning-vs-rag.md)
