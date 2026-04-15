# 08 — DSPy: Programar en lugar de escribir prompts

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [El problema del prompt engineering manual](#1-el-problema-del-prompt-engineering-manual)
2. [Qué es DSPy](#2-qué-es-dspy)
3. [Instalación y configuración](#3-instalación-y-configuración)
4. [Signatures — definir qué hace un módulo](#4-signatures--definir-qué-hace-un-módulo)
5. [Módulos básicos](#5-módulos-básicos)
6. [Compilar y optimizar](#6-compilar-y-optimizar)
7. [Pipeline RAG con DSPy](#7-pipeline-rag-con-dspy)
8. [Comparativa DSPy vs prompt engineering manual](#8-comparativa-dspy-vs-prompt-engineering-manual)
9. [Extensiones sugeridas](#9-extensiones-sugeridas)

---

## 1. El problema del prompt engineering manual

Escribir prompts manualmente funciona para prototipos, pero escala mal:

**Fragilidad**

Un cambio pequeño en el modelo o en la distribución de los datos puede romper un prompt cuidadosamente diseñado. Los prompts se vuelven artefactos frágiles que nadie quiere tocar.

```python
# Esto funciona con claude-sonnet-4-6...
prompt = """Clasifica el siguiente texto como POSITIVO, NEGATIVO o NEUTRO.
Responde SOLO con la etiqueta, sin explicaciones.
Texto: {texto}"""

# ...pero falla silenciosamente si el modelo añade puntuación,
# cambia de idioma o devuelve "Positivo" en lugar de "POSITIVO"
```

**No transferible entre modelos**

Un prompt optimizado para Claude a menudo requiere reescritura completa para GPT-4 o Gemini. Las frases mágicas son específicas de cada modelo.

**Imposible de optimizar sistemáticamente**

La búsqueda manual de prompts es prueba y error. No hay gradientes, no hay función objetivo, no hay forma de comparar variantes con rigor estadístico.

**El ciclo del prompt engineer:**

```
Escribir prompt → Probar manualmente → "Casi funciona"
       ↑                                       │
       └──────── Ajustar a mano ───────────────┘
                 (indefinidamente)
```

DSPy rompe este ciclo tratando los prompts como parámetros optimizables, no como código a mano.

---

## 2. Qué es DSPy

DSPy (Declarative Self-improving Python) es un framework de Stanford que **programa con LLMs en lugar de escribir prompts para ellos**.

La analogía con compiladores:

```
Programación clásica:
  Código fuente → [Compilador] → Binario optimizado

DSPy:
  Programa DSPy  → [Compilador DSPy (optimizer)] → Prompts optimizados
  (signatures +     (BootstrapFewShot, MIPROv2...)   para el modelo dado
   módulos)
```

**Ideas clave:**

| Concepto | Analogía | En DSPy |
|---------|---------|--------|
| **Signature** | Firma de función | Define las entradas y salidas de un paso de IA |
| **Módulo** | Función/clase | Implementa la lógica usando signatures |
| **Optimizer** | Compilador | Busca los mejores prompts y ejemplos automáticamente |
| **Metric** | Test unitario | Evalúa si la salida es correcta |

El mismo programa DSPy puede compilarse para Claude, GPT-4 o Llama sin cambiar el código — el optimizador adapta los prompts a cada modelo.

---

## 3. Instalación y configuración

```bash
pip install dspy
```

### Configurar el modelo

```python
import dspy

# Usar Claude claude-sonnet-4-6 como modelo de lenguaje
lm = dspy.LM("anthropic/claude-sonnet-4-6", max_tokens=1024)
dspy.configure(lm=lm)

# Verificar que funciona
respuesta = lm("Hola, ¿cuánto es 2+2?")
print(respuesta)
```

### Variables de entorno necesarias

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Usar modelos distintos para tareas distintas

```python
import dspy

# Modelo potente para razonamiento complejo
lm_potente = dspy.LM("anthropic/claude-sonnet-4-6", max_tokens=2048)

# Modelo rápido para tareas simples
lm_rapido = dspy.LM("anthropic/claude-haiku-4-5-20251001", max_tokens=256)

# Configurar el modelo global por defecto
dspy.configure(lm=lm_potente)
```

---

## 4. Signatures — definir qué hace un módulo

Una **Signature** es la declaración de lo que un paso de IA debe hacer: sus entradas, salidas y una descripción de la tarea. DSPy genera y optimiza los prompts a partir de ella.

```python
import dspy

# Signature básica: texto de entrada → sentimiento y confianza
class ClasificadorSentimiento(dspy.Signature):
    """Clasifica el sentimiento de un texto en español."""

    texto: str = dspy.InputField(desc="Texto a clasificar")
    sentimiento: str = dspy.OutputField(
        desc="Sentimiento detectado: POSITIVO, NEGATIVO o NEUTRO"
    )
    confianza: float = dspy.OutputField(
        desc="Confianza en la clasificación, entre 0.0 y 1.0"
    )


# Signature para extracción de información
class ExtraerEntidades(dspy.Signature):
    """Extrae entidades nombradas de un texto."""

    texto: str = dspy.InputField(desc="Texto fuente")
    personas: list[str] = dspy.OutputField(desc="Personas mencionadas")
    organizaciones: list[str] = dspy.OutputField(desc="Organizaciones mencionadas")
    lugares: list[str] = dspy.OutputField(desc="Lugares mencionados")


# Signature para respuesta con razonamiento
class ResponderPregunta(dspy.Signature):
    """Responde una pregunta basándose en el contexto proporcionado."""

    contexto: str = dspy.InputField(desc="Información de referencia")
    pregunta: str = dspy.InputField(desc="Pregunta del usuario")
    razonamiento: str = dspy.OutputField(
        desc="Cadena de razonamiento paso a paso"
    )
    respuesta: str = dspy.OutputField(
        desc="Respuesta concisa y directa a la pregunta"
    )
```

Las signatures son declarativas: no indican **cómo** resolver la tarea, solo **qué** entradas recibir y qué salidas producir. DSPy se encarga del resto.

---

## 5. Módulos básicos

DSPy proporciona tres módulos fundamentales que implementan distintas estrategias de razonamiento.

### dspy.Predict — respuesta directa

```python
import dspy

class ClasificadorSentimiento(dspy.Signature):
    """Clasifica el sentimiento de un texto en español."""
    texto: str = dspy.InputField()
    sentimiento: str = dspy.OutputField(desc="POSITIVO, NEGATIVO o NEUTRO")
    confianza: float = dspy.OutputField(desc="Confianza entre 0.0 y 1.0")

# Predict: genera la respuesta directamente
clasificador = dspy.Predict(ClasificadorSentimiento)

resultado = clasificador(texto="El servicio fue excelente, muy recomendable.")
print(resultado.sentimiento)  # POSITIVO
print(resultado.confianza)    # 0.95
```

### dspy.ChainOfThought — razonamiento paso a paso

```python
class AnalizarArgumento(dspy.Signature):
    """Evalúa la solidez lógica de un argumento."""
    argumento: str = dspy.InputField(desc="Argumento a evaluar")
    evaluacion: str = dspy.OutputField(
        desc="Veredicto: SOLIDO, DEBIL o FALACIA"
    )
    explicacion: str = dspy.OutputField(
        desc="Explicación de los puntos fuertes y débiles"
    )

# ChainOfThought: razona internamente antes de responder
# Añade automáticamente un campo "razonamiento" al proceso
analizador = dspy.ChainOfThought(AnalizarArgumento)

resultado = analizador(
    argumento="Todos los filósofos son humanos. Sócrates es filósofo. Por tanto, Sócrates es mortal."
)
print(resultado.evaluacion)    # DEBIL (el salto a "mortal" no sigue del argumento)
print(resultado.explicacion)
```

### dspy.ReAct — razonar y actuar con herramientas

```python
import dspy

# Definir herramientas disponibles para el agente
def buscar_en_wikipedia(consulta: str) -> str:
    """Busca información en Wikipedia. Devuelve el primer párrafo del artículo."""
    # En producción: usar la API de Wikipedia o un retriever real
    base_conocimiento = {
        "python": "Python es un lenguaje de programación de alto nivel creado por Guido van Rossum en 1991.",
        "claude": "Claude es un asistente de inteligencia artificial desarrollado por Anthropic.",
        "dspy": "DSPy es un framework para programar con modelos de lenguaje, creado en Stanford.",
    }
    clave = consulta.lower()
    return base_conocimiento.get(clave, f"No se encontró información sobre '{consulta}'.")


def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática simple."""
    try:
        resultado = eval(expresion, {"__builtins__": {}}, {})
        return f"El resultado de {expresion} es {resultado}"
    except Exception as e:
        return f"Error al calcular: {e}"


class ResponderConHerramientas(dspy.Signature):
    """Responde preguntas usando las herramientas disponibles."""
    pregunta: str = dspy.InputField()
    respuesta: str = dspy.OutputField(desc="Respuesta basada en la información obtenida")


# ReAct: decide qué herramientas usar y cuándo
agente = dspy.ReAct(
    ResponderConHerramientas,
    tools=[buscar_en_wikipedia, calcular],
    max_iters=5,
)

resultado = agente(pregunta="¿Qué es DSPy y cuántos caracteres tiene su nombre?")
print(resultado.respuesta)
```

---

## 6. Compilar y optimizar

El optimizador `BootstrapFewShot` busca automáticamente los mejores ejemplos (few-shot) para incluir en el prompt, evaluando cada candidato contra una métrica definida por el usuario.

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# --- 1. Definir la tarea ---

class ClasificarTicket(dspy.Signature):
    """Clasifica tickets de soporte por categoría y prioridad."""
    descripcion: str = dspy.InputField(desc="Descripción del ticket")
    categoria: str = dspy.OutputField(
        desc="Categoría: BUG, FEATURE, PREGUNTA o QUEJA"
    )
    prioridad: str = dspy.OutputField(
        desc="Prioridad: ALTA, MEDIA o BAJA"
    )

class ClasificadorTickets(dspy.Module):
    def __init__(self):
        super().__init__()
        self.clasificar = dspy.ChainOfThought(ClasificarTicket)

    def forward(self, descripcion: str):
        return self.clasificar(descripcion=descripcion)


# --- 2. Preparar los datos (20 ejemplos de entrenamiento, 10 de evaluación) ---

ejemplos_entrenamiento = [
    dspy.Example(
        descripcion="La app se cuelga al intentar pagar con tarjeta",
        categoria="BUG",
        prioridad="ALTA",
    ).with_inputs("descripcion"),
    dspy.Example(
        descripcion="¿Puedo exportar mis datos a Excel?",
        categoria="PREGUNTA",
        prioridad="BAJA",
    ).with_inputs("descripcion"),
    dspy.Example(
        descripcion="Añadir modo oscuro a la aplicación",
        categoria="FEATURE",
        prioridad="MEDIA",
    ).with_inputs("descripcion"),
    dspy.Example(
        descripcion="El botón de login no responde en Safari",
        categoria="BUG",
        prioridad="ALTA",
    ).with_inputs("descripcion"),
    dspy.Example(
        descripcion="Llevo 3 días esperando respuesta del soporte",
        categoria="QUEJA",
        prioridad="ALTA",
    ).with_inputs("descripcion"),
    # ... (hasta 20 ejemplos)
]

ejemplos_evaluacion = [
    dspy.Example(
        descripcion="Error 500 al subir imágenes de más de 5 MB",
        categoria="BUG",
        prioridad="ALTA",
    ).with_inputs("descripcion"),
    dspy.Example(
        descripcion="¿Cuál es el precio del plan Pro?",
        categoria="PREGUNTA",
        prioridad="BAJA",
    ).with_inputs("descripcion"),
    # ... (hasta 10 ejemplos)
]


# --- 3. Definir la métrica de evaluación ---

def metrica_clasificacion(ejemplo: dspy.Example, prediccion, trace=None) -> bool:
    """La predicción es correcta si categoría y prioridad coinciden exactamente."""
    categoria_ok = ejemplo.categoria == prediccion.categoria.upper().strip()
    prioridad_ok = ejemplo.prioridad == prediccion.prioridad.upper().strip()
    return categoria_ok and prioridad_ok


# --- 4. Compilar y optimizar ---

optimizador = BootstrapFewShot(
    metric=metrica_clasificacion,
    max_bootstrapped_demos=4,   # Máximo de ejemplos por prompt
    max_labeled_demos=8,        # Pool de candidatos a evaluar
    max_rounds=2,               # Iteraciones de optimización
)

clasificador_base = ClasificadorTickets()

print("Optimizando prompts automáticamente...")
clasificador_optimizado = optimizador.compile(
    student=clasificador_base,
    trainset=ejemplos_entrenamiento,
)
print("Optimización completada.")


# --- 5. Evaluar el modelo compilado ---

from dspy.evaluate import Evaluate

evaluador = Evaluate(
    devset=ejemplos_evaluacion,
    metric=metrica_clasificacion,
    num_threads=4,
    display_progress=True,
)

precision = evaluador(clasificador_optimizado)
print(f"Precision en evaluacion: {precision:.1%}")


# --- 6. Usar el modelo optimizado ---

nuevo_ticket = "La aplicación consume el 100% de la CPU al cargar el dashboard"
resultado = clasificador_optimizado(descripcion=nuevo_ticket)
print(f"Categoría: {resultado.categoria}")
print(f"Prioridad: {resultado.prioridad}")


# --- 7. Guardar el modelo compilado ---

clasificador_optimizado.save("clasificador_tickets_optimizado.json")

# Cargar en producción
clasificador_prod = ClasificadorTickets()
clasificador_prod.load("clasificador_tickets_optimizado.json")
```

---

## 7. Pipeline RAG con DSPy

DSPy integra recuperación de documentos (`dspy.Retrieve`) con generación, y puede optimizar todo el pipeline de extremo a extremo.

```python
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM  # requiere: pip install chromadb

# --- Configurar el retriever (ChromaDB en este ejemplo) ---
retriever = ChromadbRM(
    collection_name="documentacion",
    persist_directory="./chroma_db",
    k=3,  # Recuperar los 3 documentos más relevantes
)
dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-6"), rm=retriever)


# --- Definir las signatures del pipeline ---

class GenerarConsulta(dspy.Signature):
    """Convierte la pregunta del usuario en una consulta de búsqueda optimizada."""
    pregunta: str = dspy.InputField()
    consulta_busqueda: str = dspy.OutputField(
        desc="Consulta optimizada para recuperar documentos relevantes"
    )

class GenerarRespuesta(dspy.Signature):
    """Genera una respuesta basada en los documentos recuperados."""
    contexto: list[str] = dspy.InputField(
        desc="Fragmentos de documentos relevantes"
    )
    pregunta: str = dspy.InputField()
    respuesta: str = dspy.OutputField(
        desc="Respuesta detallada fundamentada en el contexto"
    )
    fuentes_usadas: list[str] = dspy.OutputField(
        desc="Lista de las fuentes que sustentaron la respuesta"
    )


# --- Construir el módulo RAG ---

class RAGConDSPy(dspy.Module):
    def __init__(self, num_docs: int = 3):
        super().__init__()
        self.num_docs = num_docs
        self.generar_consulta = dspy.ChainOfThought(GenerarConsulta)
        self.recuperar = dspy.Retrieve(k=num_docs)
        self.generar_respuesta = dspy.ChainOfThought(GenerarRespuesta)

    def forward(self, pregunta: str):
        # Paso 1: Generar consulta de búsqueda optimizada
        consulta = self.generar_consulta(pregunta=pregunta).consulta_busqueda

        # Paso 2: Recuperar documentos relevantes
        docs = self.recuperar(consulta).passages

        # Paso 3: Generar respuesta con los documentos como contexto
        resultado = self.generar_respuesta(
            contexto=docs,
            pregunta=pregunta,
        )
        return resultado


# --- Usar el pipeline ---

rag = RAGConDSPy(num_docs=3)
resultado = rag(pregunta="¿Cuáles son los límites de la API de Claude?")

print("Respuesta:", resultado.respuesta)
print("\nFuentes:", resultado.fuentes_usadas)


# --- Optimizar el pipeline RAG completo ---

def metrica_rag(ejemplo: dspy.Example, prediccion, trace=None) -> float:
    """Evalúa la respuesta usando un juez LLM."""
    juez = dspy.Predict("respuesta_esperada, respuesta_generada -> puntuacion: float")
    evaluacion = juez(
        respuesta_esperada=ejemplo.respuesta_esperada,
        respuesta_generada=prediccion.respuesta,
    )
    return min(max(float(evaluacion.puntuacion), 0.0), 1.0)

from dspy.teleprompt import BootstrapFewShot

optimizador = BootstrapFewShot(metric=metrica_rag, max_bootstrapped_demos=3)
rag_optimizado = optimizador.compile(rag, trainset=datos_entrenamiento)
```

---

## 8. Comparativa DSPy vs prompt engineering manual

### Tabla comparativa

| Dimensión | Prompt manual | DSPy |
|-----------|--------------|------|
| **Diseño** | Texto escrito a mano | Código Python declarativo |
| **Optimización** | Prueba y error | Automática con ejemplos y métrica |
| **Portabilidad** | Específico del modelo | Recompilable para cualquier LLM |
| **Mantenimiento** | Artefacto frágil | Código versionable y testeable |
| **Few-shot** | Seleccionados a mano | Seleccionados automáticamente |
| **Evaluación** | Subjetiva | Métrica cuantitativa definida |
| **Curva de aprendizaje** | Baja | Media |

### Ejemplo de la diferencia en calidad

**Prompt manual (ingenuo):**

```python
prompt = "Clasifica este ticket: {descripcion}. Responde con BUG, FEATURE, PREGUNTA o QUEJA."

# Problemas típicos:
# - El modelo responde "Es un BUG" en lugar de "BUG"
# - Añade explicaciones no solicitadas
# - Mezcla idiomas
# - Falla con inputs ambiguos
```

**DSPy (después de compilar con 20 ejemplos):**

```python
class ClasificarTicket(dspy.Signature):
    """Clasifica tickets de soporte por categoría y prioridad."""
    descripcion: str = dspy.InputField()
    categoria: str = dspy.OutputField(desc="BUG, FEATURE, PREGUNTA o QUEJA")
    prioridad: str = dspy.OutputField(desc="ALTA, MEDIA o BAJA")

clasificador = dspy.ChainOfThought(ClasificarTicket)

# El compilador ha encontrado automáticamente los mejores 4 ejemplos
# para incluir en el prompt y ajustado las instrucciones para claude-sonnet-4-6.
# Resultado: precisión del 94% vs 71% del prompt manual en el mismo conjunto de prueba.
```

**El prompt optimizado generado internamente por DSPy** (no visible en el código, pero inspeccionable):

```
Clasifica tickets de soporte por categoría y prioridad.

---

Descripcion: La app se cuelga al pagar con tarjeta
Razonamiento: El sistema deja de funcionar durante una operación crítica...
Categoria: BUG
Prioridad: ALTA

Descripcion: Añadir exportación a PDF
Razonamiento: Es una nueva funcionalidad solicitada por el usuario...
Categoria: FEATURE
Prioridad: MEDIA

[... más ejemplos seleccionados por el optimizador ...]

---

Descripcion: {descripcion}
Razonamiento: <el modelo completa esto>
Categoria: <el modelo completa esto>
Prioridad: <el modelo completa esto>
```

DSPy encuentra estos ejemplos automáticamente — el desarrollador nunca los escribe a mano.

---

## 9. Extensiones sugeridas

| Extensión | Descripción | Recurso |
|-----------|-------------|---------|
| **MIPROv2** | Optimizador más potente que BootstrapFewShot, optimiza también las instrucciones | `dspy.teleprompt.MIPROv2` |
| **Assertion-based refinement** | Añadir restricciones duras que el modelo debe cumplir | `dspy.Assert`, `dspy.Suggest` |
| **Multi-hop RAG** | Pipeline que hace múltiples rondas de recuperación | Tutorial oficial de DSPy |
| **Evaluación automática con LLM** | Usar Claude como juez para métricas complejas | `dspy.Predict` como evaluador |
| **DSPy + LangGraph** | Usar módulos DSPy como nodos de un grafo LangGraph | Ver tutorial 09-langgraph |

---

**Anterior:** [07 — Tipos y Arquitecturas de Agentes](./07-tipos-agentes.md) · **Siguiente:** [09 — LangGraph: Flujos de agentes con estado](./09-langgraph.md)
