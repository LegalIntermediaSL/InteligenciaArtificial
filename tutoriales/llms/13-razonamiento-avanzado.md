# 13 — Razonamiento Avanzado: Chain-of-Thought, Self-Consistency y Modelos de Razonamiento

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 55 min

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/llms/13-razonamiento-avanzado.ipynb)

---

## Índice

1. [Por qué el razonamiento explícito mejora la precisión](#1-por-qué-el-razonamiento-explícito-mejora-la-precisión)
2. [Chain-of-Thought (CoT)](#2-chain-of-thought-cot)
3. [Self-Consistency: votar la mejor respuesta](#3-self-consistency-votar-la-mejor-respuesta)
4. [Tree of Thoughts (ToT)](#4-tree-of-thoughts-tot)
5. [Extended Thinking de Claude](#5-extended-thinking-de-claude)
6. [Comparativa de técnicas](#6-comparativa-de-técnicas)
7. [Casos de uso](#7-casos-de-uso)
8. [Patrones anti-razonamiento a evitar](#8-patrones-anti-razonamiento-a-evitar)
9. [Extensiones sugeridas](#9-extensiones-sugeridas)

---

## 1. Por qué el razonamiento explícito mejora la precisión

Los LLMs son redes que predicen el siguiente token. Cuando se les pide una respuesta directa a un problema complejo, el modelo genera la solución en un solo paso sin posibilidad de corrección intermedia. El resultado es a menudo incorrecto en tareas que requieren múltiples inferencias encadenadas.

El razonamiento explícito resuelve esto obligando al modelo a **externalizar su proceso mental** antes de dar la respuesta final. Al escribir los pasos intermedios, el modelo puede detectar inconsistencias y corregirlas antes de comprometerse con una conclusión.

**Evidencia empírica:**

- En el benchmark GSM8K (matemáticas de primaria), GPT-3 sin CoT obtiene ~17% de acierto. Con CoT few-shot llega al ~56%. Con Self-Consistency al ~74%.
- En benchmarks de lógica formal (BBH), CoT mejora la precisión entre 20 y 40 puntos porcentuales respecto a la respuesta directa.

**Principio clave:** los tokens que el modelo genera son parte de su cómputo. Más tokens de razonamiento = más cómputo = mejores respuestas en tareas complejas.

---

## 2. Chain-of-Thought (CoT)

### 2.1 Zero-shot CoT

La forma más simple: añadir "Piensa paso a paso" (o "Let's think step by step") al final del prompt. Descubierto por Kojima et al. (2022), funciona sorprendentemente bien sin necesidad de ejemplos.

```python
import anthropic

client = anthropic.Anthropic()

def cot_zero_shot(problema: str) -> str:
    """Chain-of-Thought zero-shot: solo añadimos la instrucción de pensar."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"{problema}\n\nPiensa paso a paso antes de dar tu respuesta final."
            }
        ]
    )
    return response.content[0].text

# Ejemplo: problema aritmético con varios pasos
problema = """
Un tren sale de Madrid a las 08:00 a 120 km/h hacia Barcelona (620 km).
Otro tren sale de Barcelona a las 09:30 a 90 km/h hacia Madrid.
¿A qué hora se cruzan? ¿A qué distancia de Madrid?
"""

respuesta = cot_zero_shot(problema)
print(respuesta)
```

### 2.2 Few-shot CoT

Proporcionamos ejemplos completos de razonamiento (problema → cadena de pasos → respuesta) para que el modelo imite el formato y nivel de detalle deseado.

```python
FEW_SHOT_COT_SYSTEM = """Eres un asistente matemático experto. Resuelves cada problema
mostrando explícitamente cada paso de razonamiento con el formato:

Paso 1: [descripción]
Cálculo: [operación]
Resultado parcial: [valor]

...

Respuesta final: [valor con unidades]"""

EJEMPLOS_COT = [
    {
        "role": "user",
        "content": "Si Juan tiene 3 veces más canicas que Pedro, y Pedro tiene 12, ¿cuántas tienen entre los dos?"
    },
    {
        "role": "assistant",
        "content": """Paso 1: Determinar las canicas de Juan
Cálculo: Juan = 3 × Pedro = 3 × 12
Resultado parcial: Juan tiene 36 canicas

Paso 2: Sumar las canicas de ambos
Cálculo: Total = Juan + Pedro = 36 + 12
Resultado parcial: 48 canicas

Respuesta final: Entre los dos tienen 48 canicas."""
    }
]

def cot_few_shot(problema: str) -> str:
    """CoT con ejemplos: el modelo imita el formato de razonamiento mostrado."""
    messages = EJEMPLOS_COT + [{"role": "user", "content": problema}]
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=FEW_SHOT_COT_SYSTEM,
        messages=messages
    )
    return response.content[0].text

nuevo_problema = "Una tienda tiene 240 artículos. Vende el 35% el lunes y el 25% del resto el martes. ¿Cuántos quedan?"
print(cot_few_shot(nuevo_problema))
```

### 2.3 CoT estructurado con extracción de respuesta

En producción necesitamos extraer la respuesta final de la cadena de razonamiento de forma fiable.

```python
import re

def cot_con_extraccion(problema: str) -> dict:
    """CoT que devuelve tanto el razonamiento como la respuesta extraída."""
    prompt = f"""{problema}

Razona paso a paso. Al final escribe exactamente:
RESPUESTA FINAL: [tu respuesta]"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    texto = response.content[0].text
    
    # Extraer respuesta final
    match = re.search(r"RESPUESTA FINAL:\s*(.+?)(?:\n|$)", texto, re.IGNORECASE)
    respuesta_final = match.group(1).strip() if match else "No extraída"
    
    return {
        "razonamiento": texto,
        "respuesta": respuesta_final,
        "tokens_usados": response.usage.input_tokens + response.usage.output_tokens
    }
```

---

## 3. Self-Consistency: votar la mejor respuesta

Wang et al. (2022) demostraron que si se generan N respuestas independientes con temperatura > 0 y se elige la más frecuente, la precisión mejora significativamente respecto a una sola respuesta con CoT.

**Intuición:** el modelo puede llegar a la respuesta correcta por varios caminos. La respuesta correcta aparece con más frecuencia que cualquier respuesta incorrecta individual.

```python
from collections import Counter
from typing import Optional
import re

class SelfConsistencyReasoner:
    """
    Implementa Self-Consistency: genera N respuestas y vota la más frecuente.
    
    Args:
        model: Modelo a usar
        n_muestras: Número de respuestas independientes a generar
        temperatura: Temperatura para la diversidad (recomendado: 0.7-1.0)
    """
    
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        n_muestras: int = 5,
        temperatura: float = 0.8
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.n_muestras = n_muestras
        self.temperatura = temperatura
    
    def _generar_respuesta(self, problema: str) -> str:
        """Genera una respuesta individual con CoT."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=self.temperatura,
            messages=[{
                "role": "user",
                "content": (
                    f"{problema}\n\n"
                    "Razona paso a paso. Al final escribe:\n"
                    "RESPUESTA: [solo el valor numérico o la opción]"
                )
            }]
        )
        return response.content[0].text
    
    def _extraer_respuesta(self, texto: str) -> Optional[str]:
        """Extrae la respuesta final normalizada."""
        match = re.search(r"RESPUESTA:\s*(.+?)(?:\n|$)", texto, re.IGNORECASE)
        if match:
            # Normalizar: minúsculas, sin espacios extra
            return match.group(1).strip().lower()
        return None
    
    def razonar(self, problema: str, verbose: bool = False) -> dict:
        """
        Resuelve el problema con Self-Consistency.
        
        Returns:
            dict con respuesta ganadora, votos, confianza y trazas opcionales
        """
        respuestas_raw = []
        respuestas_extraidas = []
        
        print(f"Generando {self.n_muestras} razonamientos independientes...")
        
        for i in range(self.n_muestras):
            raw = self._generar_respuesta(problema)
            respuestas_raw.append(raw)
            
            extraida = self._extraer_respuesta(raw)
            if extraida:
                respuestas_extraidas.append(extraida)
            
            if verbose:
                print(f"\n--- Muestra {i+1} ---")
                print(raw[:300] + "..." if len(raw) > 300 else raw)
        
        if not respuestas_extraidas:
            return {"error": "No se pudo extraer ninguna respuesta estructurada"}
        
        # Votar
        conteo = Counter(respuestas_extraidas)
        ganadora, votos = conteo.most_common(1)[0]
        confianza = votos / len(respuestas_extraidas)
        
        return {
            "respuesta": ganadora,
            "votos": votos,
            "total_muestras": len(respuestas_extraidas),
            "confianza": f"{confianza:.0%}",
            "distribucion": dict(conteo),
            "trazas": respuestas_raw if verbose else None
        }


# Uso
reasoner = SelfConsistencyReasoner(n_muestras=5, temperatura=0.8)

problema_logico = """
En una carrera, Ana terminó antes que Bea. Carlos terminó después que Bea
pero antes que Diana. Elena terminó antes que Ana. ¿En qué posición terminó Carlos?
"""

resultado = reasoner.razonar(problema_logico)
print(f"\nRespuesta ganadora: {resultado['respuesta']}")
print(f"Votos: {resultado['votos']}/{resultado['total_muestras']}")
print(f"Confianza: {resultado['confianza']}")
print(f"Distribución: {resultado['distribucion']}")
```

---

## 4. Tree of Thoughts (ToT)

Yao et al. (2023) proponen explorar el espacio de soluciones como un árbol donde cada nodo es un "pensamiento" intermedio. A diferencia de CoT (una sola cadena lineal), ToT permite **retroceder** y explorar alternativas cuando una rama no parece prometedora.

**Componentes:**
1. **Generación de pensamientos:** expandir el nodo actual en N posibles continuaciones
2. **Evaluación:** puntuar cada pensamiento (¿lleva hacia la solución?)
3. **Búsqueda:** BFS o DFS para explorar el árbol

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Nodo:
    """Representa un nodo en el árbol de pensamientos."""
    contenido: str
    puntuacion: float = 0.0
    hijos: List["Nodo"] = field(default_factory=list)
    profundidad: int = 0


class TreeOfThoughts:
    """
    Implementación simplificada de Tree of Thoughts con BFS.
    
    Usa el LLM tanto para generar pensamientos como para evaluarlos.
    Adecuado para problemas donde hay múltiples estrategias posibles.
    """
    
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        n_ramas: int = 3,
        profundidad_max: int = 3
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.n_ramas = n_ramas
        self.profundidad_max = profundidad_max
    
    def _expandir(self, problema: str, camino: str) -> List[str]:
        """Genera N posibles continuaciones desde el estado actual."""
        prompt = f"""Problema: {problema}

Razonamiento hasta ahora:
{camino if camino else "(inicio)"}

Genera exactamente {self.n_ramas} posibles pasos siguientes distintos para avanzar
hacia la solución. Numera cada uno con "Opción X:".
Sé conciso (1-2 frases por opción)."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        texto = response.content[0].text
        opciones = re.findall(r"Opción \d+:\s*(.+?)(?=Opción \d+:|$)", texto, re.DOTALL)
        return [op.strip() for op in opciones[:self.n_ramas]]
    
    def _evaluar(self, problema: str, camino: str) -> float:
        """Puntúa de 0 a 10 qué tan prometedor es este camino."""
        prompt = f"""Problema: {problema}

Camino de razonamiento:
{camino}

Puntúa de 0 a 10 qué tan prometedor es este camino para resolver el problema.
Responde SOLO con el número (ej: 7.5). Sin explicación."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            return float(response.content[0].text.strip())
        except ValueError:
            return 5.0
    
    def resolver(self, problema: str) -> dict:
        """Explora el árbol con BFS y devuelve el mejor camino encontrado."""
        raiz = Nodo(contenido="", profundidad=0)
        frontera = [raiz]
        mejor_nodo = raiz
        mejor_puntuacion = 0.0
        
        for nivel in range(self.profundidad_max):
            siguiente_frontera = []
            
            for nodo in frontera:
                camino_actual = nodo.contenido
                expansiones = self._expandir(problema, camino_actual)
                
                for expansion in expansiones:
                    nuevo_camino = f"{camino_actual}\n{expansion}" if camino_actual else expansion
                    puntuacion = self._evaluar(problema, nuevo_camino)
                    
                    hijo = Nodo(
                        contenido=nuevo_camino,
                        puntuacion=puntuacion,
                        profundidad=nivel + 1
                    )
                    nodo.hijos.append(hijo)
                    siguiente_frontera.append(hijo)
                    
                    if puntuacion > mejor_puntuacion:
                        mejor_puntuacion = puntuacion
                        mejor_nodo = hijo
            
            # Mantener solo los mejores nodos para el siguiente nivel
            siguiente_frontera.sort(key=lambda n: n.puntuacion, reverse=True)
            frontera = siguiente_frontera[:self.n_ramas]
            
            print(f"Nivel {nivel+1}: mejor puntuación = {mejor_puntuacion:.1f}")
        
        return {
            "mejor_camino": mejor_nodo.contenido,
            "puntuacion": mejor_puntuacion,
            "profundidad": mejor_nodo.profundidad
        }


# Ejemplo: problema de planificación
tot = TreeOfThoughts(n_ramas=3, profundidad_max=2)
problema = "Necesito organizar una conferencia para 200 personas en 3 semanas con un presupuesto de 5.000€. ¿Cuál es el mejor plan de acción?"
resultado = tot.resolver(problema)
print("\nMejor camino de razonamiento:")
print(resultado["mejor_camino"])
```

---

## 5. Extended Thinking de Claude

Claude 3.7 Sonnet introdujo **Extended Thinking**: el modelo genera tokens de razonamiento interno ("thinking blocks") antes de producir la respuesta final. A diferencia de CoT convencional, estos tokens son procesados con mayor profundidad y el modelo puede autocorregirse durante el proceso.

### 5.1 Activación básica

```python
def extended_thinking_basico(problema: str, budget_tokens: int = 5000) -> dict:
    """
    Extended Thinking de Claude.
    
    Args:
        problema: Pregunta o tarea a resolver
        budget_tokens: Máximo de tokens para el bloque de thinking (min: 1024)
    
    Returns:
        dict con thinking, respuesta y uso de tokens
    """
    # Nota: Extended Thinking requiere claude-sonnet-4-5 o superior
    response = client.messages.create(
        model="claude-sonnet-4-5-20251001",  # Requiere Sonnet o superior
        max_tokens=budget_tokens + 2048,
        thinking={
            "type": "enabled",
            "budget_tokens": budget_tokens
        },
        messages=[{"role": "user", "content": problema}]
    )
    
    thinking_text = ""
    respuesta_text = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            respuesta_text = block.text
    
    return {
        "thinking": thinking_text,
        "respuesta": respuesta_text,
        "tokens_thinking": len(thinking_text.split()),  # Aproximación
        "tokens_totales": response.usage.input_tokens + response.usage.output_tokens
    }
```

### 5.2 Interleaved Thinking (beta)

Para tareas que requieren razonamiento alternado con llamadas a herramientas:

```python
def extended_thinking_interleaved(problema: str, budget_tokens: int = 8000) -> dict:
    """
    Interleaved Thinking: razonamiento entre llamadas a herramientas.
    Requiere la beta 'interleaved-thinking-2025-05-14'.
    """
    response = client.messages.create(
        model="claude-sonnet-4-5-20251001",
        max_tokens=budget_tokens + 4096,
        betas=["interleaved-thinking-2025-05-14"],
        thinking={
            "type": "enabled",
            "budget_tokens": budget_tokens
        },
        messages=[{"role": "user", "content": problema}]
    )
    
    bloques = []
    for block in response.content:
        bloques.append({
            "tipo": block.type,
            "contenido": getattr(block, "thinking", None) or getattr(block, "text", "")
        })
    
    return {
        "bloques": bloques,
        "n_bloques_thinking": sum(1 for b in bloques if b["tipo"] == "thinking"),
        "uso": response.usage.model_dump()
    }
```

### 5.3 Cuándo usar Extended Thinking

Extended Thinking no siempre merece la pena. Usa esta guía:

```python
def recomendar_tecnica(tarea: dict) -> str:
    """
    Recomienda la técnica de razonamiento según las características de la tarea.
    
    tarea: dict con claves 'complejidad' (1-5), 'latencia_max_s', 'coste_importa'
    """
    complejidad = tarea.get("complejidad", 3)
    latencia_max = tarea.get("latencia_max_s", 10)
    coste_importa = tarea.get("coste_importa", True)
    
    if complejidad <= 2:
        return "Respuesta directa — el problema es simple, no necesitas razonamiento explícito"
    
    if latencia_max < 3:
        return "Zero-shot CoT — el más rápido con mejora de razonamiento"
    
    if complejidad >= 4 and not coste_importa:
        return "Extended Thinking — máxima precisión para problemas muy difíciles"
    
    if complejidad >= 4 and coste_importa:
        return "Self-Consistency (N=3) — buen equilibrio precisión/coste"
    
    return "Few-shot CoT — suficiente para tareas de complejidad media"
```

---

## 6. Comparativa de técnicas

| Técnica | Latencia | Coste relativo | Precisión en tareas difíciles | Implementación |
|---------|----------|---------------|-------------------------------|----------------|
| Respuesta directa | Muy baja | 1× | Baja | Trivial |
| Zero-shot CoT | Baja | 1.5× | Media | Una línea |
| Few-shot CoT | Baja-Media | 2× | Media-Alta | Ejemplos en prompt |
| Self-Consistency (N=5) | Alta | 5× | Alta | Clase dedicada |
| Tree of Thoughts | Muy alta | 10×+ | Alta (problemas estructurados) | Compleja |
| Extended Thinking | Media-Alta | 3-8× | Muy alta | Parámetro API |

**Guía de selección rápida:**

- **Chatbot en tiempo real** → Zero-shot CoT
- **Cálculos matemáticos en batch** → Self-Consistency (N=3 a 5)
- **Planificación o resolución de puzzles** → Tree of Thoughts o Extended Thinking
- **Análisis legal / médico / científico** → Extended Thinking con alto budget
- **Respuestas de trivia o QA simple** → Respuesta directa

---

## 7. Casos de uso

### 7.1 Matemáticas y aritmética

CoT y Self-Consistency son especialmente eficaces en problemas matemáticos porque cada paso es verificable y los errores son identificables.

```python
def resolver_matematica(problema: str, tecnica: str = "self-consistency") -> str:
    """Selector de técnica para problemas matemáticos."""
    
    if tecnica == "cot":
        return cot_con_extraccion(problema)["respuesta"]
    
    elif tecnica == "self-consistency":
        reasoner = SelfConsistencyReasoner(n_muestras=5)
        resultado = reasoner.razonar(problema)
        return resultado["respuesta"]
    
    elif tecnica == "extended-thinking":
        resultado = extended_thinking_basico(problema, budget_tokens=4000)
        return resultado["respuesta"]
    
    raise ValueError(f"Técnica no reconocida: {tecnica}")
```

### 7.2 Lógica y deducción

```python
PUZZLE_LOGICO = """
Cinco casas de colores distintos están en fila. En cada casa vive una persona
de nacionalidad diferente. Cada persona bebe una bebida distinta, fuma una marca
distinta y tiene un animal distinto.

- El inglés vive en la casa roja.
- El sueco tiene un perro.
- El danés bebe té.
- La casa verde está inmediatamente a la izquierda de la blanca.
- El dueño de la casa verde bebe café.
- La persona que fuma Pall Mall tiene pájaros.
- El dueño de la casa amarilla fuma Dunhill.
- El hombre que vive en la casa del centro bebe leche.
- El noruego vive en la primera casa.
- El hombre que fuma Blends vive junto al que tiene gatos.
- El hombre que tiene caballos vive junto al que fuma Dunhill.
- El que fuma BlueMaster bebe cerveza.
- El alemán fuma Prince.
- El noruego vive junto a la casa azul.
- El hombre que fuma Blends tiene un vecino que bebe agua.

¿Quién tiene el pez?
"""

# Extended Thinking es ideal para puzzles lógicos complejos
# resultado = extended_thinking_basico(PUZZLE_LOGICO, budget_tokens=10000)
# print(resultado["respuesta"])
```

### 7.3 Planificación y código

Para generación de código complejo, CoT ayuda al modelo a descomponer el problema antes de escribir el código:

```python
def generar_codigo_con_cot(especificacion: str) -> str:
    """Genera código precedido de un plan de implementación."""
    prompt = f"""Especificación: {especificacion}

Primero, describe el plan de implementación paso a paso:
1. Qué clases/funciones necesitas
2. Cómo se relacionan entre sí
3. Casos borde a manejar

Luego escribe el código completo en Python."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

---

## 8. Patrones anti-razonamiento a evitar

Estos patrones reducen la calidad del razonamiento aunque parezcan razonables:

### ❌ Forzar respuesta corta antes del razonamiento

```python
# MAL: pedir la respuesta primero anula el beneficio del CoT
prompt_malo = "Responde en una sola palabra y luego explica por qué."

# BIEN: primero el razonamiento, luego la respuesta
prompt_bueno = "Razona paso a paso. Al final, responde con una sola palabra: RESPUESTA FINAL: [palabra]"
```

### ❌ Temperatura 0 con Self-Consistency

```python
# MAL: temperatura 0 da siempre la misma respuesta → votar es inútil
reasoner_malo = SelfConsistencyReasoner(temperatura=0.0, n_muestras=5)

# BIEN: temperatura entre 0.6 y 1.0 para diversidad real
reasoner_bueno = SelfConsistencyReasoner(temperatura=0.8, n_muestras=5)
```

### ❌ Extended Thinking para tareas simples

```python
# MAL: usar Extended Thinking para clasificación binaria simple
# Coste y latencia innecesarios, sin ganancia de precisión

# BIEN: Extended Thinking solo para problemas donde el razonamiento
# intermedio es el cuello de botella (matemáticas avanzadas, lógica compleja)
```

### ❌ CoT con instrucciones contradictorias

```python
# MAL: "sé breve y piensa paso a paso" son objetivos contradictorios
prompt_contradictorio = "En no más de 50 palabras, piensa paso a paso y resuelve..."

# BIEN: dar espacio al razonamiento, restringir solo la respuesta final
prompt_claro = "Razona paso a paso sin límite de extensión. La RESPUESTA FINAL debe ser ≤20 palabras."
```

### ❌ Evaluar CoT por la cadena, no por la respuesta

El razonamiento puede ser incorrecto pero la respuesta correcta (o viceversa). Siempre evalúa la respuesta final, no la coherencia aparente de la cadena.

---

## 9. Extensiones sugeridas

- **Program-of-Thought (PoT):** en lugar de razonar en lenguaje natural, generar código Python que se ejecuta para obtener la respuesta — ideal para cálculos exactos
- **Reflexion:** agentes que evalúan sus propias respuestas y se auto-corrigen en bucle
- **Chain-of-Verification (CoVe):** generar preguntas de verificación para comprobar cada paso del razonamiento
- **Metacognitive prompting:** pedir al modelo que estime su propia confianza y señale los pasos donde puede equivocarse
- **Ensemble heterogéneo:** combinar CoT con diferentes modelos y hacer voting entre ellos

---

*Siguiente artículo: [14 — Multimodalidad Avanzada: Visión, Audio y Video](../multimodalidad/)*
