# 04 — Local vs cloud: cuándo usar cada opción

> **Bloque:** IA local · **Nivel:** Avanzado · **Tiempo estimado:** 40 min

---

## Índice

1. [Dimensiones de comparación](#1-dimensiones-de-comparación)
2. [Análisis de costes](#2-análisis-de-costes)
3. [Privacidad y compliance](#3-privacidad-y-compliance)
4. [Calidad de respuesta](#4-calidad-de-respuesta)
5. [Arquitectura híbrida](#5-arquitectura-híbrida)
6. [Tabla de decisión](#6-tabla-de-decisión)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Dimensiones de comparación

La elección entre ejecutar un modelo localmente o usar una API cloud no tiene una respuesta única. Depende del contexto, los requisitos del proyecto y las restricciones de cada organización.

### Comparativa por dimensiones

| Dimensión | Local (Ollama / Transformers) | Cloud (OpenAI, Anthropic, Google) |
|-----------|-------------------------------|-----------------------------------|
| **Coste variable** | Casi cero tras la inversión inicial | Pago por token; escala con el uso |
| **Coste fijo** | Hardware (GPU, RAM, servidor) | Ninguno o muy bajo |
| **Privacidad** | Datos nunca salen de tu infraestructura | Datos enviados a servidores de terceros |
| **Latencia** | Depende de tu hardware (50-500 ms) | Red + cola del proveedor (100-2000 ms) |
| **Calidad del modelo** | Modelos open-source (muy buenos en 7B+) | Modelos propietarios de frontera (GPT-4o, Claude 3.5) |
| **Actualización del modelo** | Manual: tienes que descargar nuevas versiones | Automática: el proveedor actualiza sin aviso |
| **Escalabilidad** | Limitada por hardware propio | Prácticamente ilimitada |
| **Mantenimiento** | Alto: hardware, drivers, actualizaciones | Ninguno |
| **Disponibilidad** | Depende de tu infraestructura | SLA del proveedor (suele ser 99.9%) |
| **Personalización** | Total: fine-tuning, system prompts propios | Limitada a lo que permite la API |
| **Compliance** | Tú controlas los datos en todo momento | Depende de los términos del proveedor |

### Cuándo inclinarse por cada opción

**Preferir local cuando:**
- Los datos son confidenciales o regulados (historiales médicos, contratos legales, datos financieros).
- El volumen de consultas es alto y el coste por token se volvería significativo.
- Necesitas operar sin conexión a internet.
- Quieres control total sobre el modelo y su comportamiento.

**Preferir cloud cuando:**
- Necesitas la máxima calidad del modelo disponible.
- El volumen es bajo o impredecible y no justifica una inversión en hardware.
- Quieres empezar rápido sin gestionar infraestructura.
- Necesitas capacidades multimodales avanzadas (visión, audio, código).

---

## 2. Análisis de costes

### Coste por millón de tokens (abril 2025, precios aproximados)

| Modelo | Tipo | Input ($/M tokens) | Output ($/M tokens) |
|--------|------|--------------------|---------------------|
| GPT-4o | Cloud | $2.50 | $10.00 |
| GPT-4o mini | Cloud | $0.15 | $0.60 |
| Claude 3.5 Haiku | Cloud | $0.80 | $4.00 |
| Claude 3.5 Sonnet | Cloud | $3.00 | $15.00 |
| Gemini 1.5 Flash | Cloud | $0.075 | $0.30 |
| llama3.2 (local, Ollama) | Local | $0 | $0 |
| mistral:7b (local) | Local | $0 | $0 |
| phi-3-mini (local) | Local | $0 | $0 |

### Cálculo de amortización de GPU

```python
# comparativa_costes.py
# Modelo de coste: cuándo compensa una GPU frente a la API cloud

def calcular_breakeven(
    precio_gpu_eur: float,
    tokens_por_mes: int,
    coste_input_por_millon: float,
    coste_output_por_millon: float,
    ratio_input_output: float = 0.7,  # 70% input, 30% output
    electricidad_eur_kwh: float = 0.15,
    consumo_gpu_w: float = 300,
    horas_uso_diario: float = 8,
    vida_util_meses: int = 36,
) -> dict:
    """
    Calcula en cuántos meses se amortiza una GPU frente a usar la API cloud.

    Args:
        precio_gpu_eur: Precio de compra de la GPU en euros.
        tokens_por_mes: Tokens procesados mensualmente.
        coste_input_por_millon: Precio de la API por millón de tokens de entrada.
        coste_output_por_millon: Precio de la API por millón de tokens de salida.
        ratio_input_output: Fracción del total que son tokens de entrada.
        electricidad_eur_kwh: Precio de la electricidad en EUR/kWh.
        consumo_gpu_w: Consumo de la GPU en vatios.
        horas_uso_diario: Horas de uso de la GPU al día.
        vida_util_meses: Vida útil estimada de la GPU en meses.

    Returns:
        Diccionario con costes mensuales y punto de equilibrio.
    """
    tokens_input = tokens_por_mes * ratio_input_output
    tokens_output = tokens_por_mes * (1 - ratio_input_output)

    # Coste mensual de la API cloud
    coste_cloud_mes = (
        (tokens_input / 1_000_000) * coste_input_por_millon
        + (tokens_output / 1_000_000) * coste_output_por_millon
    )

    # Coste mensual de electricidad (local)
    kwh_por_mes = (consumo_gpu_w / 1000) * horas_uso_diario * 30
    coste_electricidad_mes = kwh_por_mes * electricidad_eur_kwh

    # Amortización mensual de la GPU
    amortizacion_mes = precio_gpu_eur / vida_util_meses

    # Coste total mensual local
    coste_local_mes = coste_electricidad_mes + amortizacion_mes

    # Punto de equilibrio (meses hasta que la local es más barata)
    if coste_cloud_mes <= coste_local_mes:
        breakeven = None  # La nube siempre es más barata con este volumen
    else:
        # Mes en que el ahorro acumulado cubre el precio de la GPU
        ahorro_mensual = coste_cloud_mes - coste_electricidad_mes
        breakeven = precio_gpu_eur / ahorro_mensual

    return {
        "tokens_por_mes": tokens_por_mes,
        "coste_cloud_mes_eur": round(coste_cloud_mes, 2),
        "coste_local_mes_eur": round(coste_local_mes, 2),
        "ahorro_mensual_eur": round(coste_cloud_mes - coste_local_mes, 2),
        "breakeven_meses": round(breakeven, 1) if breakeven else None,
        "ahorro_total_vida_util_eur": round(
            (coste_cloud_mes - coste_local_mes) * vida_util_meses - precio_gpu_eur, 2
        ),
    }


# ─── Escenarios de ejemplo ─────────────────────────────────────────────────
escenarios = [
    {
        "descripcion": "Startup con uso moderado — GPU RTX 4070 (800 EUR) vs GPT-4o mini",
        "precio_gpu_eur": 800,
        "tokens_por_mes": 5_000_000,       # 5M tokens/mes
        "coste_input": 0.15,
        "coste_output": 0.60,
    },
    {
        "descripcion": "Empresa mediana — GPU RTX 4090 (1800 EUR) vs GPT-4o",
        "precio_gpu_eur": 1800,
        "tokens_por_mes": 50_000_000,      # 50M tokens/mes
        "coste_input": 2.50,
        "coste_output": 10.00,
    },
    {
        "descripcion": "Uso personal — sin GPU, solo CPU vs Claude Haiku",
        "precio_gpu_eur": 0,
        "tokens_por_mes": 500_000,         # 500K tokens/mes
        "coste_input": 0.80,
        "coste_output": 4.00,
    },
]

for escenario in escenarios:
    print(f"\n{'='*60}")
    print(f"{escenario['descripcion']}")
    print("="*60)

    resultado = calcular_breakeven(
        precio_gpu_eur=escenario["precio_gpu_eur"],
        tokens_por_mes=escenario["tokens_por_mes"],
        coste_input_por_millon=escenario["coste_input"],
        coste_output_por_millon=escenario["coste_output"],
    )

    print(f"  Tokens/mes:           {resultado['tokens_por_mes']:>12,}")
    print(f"  Coste cloud/mes:      {resultado['coste_cloud_mes_eur']:>12.2f} EUR")
    print(f"  Coste local/mes:      {resultado['coste_local_mes_eur']:>12.2f} EUR")
    print(f"  Ahorro mensual:       {resultado['ahorro_mensual_eur']:>12.2f} EUR")

    if resultado["breakeven_meses"]:
        print(f"  Punto equilibrio:     {resultado['breakeven_meses']:>10.1f} meses")
        print(f"  Ahorro total (36m):   {resultado['ahorro_total_vida_util_eur']:>12.2f} EUR")
    else:
        print("  La API cloud es más económica con este volumen.")
```

---

## 3. Privacidad y compliance

### Casos donde los datos no pueden salir de la infraestructura propia

| Sector | Regulación aplicable | Dato sensible | Solución recomendada |
|--------|---------------------|---------------|----------------------|
| **Sanidad** | HIPAA (EE.UU.), RGPD (EU) | Historiales clínicos, diagnósticos | IA local en servidores propios |
| **Legal** | Secreto profesional, RGPD | Contratos, comunicaciones con clientes | IA local o nube privada |
| **Finanzas** | PCI-DSS, MiFID II, RGPD | Datos bancarios, transacciones | IA local o acuerdo DPA con proveedor |
| **Educación** | FERPA (EE.UU.), LOPD (ES) | Expedientes académicos | IA local o nube con sede EU |
| **Gobierno** | ENS, RGPD | Datos ciudadanos | IA local en infraestructura pública |
| **RRHH** | RGPD | Evaluaciones, nóminas, datos personales | IA local |

### Qué certifica cada opción

**IA local (Ollama, Transformers):**
- Los datos nunca abandonan tu infraestructura.
- No hay procesador de datos de terceros.
- Compatible con RGPD sin necesidad de DPA adicional.
- Auditería total del procesamiento.

**APIs cloud — certificaciones típicas:**
- OpenAI Enterprise: SOC 2 Type II, HIPAA (con acuerdo BAA).
- Anthropic: SOC 2 Type II, en proceso de certificaciones adicionales.
- Azure OpenAI Service: ISO 27001, SOC 1/2/3, HIPAA, PCI-DSS, FedRAMP.
- Google Vertex AI: ISO 27001, SOC 2, HIPAA, PCI-DSS.

> **Nota**: incluso con certificaciones, enviar datos personales a una API cloud requiere un Acuerdo de Procesamiento de Datos (DPA) firmado y una base legal bajo el RGPD.

---

## 4. Calidad de respuesta

Benchmark informal con la misma pregunta técnica enviada a cuatro modelos diferentes (resultados representativos, no rigurosos):

**Pregunta:** "Explica en detalle cómo funciona el algoritmo de atención (attention mechanism) en los transformers. Incluye la fórmula matemática y un ejemplo de por qué es mejor que las RNN."

```python
# benchmark_informal.py
# Requiere: pip install ollama openai anthropic
import time
import ollama
from openai import OpenAI
import anthropic

PREGUNTA = (
    "Explica en detalle cómo funciona el mecanismo de atención (attention mechanism) "
    "en los transformers. Incluye la fórmula matemática y explica por qué es superior "
    "a las RNN para capturar dependencias de largo alcance."
)

resultados = {}

# ─── llama3.2 (local, Ollama) ──────────────────────────────────────────────
print("Consultando llama3.2 (local)...")
t0 = time.time()
resp = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": PREGUNTA}],
)
resultados["llama3.2 (local)"] = {
    "respuesta": resp["message"]["content"],
    "tiempo_s": round(time.time() - t0, 2),
}

# ─── Mistral (local, Ollama) ───────────────────────────────────────────────
print("Consultando mistral (local)...")
t0 = time.time()
resp = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": PREGUNTA}],
)
resultados["mistral:7b (local)"] = {
    "respuesta": resp["message"]["content"],
    "tiempo_s": round(time.time() - t0, 2),
}

# ─── Claude Haiku (cloud) ──────────────────────────────────────────────────
print("Consultando Claude Haiku (cloud)...")
cliente_anthropic = anthropic.Anthropic()  # Requiere ANTHROPIC_API_KEY
t0 = time.time()
msg = cliente_anthropic.messages.create(
    model="claude-haiku-20240307",
    max_tokens=1024,
    messages=[{"role": "user", "content": PREGUNTA}],
)
resultados["Claude Haiku (cloud)"] = {
    "respuesta": msg.content[0].text,
    "tiempo_s": round(time.time() - t0, 2),
}

# ─── GPT-4o (cloud) ───────────────────────────────────────────────────────
print("Consultando GPT-4o (cloud)...")
cliente_openai = OpenAI()  # Requiere OPENAI_API_KEY
t0 = time.time()
comp = cliente_openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": PREGUNTA}],
    max_tokens=1024,
)
resultados["GPT-4o (cloud)"] = {
    "respuesta": comp.choices[0].message.content,
    "tiempo_s": round(time.time() - t0, 2),
}

# ─── Mostrar resumen ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("RESUMEN DE RESULTADOS")
print("="*70)

criterios = [
    "Incluye fórmula Attention(Q,K,V) = softmax(QK^T/√d_k)V",
    "Explica Q, K, V con claridad",
    "Compara con RNN y dependencias de largo alcance",
    "Menciona complejidad cuadrática O(n²)",
    "Proporciona intuición o ejemplo concreto",
]

for modelo, datos in resultados.items():
    print(f"\n{modelo} (tiempo: {datos['tiempo_s']}s)")
    print(f"  Longitud respuesta: {len(datos['respuesta'])} caracteres")
    respuesta_lower = datos["respuesta"].lower()

    for criterio in criterios:
        # Verificación heurística simple por palabras clave
        palabras_clave = {
            "Incluye fórmula": ["softmax", "qk", "√d"],
            "Explica Q, K, V": ["query", "key", "value"],
            "Compara con RNN": ["rnn", "recurrent", "dependencia"],
            "Menciona complejidad": ["o(n²)", "o(n^2)", "cuadrática", "cuadratica"],
            "Proporciona intuición": ["por ejemplo", "imagine", "imagina", "intuitivamente"],
        }
        clave_busqueda = list(palabras_clave.keys())[criterios.index(criterio)]
        terminos = palabras_clave[clave_busqueda]
        presente = any(t in respuesta_lower for t in terminos)
        print(f"  {'[OK]' if presente else '[ ]'} {criterio}")
```

### Observaciones típicas del benchmark

| Modelo | Precisión técnica | Claridad | Velocidad | Coste |
|--------|------------------|----------|-----------|-------|
| llama3.2:7b (local) | Buena | Buena | ~15s en CPU, ~3s con GPU | $0 |
| mistral:7b (local) | Muy buena | Muy buena | ~12s en CPU, ~2s con GPU | $0 |
| Claude Haiku (cloud) | Excelente | Excelente | ~1.5s | ~$0.002 |
| GPT-4o (cloud) | Excelente | Excelente | ~2s | ~$0.025 |

**Conclusión del benchmark**: para tareas técnicas complejas, los modelos cloud siguen siendo superiores en calidad y latencia, pero modelos locales de 7B como Mistral son suficientemente buenos para la mayoría de casos de uso del mundo real.

---

## 5. Arquitectura híbrida

Un router inteligente que selecciona automáticamente entre ejecución local y cloud según la complejidad, sensibilidad y coste estimado de cada consulta.

```python
# router_hibrido.py
# pip install ollama openai anthropic tiktoken
import re
import ollama
from openai import OpenAI
import anthropic

# ─── Clasificadores de la consulta ────────────────────────────────────────

PATRONES_DATOS_SENSIBLES = [
    r"\b(contraseña|password|clave|token|api[ _]?key)\b",
    r"\b(dni|nif|pasaporte|número de seguridad social)\b",
    r"\b(tarjeta|iban|cuenta bancaria|número de cuenta)\b",
    r"\b(diagnóstico|historial médico|paciente|medicamento)\b",
    r"\b(contrato|acuerdo|cláusula confidencial)\b",
]

INDICADORES_COMPLEJIDAD_ALTA = [
    "razona paso a paso",
    "análisis profundo",
    "compara y contrasta",
    "evalúa críticamente",
    "diseña una arquitectura",
    "demuestra matemáticamente",
    "plan detallado",
]


def es_dato_sensible(texto: str) -> bool:
    """Detecta si el texto contiene datos potencialmente sensibles."""
    texto_lower = texto.lower()
    return any(re.search(patron, texto_lower) for patron in PATRONES_DATOS_SENSIBLES)


def estimar_complejidad(texto: str) -> str:
    """Estima la complejidad de la consulta: 'baja', 'media' o 'alta'."""
    texto_lower = texto.lower()

    if any(ind in texto_lower for ind in INDICADORES_COMPLEJIDAD_ALTA):
        return "alta"

    # Longitud como proxy de complejidad
    palabras = len(texto.split())
    if palabras > 100:
        return "alta"
    if palabras > 30:
        return "media"
    return "baja"


def estimar_tokens(texto: str) -> int:
    """Estimación rápida de tokens (1 token ≈ 4 caracteres)."""
    return len(texto) // 4


# ─── Modelos disponibles ───────────────────────────────────────────────────

MODELOS = {
    "local_rapido": {
        "nombre": "llama3.2:1b",
        "tipo": "local",
        "coste_por_millon": 0.0,
        "descripcion": "Modelo local ligero, respuestas rápidas",
    },
    "local_calidad": {
        "nombre": "mistral",
        "tipo": "local",
        "coste_por_millon": 0.0,
        "descripcion": "Modelo local de mayor calidad",
    },
    "cloud_economico": {
        "nombre": "gpt-4o-mini",
        "tipo": "cloud_openai",
        "coste_por_millon": 0.15,
        "descripcion": "API cloud económica, buena calidad",
    },
    "cloud_premium": {
        "nombre": "gpt-4o",
        "tipo": "cloud_openai",
        "coste_por_millon": 2.50,
        "descripcion": "API cloud de máxima calidad",
    },
}


# ─── Lógica del router ─────────────────────────────────────────────────────

def seleccionar_modelo(
    consulta: str,
    forzar_local: bool = False,
    presupuesto_max_eur: float = 0.01,
) -> dict:
    """
    Decide qué modelo usar según las características de la consulta.

    Args:
        consulta: El texto de la consulta del usuario.
        forzar_local: Si True, siempre usa un modelo local.
        presupuesto_max_eur: Coste máximo aceptable por consulta.

    Returns:
        Diccionario con la configuración del modelo seleccionado.
    """
    sensible = es_dato_sensible(consulta)
    complejidad = estimar_complejidad(consulta)
    tokens_estimados = estimar_tokens(consulta)

    razonamiento = []

    # Regla 1: Datos sensibles → siempre local
    if sensible or forzar_local:
        razonamiento.append("Datos sensibles detectados → ejecución local obligatoria")
        if complejidad == "alta":
            modelo = MODELOS["local_calidad"]
            razonamiento.append("Complejidad alta → modelo local de calidad (Mistral)")
        else:
            modelo = MODELOS["local_rapido"]
            razonamiento.append("Complejidad baja/media → modelo local rápido")
        return {"modelo": modelo, "razonamiento": razonamiento}

    # Regla 2: Calcular coste estimado de la API cloud
    coste_estimado = (tokens_estimados / 1_000_000) * MODELOS["cloud_economico"]["coste_por_millon"]

    # Regla 3: Complejidad alta con presupuesto disponible → cloud premium
    if complejidad == "alta" and coste_estimado < presupuesto_max_eur:
        modelo = MODELOS["cloud_premium"]
        razonamiento.append(f"Complejidad alta + coste aceptable ({coste_estimado:.4f} EUR) → cloud premium")
        return {"modelo": modelo, "razonamiento": razonamiento}

    # Regla 4: Complejidad media con presupuesto → cloud económico
    if complejidad == "media" and coste_estimado < presupuesto_max_eur:
        modelo = MODELOS["cloud_economico"]
        razonamiento.append(f"Complejidad media + coste aceptable ({coste_estimado:.4f} EUR) → cloud económico")
        return {"modelo": modelo, "razonamiento": razonamiento}

    # Regla 5: Coste demasiado alto o complejidad baja → local
    modelo = MODELOS["local_calidad"]
    razonamiento.append(f"Complejidad baja o coste fuera de presupuesto → local")
    return {"modelo": modelo, "razonamiento": razonamiento}


# ─── Motor de inferencia unificado ────────────────────────────────────────

def inferir(consulta: str, config_modelo: dict) -> str:
    """Ejecuta la inferencia en el modelo seleccionado."""
    modelo = config_modelo["modelo"]

    if modelo["tipo"] == "local":
        resp = ollama.chat(
            model=modelo["nombre"],
            messages=[{"role": "user", "content": consulta}],
        )
        return resp["message"]["content"]

    elif modelo["tipo"] == "cloud_openai":
        cliente = OpenAI()
        comp = cliente.chat.completions.create(
            model=modelo["nombre"],
            messages=[{"role": "user", "content": consulta}],
        )
        return comp.choices[0].message.content

    raise ValueError(f"Tipo de modelo desconocido: {modelo['tipo']}")


# ─── Prueba del router ─────────────────────────────────────────────────────

consultas_prueba = [
    "¿Cuál es la capital de Francia?",
    "Diseña una arquitectura de microservicios para un sistema de pagos de alta disponibilidad con requisitos de latencia sub-100ms.",
    "Mi DNI es 12345678A y quiero saber si puedo usar IA para procesar datos de clientes.",
    "Explica brevemente qué es una API REST.",
]

print("=== Router híbrido local/cloud ===\n")

for consulta in consultas_prueba:
    config = seleccionar_modelo(consulta, presupuesto_max_eur=0.005)
    modelo = config["modelo"]

    print(f"Consulta: {consulta[:70]}{'...' if len(consulta) > 70 else ''}")
    print(f"Modelo seleccionado: {modelo['nombre']} ({modelo['tipo']})")
    print(f"Motivo: {' | '.join(config['razonamiento'])}")

    # Descomentar para ejecutar la inferencia real:
    # respuesta = inferir(consulta, config)
    # print(f"Respuesta: {respuesta[:200]}...")

    print()
```

---

## 6. Tabla de decisión

| Caso de uso | Recomendación | Justificación |
|-------------|---------------|---------------|
| Chatbot de atención al cliente con datos de usuario | **Local** | Los datos del cliente (nombre, historial de compras) son personales; RGPD. |
| Asistente de escritura personal (sin datos sensibles) | **Cloud** | Máxima calidad, bajo volumen, sin restricciones de privacidad. |
| Análisis de contratos legales internos | **Local** | Secreto profesional; los contratos no pueden compartirse con terceros. |
| Generación de contenido de marketing | **Cloud** | Alta calidad creativa; el contenido no es sensible. |
| Clasificación de correos en un servidor corporativo | **Local** | Alto volumen + datos internos de la empresa. |
| Prototipo de chatbot para demo | **Cloud** | Rapidez de despliegue; no hay datos reales todavía. |
| Diagnóstico médico asistido por IA | **Local** (o cloud certificado HIPAA) | Datos de salud; regulación estricta. |
| Búsqueda semántica sobre documentación pública | **Cloud** o **Local** | Indiferente; elegir según volumen y coste. |
| Aplicación móvil con análisis de texto offline | **Transformers.js / Local** | Sin conexión; privacidad total. |
| Investigación y experimentación con modelos | **Local** | Control total, sin coste por experimento. |
| Producción con SLA estricto (99.9% uptime) | **Cloud** | Infraestructura gestionada por el proveedor. |
| Procesamiento de 100M tokens/día | **Local** (con GPU) | A ese volumen, la GPU se amortiza en semanas. |

---

## 7. Extensiones sugeridas

- **LiteLLM**: librería que unifica la interfaz de más de 100 proveedores de LLM (incluido Ollama) con la misma API, facilitando el cambio entre local y cloud (`pip install litellm`).
- **PrivateGPT**: solución completa para RAG privado sobre documentos propios, 100% local.
- **Semantic Router**: sistema de enrutado semántico para clasificar consultas y dirigirlas al modelo adecuado sin reglas manuales.
- **MLflow**: tracking de experimentos para comparar sistemáticamente modelos locales y cloud con métricas objetivas.
- **Weights & Biases**: monitorización de costes de API y rendimiento de modelos en producción.

---

**Fin del bloque**
