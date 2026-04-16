# 02 — A/B testing de LLMs

> **Bloque:** MLOps · **Nivel:** Práctico · **Tiempo estimado:** 50 min

---

## Índice

1. [Diseño del experimento: hipótesis y métricas](#1-diseño-del-experimento-hipótesis-y-métricas)
2. [Tamaño de muestra: cuántas peticiones necesitas](#2-tamaño-de-muestra-cuántas-peticiones-necesitas)
3. [Shadow deployment: comparación offline](#3-shadow-deployment-comparación-offline)
4. [Router A/B con FastAPI y Redis](#4-router-ab-con-fastapi-y-redis)
5. [Análisis estadístico con scipy](#5-análisis-estadístico-con-scipy)
6. [Rollout gradual: canary deployment](#6-rollout-gradual-canary-deployment)

---

## 1. Diseño del experimento: hipótesis y métricas

Un A/B test mal diseñado da resultados inútiles. Antes de escribir código debes definir:

```
┌──────────────────────────────────────────────────────────┐
│  PLANTILLA DE DISEÑO DE EXPERIMENTO                      │
│                                                          │
│  Hipótesis: "Cambiar de gpt-4o a claude-sonnet-4-6 en   │
│  el asistente de soporte reduce el tiempo de resolución  │
│  de tickets sin degradar la satisfacción del usuario."   │
│                                                          │
│  Variante A (control):  gpt-4o-mini, prompt v1          │
│  Variante B (test):     claude-sonnet-4-6, prompt v2    │
│                                                          │
│  Métricas primarias:                                     │
│    - Tasa de resolución en primer turno (proporción)     │
│    - Tiempo medio de conversación (continua, segundos)   │
│                                                          │
│  Métricas de guardarraíl:                                │
│    - Satisfacción usuario (CSAT, 1-5)                   │
│    - Tasa de escalado a humano                           │
│    - Latencia p95 < 3 segundos                           │
│                                                          │
│  Duración: 14 días o hasta alcanzar potencia estadística │
│  Nivel de significancia: α = 0.05, potencia = 0.80      │
└──────────────────────────────────────────────────────────┘
```

**Errores comunes en A/B tests de LLMs:**

- **Contaminación de variantes**: el mismo usuario recibe A y B en sesiones distintas. Solución: asignar por `user_id`, no por request.
- **Múltiples comparaciones**: comparar 5 métricas con α=0.05 da una probabilidad real de falso positivo de ~23%. Usa corrección de Bonferroni o define una sola métrica primaria.
- **Novedad del efecto**: las métricas mejoran temporalmente solo porque algo cambió. Ejecuta el test al menos 7 días.
- **Tráfico no estacionario**: si el tráfico varía por hora/día, asegúrate de que las variantes tienen igual representación temporal.

---

## 2. Tamaño de muestra: cuántas peticiones necesitas

```bash
pip install scipy numpy pandas
```

```python
# mlops/ab_sample_size.py
"""
Cálculo del tamaño de muestra para A/B tests de LLMs.
Cubre métricas de proporción (tasa de resolución) y continuas (latencia).
"""
import numpy as np
from scipy import stats
from scipy.stats import norm
import math


# ---------------------------------------------------------------------------
# Tamaño de muestra para métricas de proporción
# (tasas de éxito, CSAT binario, escalados, etc.)
# ---------------------------------------------------------------------------

def tamaño_muestra_proporcion(
    tasa_base: float,
    efecto_minimo_detectable: float,
    alpha: float = 0.05,
    potencia: float = 0.80,
    bilateral: bool = True,
) -> int:
    """
    Calcula el tamaño de muestra por variante para comparar dos proporciones.

    Args:
        tasa_base: Tasa esperada en el grupo de control (entre 0 y 1).
        efecto_minimo_detectable: Diferencia absoluta mínima que quieres detectar.
                                  Ej: 0.05 significa detectar un cambio de 65% a 70%.
        alpha: Nivel de significancia (probabilidad de falso positivo).
        potencia: Probabilidad de detectar el efecto si existe (1 - β).
        bilateral: True para test de dos colas (detecta mejora o empeoramiento).

    Returns:
        Número de muestras necesarias por variante.
    """
    alpha_ajustado = alpha / 2 if bilateral else alpha
    z_alpha = norm.ppf(1 - alpha_ajustado)
    z_beta = norm.ppf(potencia)

    p1 = tasa_base
    p2 = tasa_base + efecto_minimo_detectable

    # Fórmula estándar para dos proporciones
    numerador = (z_alpha * math.sqrt(2 * p1 * (1 - p1)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominador = (p2 - p1) ** 2

    n = math.ceil(numerador / denominador)
    return n


# ---------------------------------------------------------------------------
# Tamaño de muestra para métricas continuas
# (latencia, tokens generados, longitud de respuesta, etc.)
# ---------------------------------------------------------------------------

def tamaño_muestra_media(
    media_base: float,
    desviacion_tipica: float,
    efecto_minimo_detectable: float,
    alpha: float = 0.05,
    potencia: float = 0.80,
) -> int:
    """
    Calcula el tamaño de muestra por variante para comparar dos medias (t-test).

    Args:
        media_base: Media esperada del grupo control.
        desviacion_tipica: Desviación típica esperada (estimada de datos históricos).
        efecto_minimo_detectable: Diferencia absoluta de medias que quieres detectar.

    Returns:
        Número de muestras necesarias por variante.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(potencia)

    d = efecto_minimo_detectable / desviacion_tipica  # Tamaño del efecto (Cohen's d)
    n = math.ceil(2 * ((z_alpha + z_beta) / d) ** 2)
    return n


# ---------------------------------------------------------------------------
# Estimador de duración del experimento
# ---------------------------------------------------------------------------

def estimar_duracion(
    n_por_variante: int,
    peticiones_diarias: int,
    num_variantes: int = 2,
    fraccion_tráfico: float = 1.0,
) -> dict:
    """
    Estima cuántos días necesita el experimento.

    Args:
        n_por_variante: Muestras necesarias por variante.
        peticiones_diarias: Peticiones al día en el servicio.
        num_variantes: Número de variantes (2 para A/B clásico).
        fraccion_tráfico: Fracción del tráfico asignada al experimento (0-1).
    """
    peticiones_experimento = peticiones_diarias * fraccion_tráfico
    peticiones_por_variante_dia = peticiones_experimento / num_variantes
    dias_necesarios = math.ceil(n_por_variante / peticiones_por_variante_dia)

    return {
        "n_por_variante": n_por_variante,
        "total_peticiones_necesarias": n_por_variante * num_variantes,
        "dias_estimados": dias_necesarios,
        "fecha_fin_estimada": f"Día {dias_necesarios} desde el inicio",
    }


# ---------------------------------------------------------------------------
# Resumen de planificación del experimento
# ---------------------------------------------------------------------------

def planificar_experimento(
    nombre: str,
    tasa_base: float,
    efecto_esperado: float,
    peticiones_diarias: int,
    media_base: float | None = None,
    std_base: float | None = None,
    efecto_continuo: float | None = None,
) -> None:
    """Imprime el plan completo del experimento."""
    print(f"\n{'='*60}")
    print(f"PLAN DE EXPERIMENTO: {nombre}")
    print(f"{'='*60}")

    # Métrica de proporción
    n_prop = tamaño_muestra_proporcion(tasa_base, efecto_esperado)
    duracion_prop = estimar_duracion(n_prop, peticiones_diarias)
    print(f"\nMétrica de proporción:")
    print(f"  Tasa base:          {tasa_base:.1%}")
    print(f"  Efecto a detectar:  {efecto_esperado:+.1%} ({tasa_base:.1%} → {tasa_base+efecto_esperado:.1%})")
    print(f"  N por variante:     {n_prop:,}")
    print(f"  Duración estimada:  {duracion_prop['dias_estimados']} días "
          f"({peticiones_diarias:,} peticiones/día)")

    # Métrica continua (opcional)
    if media_base is not None and std_base is not None and efecto_continuo is not None:
        n_cont = tamaño_muestra_media(media_base, std_base, efecto_continuo)
        duracion_cont = estimar_duracion(n_cont, peticiones_diarias)
        print(f"\nMétrica continua (latencia):")
        print(f"  Media base:         {media_base:.0f} ms")
        print(f"  Desv. típica:       {std_base:.0f} ms")
        print(f"  Efecto a detectar:  {efecto_continuo:+.0f} ms")
        print(f"  N por variante:     {n_cont:,}")
        print(f"  Duración estimada:  {duracion_cont['dias_estimados']} días")

        n_final = max(n_prop, n_cont)
        print(f"\nN final (máximo de ambas): {n_final:,} por variante")


if __name__ == "__main__":
    planificar_experimento(
        nombre="Migración GPT-4o-mini → Claude Sonnet",
        tasa_base=0.65,           # 65% de resolución en primer turno (control)
        efecto_esperado=0.05,     # Queremos detectar mejora de 5 puntos porcentuales
        peticiones_diarias=5000,
        media_base=1800,          # 1800ms de latencia media
        std_base=600,             # 600ms de desviación típica
        efecto_continuo=-200,     # Queremos detectar reducción de 200ms
    )
```

---

## 3. Shadow deployment: comparación offline

El shadow deployment envía cada request a ambos modelos pero solo devuelve la respuesta del modelo de control. El modelo B procesa en paralelo para comparar calidad sin afectar a usuarios.

```python
# mlops/shadow_deployment.py
"""
Shadow deployment: cada request se envía a dos modelos simultáneamente.
Solo se devuelve la respuesta del modelo A (control) al usuario.
Los resultados del modelo B se almacenan para análisis posterior.
"""
import asyncio
import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
import anthropic
import httpx

client = anthropic.AsyncAnthropic()

# Base de datos para almacenar comparaciones
DB_PATH = "shadow_comparaciones.db"


def inicializar_db() -> None:
    """Crea la tabla de comparaciones si no existe."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS comparaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            request_id TEXT NOT NULL,
            user_id TEXT,
            prompt TEXT NOT NULL,
            respuesta_a TEXT NOT NULL,
            respuesta_b TEXT NOT NULL,
            tokens_a INTEGER,
            tokens_b INTEGER,
            latencia_a_ms REAL,
            latencia_b_ms REAL,
            modelo_a TEXT,
            modelo_b TEXT,
            -- Campos para análisis posterior
            longitud_a INTEGER,
            longitud_b INTEGER,
            puntuacion_calidad_a REAL,   -- Relleno por el evaluador offline
            puntuacion_calidad_b REAL
        )
    """)
    con.commit()
    con.close()


async def llamar_modelo(
    modelo: str,
    prompt: str,
    system: str = "Eres un asistente de soporte al cliente. Responde en español.",
    max_tokens: int = 512,
) -> dict:
    """Llama a un modelo y devuelve respuesta + métricas de latencia."""
    inicio = time.perf_counter()
    try:
        mensaje = await client.messages.create(
            model=modelo,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        latencia_ms = (time.perf_counter() - inicio) * 1000
        return {
            "respuesta": mensaje.content[0].text,
            "tokens": mensaje.usage.input_tokens + mensaje.usage.output_tokens,
            "latencia_ms": latencia_ms,
            "modelo": mensaje.model,
            "error": None,
        }
    except Exception as e:
        latencia_ms = (time.perf_counter() - inicio) * 1000
        return {
            "respuesta": "",
            "tokens": 0,
            "latencia_ms": latencia_ms,
            "modelo": modelo,
            "error": str(e),
        }


async def shadow_request(
    prompt: str,
    request_id: str,
    user_id: str = "anonimo",
    modelo_a: str = "claude-haiku-4-5",     # Control: modelo actual en prod
    modelo_b: str = "claude-sonnet-4-6",    # Test: candidato
    system: str = "Eres un asistente de soporte al cliente. Responde en español.",
) -> str:
    """
    Ejecuta ambos modelos en paralelo y devuelve solo la respuesta del modelo A.
    La comparación se guarda en SQLite para análisis posterior.
    """
    # Ejecutar ambos modelos en paralelo (no bloqueante)
    resultado_a, resultado_b = await asyncio.gather(
        llamar_modelo(modelo_a, prompt, system),
        llamar_modelo(modelo_b, prompt, system),
    )

    # Guardar comparación para análisis offline
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO comparaciones
        (timestamp, request_id, user_id, prompt, respuesta_a, respuesta_b,
         tokens_a, tokens_b, latencia_a_ms, latencia_b_ms, modelo_a, modelo_b,
         longitud_a, longitud_b)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        request_id,
        user_id,
        prompt,
        resultado_a["respuesta"],
        resultado_b["respuesta"],
        resultado_a["tokens"],
        resultado_b["tokens"],
        resultado_a["latencia_ms"],
        resultado_b["latencia_ms"],
        resultado_a["modelo"],
        resultado_b["modelo"],
        len(resultado_a["respuesta"]),
        len(resultado_b["respuesta"]),
    ))
    con.commit()
    con.close()

    # SOLO devolver la respuesta del modelo de control
    return resultado_a["respuesta"]


# ---------------------------------------------------------------------------
# Análisis offline de comparaciones
# ---------------------------------------------------------------------------

def analizar_shadow_results(db_path: str = DB_PATH) -> None:
    """
    Analiza las comparaciones almacenadas y muestra estadísticas comparativas.
    """
    import pandas as pd

    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM comparaciones", con)
    con.close()

    if df.empty:
        print("No hay comparaciones registradas aún.")
        return

    print(f"\n{'='*60}")
    print(f"ANÁLISIS SHADOW DEPLOYMENT ({len(df)} comparaciones)")
    print(f"{'='*60}")

    print("\nLatencia (ms):")
    print(f"  Modelo A (control): media={df['latencia_a_ms'].mean():.0f} | "
          f"p50={df['latencia_a_ms'].median():.0f} | "
          f"p95={df['latencia_a_ms'].quantile(0.95):.0f}")
    print(f"  Modelo B (test):    media={df['latencia_b_ms'].mean():.0f} | "
          f"p50={df['latencia_b_ms'].median():.0f} | "
          f"p95={df['latencia_b_ms'].quantile(0.95):.0f}")

    print("\nTokens consumidos:")
    print(f"  Modelo A: {df['tokens_a'].mean():.0f} tokens/request")
    print(f"  Modelo B: {df['tokens_b'].mean():.0f} tokens/request")

    print("\nLongitud de respuesta (caracteres):")
    print(f"  Modelo A: {df['longitud_a'].mean():.0f} chars")
    print(f"  Modelo B: {df['longitud_b'].mean():.0f} chars")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def demo_shadow():
    """Ejecuta una serie de requests en modo shadow."""
    inicializar_db()
    prompts = [
        "¿Cuánto tiempo tarda un reembolso?",
        "Mi pedido llegó dañado, ¿qué hago?",
        "¿Puedo cambiar la dirección de envío una vez realizado el pedido?",
        "No puedo iniciar sesión en mi cuenta.",
        "¿Cuáles son los métodos de pago aceptados?",
    ]

    print("Ejecutando shadow deployment...")
    for i, prompt in enumerate(prompts):
        respuesta = await shadow_request(
            prompt=prompt,
            request_id=f"req-{i:04d}",
            user_id=f"user-{i % 3}",
        )
        print(f"  [{i+1}/{len(prompts)}] '{prompt[:40]}...' → {len(respuesta)} chars")

    analizar_shadow_results()


if __name__ == "__main__":
    asyncio.run(demo_shadow())
```

---

## 4. Router A/B con FastAPI y Redis

En producción, el router A/B asigna usuarios a variantes de forma determinista (siempre el mismo usuario recibe el mismo modelo) usando Redis como almacén rápido de asignaciones.

```bash
pip install fastapi uvicorn redis httpx
```

```python
# mlops/ab_router.py
"""
Router A/B para LLMs con FastAPI y Redis.
- Asignación determinista por user_id (mismo usuario, siempre misma variante)
- Porcentaje de tráfico configurable
- Registro de asignaciones para análisis
- Endpoint de control para cambiar el split sin reiniciar
"""
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Optional
import asyncio

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import anthropic

app = FastAPI(title="A/B Router LLM", version="1.0")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client: Optional[aioredis.Redis] = None

# Configuración del experimento (en producción, se almacena en Redis)
CONFIG_AB = {
    "experimento_activo": True,
    "porcentaje_variante_b": 50,         # % de tráfico que va al modelo B
    "modelo_a": "claude-haiku-4-5",      # Control
    "modelo_b": "claude-sonnet-4-6",     # Test
    "sistema_a": "Eres un asistente de soporte. Responde de forma concisa.",
    "sistema_b": "Eres un asistente de soporte experto. Sé detallado y empático.",
    "experimento_id": "exp-001-soporte",
}

anthropic_client = anthropic.AsyncAnthropic()


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

class SolicitudChat(BaseModel):
    mensaje: str
    historial: list[dict] = []


class RespuestaChat(BaseModel):
    respuesta: str
    variante: str          # "A" o "B" — útil para debugging
    request_id: str
    latencia_ms: float


# ---------------------------------------------------------------------------
# Lógica de asignación
# ---------------------------------------------------------------------------

def asignar_variante(user_id: str, porcentaje_b: int) -> str:
    """
    Asigna una variante de forma determinista basándose en el user_id.
    El mismo user_id siempre recibe la misma variante.
    Usa un hash para distribución uniforme.
    """
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    percentil = hash_val % 100
    return "B" if percentil < porcentaje_b else "A"


async def registrar_asignacion(
    request_id: str,
    user_id: str,
    variante: str,
    latencia_ms: float,
    tokens: int,
    experimento_id: str,
) -> None:
    """Registra la asignación en Redis para análisis posterior."""
    if redis_client is None:
        return
    datos = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "user_id": user_id,
        "variante": variante,
        "latencia_ms": latencia_ms,
        "tokens": tokens,
        "experimento_id": experimento_id,
    }
    # Guardar en una lista FIFO con TTL de 30 días
    await redis_client.lpush(f"ab:{experimento_id}:asignaciones", json.dumps(datos))
    await redis_client.expire(f"ab:{experimento_id}:asignaciones", 30 * 24 * 3600)

    # Contadores por variante para métricas rápidas
    await redis_client.incr(f"ab:{experimento_id}:conteo:{variante}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global redis_client
    try:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        print("Redis conectado")
    except Exception as e:
        print(f"Redis no disponible: {e}. Continuando sin persistencia.")
        redis_client = None


@app.post("/chat", response_model=RespuestaChat)
async def chat(
    solicitud: SolicitudChat,
    x_user_id: str = Header(default="anonimo"),
    x_request_id: str = Header(default=""),
) -> RespuestaChat:
    """
    Endpoint principal. Asigna al usuario a una variante y devuelve la respuesta
    del modelo correspondiente.
    """
    if not x_request_id:
        import uuid
        x_request_id = str(uuid.uuid4())[:8]

    # Determinar variante
    config = CONFIG_AB
    if not config["experimento_activo"]:
        variante = "A"
    else:
        variante = asignar_variante(x_user_id, config["porcentaje_variante_b"])

    modelo = config[f"modelo_{variante.lower()}"]
    system = config[f"sistema_{variante.lower()}"]

    # Construir historial de mensajes
    mensajes = solicitud.historial.copy()
    mensajes.append({"role": "user", "content": solicitud.mensaje})

    inicio = time.perf_counter()
    try:
        respuesta_api = await anthropic_client.messages.create(
            model=modelo,
            max_tokens=512,
            system=system,
            messages=mensajes,
        )
        latencia_ms = (time.perf_counter() - inicio) * 1000
        texto_respuesta = respuesta_api.content[0].text
        tokens = respuesta_api.usage.input_tokens + respuesta_api.usage.output_tokens

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del modelo: {e}")

    # Registrar asignación de forma asíncrona (no bloquea la respuesta)
    asyncio.create_task(
        registrar_asignacion(
            x_request_id, x_user_id, variante,
            latencia_ms, tokens, config["experimento_id"],
        )
    )

    return RespuestaChat(
        respuesta=texto_respuesta,
        variante=variante,
        request_id=x_request_id,
        latencia_ms=round(latencia_ms, 1),
    )


@app.get("/ab/config")
async def obtener_config() -> dict:
    """Devuelve la configuración actual del experimento."""
    return {
        "config": CONFIG_AB,
        "descripcion": "Modifica porcentaje_variante_b entre 0 y 100",
    }


@app.put("/ab/config")
async def actualizar_config(nuevo_porcentaje_b: int) -> dict:
    """
    Cambia el porcentaje de tráfico hacia la variante B en caliente.
    Útil para el rollout gradual (canary → full).
    """
    if not 0 <= nuevo_porcentaje_b <= 100:
        raise HTTPException(status_code=400, detail="Porcentaje debe estar entre 0 y 100")
    CONFIG_AB["porcentaje_variante_b"] = nuevo_porcentaje_b
    return {"mensaje": f"Porcentaje B actualizado a {nuevo_porcentaje_b}%", "config": CONFIG_AB}


@app.get("/ab/metricas")
async def obtener_metricas_ab() -> dict:
    """Devuelve conteos rápidos por variante desde Redis."""
    if redis_client is None:
        return {"error": "Redis no disponible"}

    exp_id = CONFIG_AB["experimento_id"]
    conteo_a = await redis_client.get(f"ab:{exp_id}:conteo:A") or "0"
    conteo_b = await redis_client.get(f"ab:{exp_id}:conteo:B") or "0"

    total = int(conteo_a) + int(conteo_b)
    return {
        "experimento_id": exp_id,
        "variante_a": {"solicitudes": int(conteo_a), "porcentaje": f"{int(conteo_a)/max(total,1)*100:.1f}%"},
        "variante_b": {"solicitudes": int(conteo_b), "porcentaje": f"{int(conteo_b)/max(total,1)*100:.1f}%"},
        "total": total,
    }


# uvicorn mlops.ab_router:app --reload
```

---

## 5. Análisis estadístico con scipy

Una vez recogidos los datos, el análisis determina si las diferencias son estadísticamente significativas o producto del azar.

```python
# mlops/ab_analysis.py
"""
Análisis estadístico de resultados de A/B test para LLMs.
Cubre: test de proporciones, t-test, visualización y reporte final.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Generación de datos simulados (representando un experimento real)
# ---------------------------------------------------------------------------

def generar_datos_experimento(
    n_a: int = 2500,
    n_b: int = 2450,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simula los resultados de un A/B test de asistente de soporte.
    Variante A: GPT-4o-mini (control) | Variante B: Claude Sonnet (test)
    """
    rng = np.random.default_rng(seed)

    # Variante A (control)
    df_a = pd.DataFrame({
        "variante": "A",
        "resuelto_primer_turno": rng.binomial(1, 0.64, n_a),  # Tasa real: 64%
        "csat": rng.choice([1, 2, 3, 4, 5], n_a, p=[0.05, 0.08, 0.17, 0.40, 0.30]),
        "latencia_ms": rng.gamma(shape=4, scale=450, size=n_a),
        "tokens": rng.normal(680, 120, n_a).astype(int).clip(50),
        "escalado_humano": rng.binomial(1, 0.12, n_a),
    })

    # Variante B (test) — ligeramente mejor en resolución, más lento
    df_b = pd.DataFrame({
        "variante": "B",
        "resuelto_primer_turno": rng.binomial(1, 0.69, n_b),  # Tasa real: 69%
        "csat": rng.choice([1, 2, 3, 4, 5], n_b, p=[0.03, 0.06, 0.15, 0.38, 0.38]),
        "latencia_ms": rng.gamma(shape=4, scale=520, size=n_b),
        "tokens": rng.normal(820, 150, n_b).astype(int).clip(50),
        "escalado_humano": rng.binomial(1, 0.09, n_b),
    })

    return pd.concat([df_a, df_b], ignore_index=True)


# ---------------------------------------------------------------------------
# Tests estadísticos
# ---------------------------------------------------------------------------

def test_proporciones(
    n_exito_a: int, n_total_a: int,
    n_exito_b: int, n_total_b: int,
    nombre_metrica: str = "tasa",
    alpha: float = 0.05,
) -> dict:
    """
    Test chi-cuadrado para comparar dos proporciones.
    Devuelve: estadístico, p-valor, intervalo de confianza y conclusión.
    """
    tabla_contingencia = np.array([
        [n_exito_a, n_total_a - n_exito_a],
        [n_exito_b, n_total_b - n_exito_b],
    ])
    chi2, p_valor, dof, _ = chi2_contingency(tabla_contingencia, correction=False)

    tasa_a = n_exito_a / n_total_a
    tasa_b = n_exito_b / n_total_b
    diferencia = tasa_b - tasa_a
    mejora_relativa = diferencia / tasa_a * 100

    # Intervalo de confianza para la diferencia de proporciones
    se = np.sqrt(tasa_a * (1 - tasa_a) / n_total_a + tasa_b * (1 - tasa_b) / n_total_b)
    z = stats.norm.ppf(1 - alpha / 2)
    ic_inferior = diferencia - z * se
    ic_superior = diferencia + z * se

    significativo = p_valor < alpha

    return {
        "metrica": nombre_metrica,
        "tasa_a": tasa_a,
        "tasa_b": tasa_b,
        "diferencia_absoluta": diferencia,
        "mejora_relativa_pct": mejora_relativa,
        "chi2": chi2,
        "p_valor": p_valor,
        "ic_95": (ic_inferior, ic_superior),
        "significativo": significativo,
        "conclusion": (
            f"SIGNIFICATIVO (p={p_valor:.4f} < α={alpha}): B {'mejora' if diferencia > 0 else 'empeora'} A"
            if significativo
            else f"NO SIGNIFICATIVO (p={p_valor:.4f} ≥ α={alpha})"
        ),
    }


def test_medias(
    valores_a: np.ndarray,
    valores_b: np.ndarray,
    nombre_metrica: str = "métrica",
    alpha: float = 0.05,
    usar_mann_whitney: bool = True,
) -> dict:
    """
    Compara dos grupos para métricas continuas.
    Usa Mann-Whitney U por defecto (más robusto para distribuciones asimétricas
    como la latencia). Opcionalmente usa t-test de Welch.
    """
    if usar_mann_whitney:
        # Mann-Whitney U: no asume distribución normal
        estadistico, p_valor = mannwhitneyu(valores_a, valores_b, alternative="two-sided")
        nombre_test = "Mann-Whitney U"
    else:
        # t-test de Welch: asume normalidad pero robusto a varianzas distintas
        estadistico, p_valor = ttest_ind(valores_a, valores_b, equal_var=False)
        nombre_test = "t-test Welch"

    diferencia_medias = np.mean(valores_b) - np.mean(valores_a)
    significativo = p_valor < alpha

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.std(valores_a) ** 2 + np.std(valores_b) ** 2) / 2)
    cohens_d = diferencia_medias / pooled_std if pooled_std > 0 else 0

    return {
        "metrica": nombre_metrica,
        "test": nombre_test,
        "media_a": np.mean(valores_a),
        "media_b": np.mean(valores_b),
        "mediana_a": np.median(valores_a),
        "mediana_b": np.median(valores_b),
        "diferencia_medias": diferencia_medias,
        "cohens_d": cohens_d,
        "p_valor": p_valor,
        "significativo": significativo,
        "conclusion": (
            f"SIGNIFICATIVO (p={p_valor:.4f}): B {'menor' if diferencia_medias < 0 else 'mayor'} que A"
            if significativo
            else f"NO SIGNIFICATIVO (p={p_valor:.4f})"
        ),
    }


# ---------------------------------------------------------------------------
# Reporte completo
# ---------------------------------------------------------------------------

def generar_reporte(df: pd.DataFrame, alpha: float = 0.05) -> None:
    """Genera el reporte completo del A/B test con corrección de Bonferroni."""
    df_a = df[df["variante"] == "A"]
    df_b = df[df["variante"] == "B"]

    n_metricas = 3  # Número de métricas que vamos a testear
    alpha_ajustado = alpha / n_metricas  # Corrección de Bonferroni

    print(f"\n{'='*65}")
    print("REPORTE A/B TEST: Asistente de soporte")
    print(f"{'='*65}")
    print(f"Variante A (control): n={len(df_a):,}  |  Variante B (test): n={len(df_b):,}")
    print(f"α ajustado (Bonferroni, {n_metricas} métricas): {alpha_ajustado:.4f}")

    resultados = []

    # --- 1. Tasa de resolución en primer turno ---
    r1 = test_proporciones(
        df_a["resuelto_primer_turno"].sum(), len(df_a),
        df_b["resuelto_primer_turno"].sum(), len(df_b),
        "Resolución primer turno",
        alpha=alpha_ajustado,
    )
    resultados.append(r1)

    # --- 2. Tasa de escalado a humano (métrica de guardarraíl) ---
    r2 = test_proporciones(
        df_a["escalado_humano"].sum(), len(df_a),
        df_b["escalado_humano"].sum(), len(df_b),
        "Tasa escalado humano",
        alpha=alpha_ajustado,
    )
    resultados.append(r2)

    # --- 3. Latencia (métrica continua) ---
    r3 = test_medias(
        df_a["latencia_ms"].values,
        df_b["latencia_ms"].values,
        "Latencia (ms)",
        alpha=alpha_ajustado,
    )
    resultados.append(r3)

    # Imprimir tabla resumen
    print(f"\n{'Métrica':<28} {'A':>8} {'B':>8} {'Δ':>8} {'p-valor':>10}  Resultado")
    print("-" * 75)
    for r in resultados:
        if "tasa_a" in r:
            val_a = f"{r['tasa_a']:.1%}"
            val_b = f"{r['tasa_b']:.1%}"
            delta = f"{r['diferencia_absoluta']:+.1%}"
        else:
            val_a = f"{r['media_a']:.0f}"
            val_b = f"{r['media_b']:.0f}"
            delta = f"{r['diferencia_medias']:+.0f}"

        sig_marker = "✓" if r["significativo"] else "✗"
        print(f"  {r['metrica']:<26} {val_a:>8} {val_b:>8} {delta:>8} {r['p_valor']:>10.4f}  {sig_marker} {r['conclusion'][:30]}")

    # Resumen ejecutivo
    metricas_sig = [r for r in resultados if r["significativo"]]
    print(f"\n{'='*65}")
    print(f"RESUMEN EJECUTIVO")
    print(f"  {len(metricas_sig)}/{len(resultados)} métricas con significancia estadística.")
    if metricas_sig:
        print("  Diferencias significativas encontradas en:")
        for r in metricas_sig:
            print(f"    - {r['metrica']}")
    print(f"\nRECOMENDACIÓN:")
    tasa_ok = resultados[0]["significativo"] and resultados[0]["diferencia_absoluta"] > 0
    guardarril_ok = not (resultados[1]["significativo"] and resultados[1]["diferencia_absoluta"] > 0)
    if tasa_ok and guardarril_ok:
        print("  PROMOVER variante B: mejora significativa en resolución sin degradar guardarraíles.")
    else:
        print("  MANTENER variante A: B no demuestra mejora o degrada guardarraíles.")


if __name__ == "__main__":
    df = generar_datos_experimento(n_a=2500, n_b=2450)
    generar_reporte(df)
```

---

## 6. Rollout gradual: canary deployment

El canary deployment incrementa el tráfico al nuevo modelo de forma progresiva, permitiendo detectar problemas antes de afectar al 100% de los usuarios.

```python
# mlops/canary_deployment.py
"""
Canary deployment para LLMs: incremento gradual del tráfico hacia el modelo B.
Monitorea métricas en tiempo real y hace rollback automático si se detectan problemas.
"""
import asyncio
import json
import time
import httpx
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class EstadoCanary:
    """Estado del canary deployment en memoria."""
    porcentaje_b: int = 0           # Porcentaje de tráfico hacia B (0-100)
    etapa_actual: str = "inicial"
    activo: bool = True

    # Ventana deslizante de métricas recientes (últimas 100 peticiones por variante)
    errores_a: deque = field(default_factory=lambda: deque(maxlen=100))
    errores_b: deque = field(default_factory=lambda: deque(maxlen=100))
    latencias_a: deque = field(default_factory=lambda: deque(maxlen=100))
    latencias_b: deque = field(default_factory=lambda: deque(maxlen=100))

    historial_etapas: list = field(default_factory=list)


@dataclass
class ConfigCanary:
    """Configuración del plan de rollout gradual."""
    etapas: list[tuple[str, int]] = field(default_factory=lambda: [
        # (nombre_etapa, porcentaje_tráfico_b)
        ("canary_5pct",    5),
        ("canary_10pct",  10),
        ("canary_25pct",  25),
        ("canary_50pct",  50),
        ("canary_75pct",  75),
        ("produccion",   100),
    ])
    espera_entre_etapas_s: int = 300   # 5 minutos entre etapas (en demo: 5s)
    umbral_tasa_error: float = 0.05    # Rollback si tasa error B > 5%
    umbral_latencia_p95_ms: float = 4000  # Rollback si p95 latencia B > 4s
    peticiones_minimas_analisis: int = 10  # Mínimo para analizar métricas


estado = EstadoCanary()
config_canary = ConfigCanary()


def registrar_resultado(variante: str, latencia_ms: float, error: bool) -> None:
    """Registra el resultado de una petición en las ventanas de estado."""
    if variante == "A":
        estado.errores_a.append(1 if error else 0)
        estado.latencias_a.append(latencia_ms)
    else:
        estado.errores_b.append(1 if error else 0)
        estado.latencias_b.append(latencia_ms)


def calcular_metricas(variante: str) -> dict:
    """Calcula métricas actuales de una variante."""
    errores = list(estado.errores_a if variante == "A" else estado.errores_b)
    latencias = list(estado.latencias_a if variante == "A" else estado.latencias_b)

    if not errores:
        return {"n": 0, "tasa_error": 0.0, "latencia_p95": 0.0, "latencia_media": 0.0}

    import numpy as np
    return {
        "n": len(errores),
        "tasa_error": sum(errores) / len(errores),
        "latencia_p95": float(np.percentile(latencias, 95)) if latencias else 0.0,
        "latencia_media": float(np.mean(latencias)) if latencias else 0.0,
    }


def evaluar_salud_canary() -> tuple[bool, str]:
    """
    Evalúa si el canary está sano o debe hacerse rollback.
    Returns: (sano: bool, razon: str)
    """
    metricas_b = calcular_metricas("B")
    if metricas_b["n"] < config_canary.peticiones_minimas_analisis:
        return True, f"Datos insuficientes ({metricas_b['n']} peticiones)"

    # Verificar tasa de error
    if metricas_b["tasa_error"] > config_canary.umbral_tasa_error:
        razon = (f"Tasa de error B ({metricas_b['tasa_error']:.1%}) "
                 f"supera umbral ({config_canary.umbral_tasa_error:.1%})")
        return False, razon

    # Verificar latencia P95
    if metricas_b["latencia_p95"] > config_canary.umbral_latencia_p95_ms:
        razon = (f"Latencia P95 B ({metricas_b['latencia_p95']:.0f}ms) "
                 f"supera umbral ({config_canary.umbral_latencia_p95_ms:.0f}ms)")
        return False, razon

    return True, "Métricas dentro de umbral"


def avanzar_etapa() -> bool:
    """
    Intenta avanzar a la siguiente etapa del rollout.
    Retorna True si avanzó, False si ya está en la última etapa.
    """
    etapas = config_canary.etapas
    porcentajes = [e[1] for e in etapas]

    if estado.porcentaje_b >= 100:
        return False

    # Encontrar la siguiente etapa
    siguiente_idx = next(
        (i for i, (_, p) in enumerate(etapas) if p > estado.porcentaje_b), None
    )
    if siguiente_idx is None:
        return False

    nombre, nuevo_pct = etapas[siguiente_idx]
    estado.porcentaje_b = nuevo_pct
    estado.etapa_actual = nombre
    estado.historial_etapas.append({
        "timestamp": datetime.utcnow().isoformat(),
        "etapa": nombre,
        "porcentaje_b": nuevo_pct,
    })
    print(f"  [{datetime.now():%H:%M:%S}] Avanzando a etapa '{nombre}' → {nuevo_pct}% tráfico a B")
    return True


def hacer_rollback(razon: str) -> None:
    """Revierte el canary al 0% de tráfico B."""
    estado.porcentaje_b = 0
    estado.etapa_actual = "rollback"
    estado.activo = False
    estado.historial_etapas.append({
        "timestamp": datetime.utcnow().isoformat(),
        "etapa": "ROLLBACK",
        "porcentaje_b": 0,
        "razon": razon,
    })
    print(f"\n  *** ROLLBACK AUTOMÁTICO ***")
    print(f"  Razón: {razon}")
    print(f"  Tráfico revertido al 100% modelo A\n")


async def orquestador_canary(
    espera_entre_etapas_s: int = 5,  # Reducido para demo
    verbose: bool = True,
) -> None:
    """
    Orquestador del canary: avanza etapas, monitorea y hace rollback si es necesario.
    En producción, este proceso corre como un worker o tarea de Prefect/Airflow.
    """
    print(f"Iniciando canary deployment...")
    print(f"Plan: {' → '.join(f'{p}%' for _, p in config_canary.etapas)}")

    while estado.activo and estado.porcentaje_b < 100:
        avanzado = avanzar_etapa()
        if not avanzado:
            break

        # Esperar y monitorear durante la etapa
        inicio_etapa = time.time()
        while time.time() - inicio_etapa < espera_entre_etapas_s:
            await asyncio.sleep(1)

            sano, razon = evaluar_salud_canary()
            if not sano:
                hacer_rollback(razon)
                return

            if verbose:
                metricas_b = calcular_metricas("B")
                print(f"    B: n={metricas_b['n']:3d} | "
                      f"error={metricas_b['tasa_error']:.1%} | "
                      f"p95={metricas_b['latencia_p95']:.0f}ms | "
                      f"estado=OK")

    if estado.activo and estado.porcentaje_b == 100:
        estado.etapa_actual = "completado"
        print("\n  Canary completado exitosamente. Modelo B al 100% del tráfico.")
    elif not estado.activo:
        print("  Canary detenido por rollback.")


# ---------------------------------------------------------------------------
# Simulación de tráfico para demo
# ---------------------------------------------------------------------------

async def simular_trafico(n_peticiones: int = 200) -> None:
    """Simula tráfico concurrent que registra resultados de A y B."""
    import random
    import hashlib

    for i in range(n_peticiones):
        user_id = f"user-{i % 50}"
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        percentil = hash_val % 100
        variante = "B" if percentil < estado.porcentaje_b else "A"

        # Simular latencia y error (B tiene 1% de error en escenario sano)
        latencia = random.gauss(1800 if variante == "A" else 2100, 400)
        error = random.random() < (0.01 if variante == "B" else 0.005)

        registrar_resultado(variante, max(latencia, 100), error)
        await asyncio.sleep(0.02)  # 50 req/s simuladas


async def demo_canary():
    """Demo completa del canary deployment."""
    # Ejecutar orquestador y simulación de tráfico en paralelo
    await asyncio.gather(
        orquestador_canary(espera_entre_etapas_s=5),
        simular_trafico(n_peticiones=1000),
    )

    print("\n=== HISTORIAL DE ETAPAS ===")
    for etapa in estado.historial_etapas:
        print(f"  {etapa['timestamp'][11:19]} | {etapa['etapa']:20} | {etapa['porcentaje_b']}% tráfico B")


if __name__ == "__main__":
    asyncio.run(demo_canary())
```

---

**Anterior:** [Registro y versionado de modelos](./01-registro-modelos.md) · **Siguiente:** [Detección de drift en producción](./03-deteccion-drift.md)
