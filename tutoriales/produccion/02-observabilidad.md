# 02 — Observabilidad y tracing de LLMs

> **Bloque:** Producción · **Nivel:** Práctico · **Tiempo estimado:** 40 min

---

## Índice

1. [Por qué la observabilidad es crítica en producción](#1-por-qué-la-observabilidad-es-crítica-en-producción)
2. [Logging básico de llamadas a la API](#2-logging-básico-de-llamadas-a-la-api)
3. [Tracing con Langfuse](#3-tracing-con-langfuse)
4. [Métricas de negocio — dashboard de costes por usuario](#4-métricas-de-negocio--dashboard-de-costes-por-usuario)
5. [Alertas — detectar errores y latencia alta](#5-alertas--detectar-errores-y-latencia-alta)
6. [Extensiones sugeridas](#6-extensiones-sugeridas)

---

## 1. Por qué la observabilidad es crítica en producción

Un LLM en producción es una caja negra con efectos secundarios costosos: cada llamada consume tokens y tiempo. Sin observabilidad no puedes responder a preguntas básicas como:

- ¿Cuánto me está costando cada usuario?
- ¿Qué prompts están fallando y por qué?
- ¿Por qué la latencia aumentó esta semana?
- ¿Cuántos tokens usa de media mi aplicación?

**Los tres pilares de la observabilidad en LLMs:**

```
┌─────────────────────────────────────────────────────┐
│                  OBSERVABILIDAD                     │
│                                                     │
│  Logs        Métricas         Trazas (Traces)       │
│  ─────       ────────         ─────────────────     │
│  Qué pasó    Cuánto/cuándo    Flujo completo de      │
│  (texto)     (números)        una request            │
└─────────────────────────────────────────────────────┘
```

En este tutorial construiremos las tres capas, desde un decorador de logging básico hasta un sistema completo con Langfuse.

---

## 2. Logging básico de llamadas a la API

Un decorador Python que envuelve cualquier llamada a la API de Claude y registra tokens, latencia y coste estimado sin modificar el código existente.

```bash
pip install anthropic python-dotenv
```

```python
# observabilidad/logging_basico.py
import time
import json
import logging
import functools
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración del logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_calls.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("llm_observer")

# Precios orientativos en USD por millón de tokens (claude-sonnet-4-6)
PRECIO_INPUT_POR_MILLON = 3.00
PRECIO_OUTPUT_POR_MILLON = 15.00


def calcular_coste(tokens_entrada: int, tokens_salida: int) -> float:
    """Estima el coste en USD de una llamada."""
    coste_entrada = (tokens_entrada / 1_000_000) * PRECIO_INPUT_POR_MILLON
    coste_salida = (tokens_salida / 1_000_000) * PRECIO_OUTPUT_POR_MILLON
    return round(coste_entrada + coste_salida, 6)


# ---------------------------------------------------------------------------
# Decorador de logging
# ---------------------------------------------------------------------------

def log_llm_call(func):
    """
    Decorador que registra automáticamente métricas de cada llamada a la API.

    Uso:
        @log_llm_call
        def mi_llamada(...) -> anthropic.types.Message:
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        error = None
        respuesta = None

        try:
            respuesta = func(*args, **kwargs)
            return respuesta
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)

            if respuesta is not None and hasattr(respuesta, "usage"):
                tokens_entrada = respuesta.usage.input_tokens
                tokens_salida = respuesta.usage.output_tokens
                coste = calcular_coste(tokens_entrada, tokens_salida)
                modelo = respuesta.model
                stop_reason = respuesta.stop_reason
            else:
                tokens_entrada = tokens_salida = coste = 0
                modelo = kwargs.get("model", "desconocido")
                stop_reason = "error"

            registro = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "funcion": func.__name__,
                "modelo": modelo,
                "tokens_entrada": tokens_entrada,
                "tokens_salida": tokens_salida,
                "tokens_total": tokens_entrada + tokens_salida,
                "coste_usd": coste,
                "latencia_ms": latencia_ms,
                "stop_reason": stop_reason,
                "error": error,
            }

            if error:
                logger.error(json.dumps(registro, ensure_ascii=False))
            else:
                logger.info(json.dumps(registro, ensure_ascii=False))

    return wrapper


# ---------------------------------------------------------------------------
# Cliente instrumentado
# ---------------------------------------------------------------------------

client = anthropic.Anthropic()


@log_llm_call
def llamar_claude(
    prompt: str,
    system: str = "Eres un asistente útil.",
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 512,
) -> anthropic.types.Message:
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preguntas = [
        "¿Cuál es la capital de Japón?",
        "Explica qué es un transformer en 2 frases.",
        "Escribe un haiku sobre inteligencia artificial.",
    ]

    for pregunta in preguntas:
        print(f"\nPregunta: {pregunta}")
        respuesta = llamar_claude(pregunta)
        print(f"Respuesta: {respuesta.content[0].text[:100]}...")

    print("\nLogs guardados en llm_calls.log")
```

Los logs tienen formato JSON por línea (JSONL), fácil de parsear con `pandas` o `jq`:

```bash
# Ver los últimos 5 registros
tail -5 llm_calls.log | python -m json.tool

# Filtrar solo errores
grep '"error":' llm_calls.log | grep -v '"error": null'
```

---

## 3. Tracing con Langfuse

Langfuse es una plataforma open source de observabilidad para LLMs. Permite ver cada llamada como un árbol de trazas con entradas, salidas, tokens y costes. Tiene una versión cloud gratuita y se puede autoalojar con Docker.

```bash
pip install langfuse anthropic python-dotenv
```

**Paso 1: Crear cuenta y obtener credenciales**

1. Registrarse en [cloud.langfuse.com](https://cloud.langfuse.com) (gratuito)
2. Crear un proyecto
3. Copiar `LANGFUSE_PUBLIC_KEY` y `LANGFUSE_SECRET_KEY`

Fichero `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

```python
# observabilidad/tracing_langfuse.py
import os
from dotenv import load_dotenv
import anthropic
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

load_dotenv()

# Inicializar clientes
client = anthropic.Anthropic()
langfuse = Langfuse()  # Lee LANGFUSE_PUBLIC_KEY, SECRET_KEY y HOST del .env


# ---------------------------------------------------------------------------
# Decorador @observe — tracing automático de funciones
# ---------------------------------------------------------------------------

@observe(name="chatbot-principal")
def responder_usuario(
    pregunta: str,
    historial: list[dict],
    user_id: str = "anonimo",
) -> str:
    """
    Función principal del chatbot. El decorador @observe la registra
    como una traza en Langfuse con todas sus llamadas internas.
    """
    # Asociar la traza al usuario para métricas por usuario
    langfuse_context.update_current_trace(
        user_id=user_id,
        tags=["chatbot", "produccion"],
        metadata={"historial_turnos": len(historial)},
    )

    # Añadir la pregunta al historial
    historial.append({"role": "user", "content": pregunta})

    # Llamar a Claude (registrada como span hijo de la traza)
    mensaje = _llamar_modelo(historial)

    respuesta = mensaje.content[0].text
    historial.append({"role": "assistant", "content": respuesta})

    # Registrar métricas de negocio en la traza
    langfuse_context.update_current_observation(
        usage={
            "input": mensaje.usage.input_tokens,
            "output": mensaje.usage.output_tokens,
        }
    )

    return respuesta


@observe(name="llamada-claude", as_type="generation")
def _llamar_modelo(historial: list[dict]) -> anthropic.types.Message:
    """Llamada al modelo — registrada como 'generation' en Langfuse."""
    modelo = "claude-sonnet-4-6"

    # Informar a Langfuse sobre el modelo y los parámetros antes de llamar
    langfuse_context.update_current_observation(
        model=modelo,
        model_parameters={"max_tokens": 512, "temperature": 1.0},
        input=historial,
    )

    mensaje = client.messages.create(
        model=modelo,
        max_tokens=512,
        system="Eres un asistente experto en inteligencia artificial. Responde en español.",
        messages=historial,
    )

    # Registrar la respuesta y el uso de tokens
    langfuse_context.update_current_observation(
        output=mensaje.content[0].text,
        usage={
            "input": mensaje.usage.input_tokens,
            "output": mensaje.usage.output_tokens,
        },
    )

    return mensaje


# ---------------------------------------------------------------------------
# Tracing manual con spans (para flujos más complejos)
# ---------------------------------------------------------------------------

def flujo_con_rag(pregunta: str, user_id: str = "anonimo") -> str:
    """
    Ejemplo de tracing manual para un flujo RAG con múltiples pasos.
    Cada paso se registra como un span independiente.
    """
    # Crear traza principal
    traza = langfuse.trace(
        name="flujo-rag",
        user_id=user_id,
        input={"pregunta": pregunta},
        tags=["rag"],
    )

    # Span 1: Recuperación de documentos
    span_retrieval = traza.span(
        name="recuperar-documentos",
        input={"query": pregunta},
    )
    # Aquí iría la llamada al vector store
    documentos = [f"Documento de ejemplo sobre: {pregunta[:30]}"]
    span_retrieval.end(output={"documentos": documentos, "num_docs": len(documentos)})

    # Span 2: Generación (llamada al LLM)
    span_gen = traza.generation(
        name="generar-respuesta",
        model="claude-sonnet-4-6",
        model_parameters={"max_tokens": 400},
        input=[
            {"role": "user", "content": f"Contexto: {documentos[0]}\n\nPregunta: {pregunta}"}
        ],
    )

    mensaje = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system="Usa el contexto proporcionado para responder la pregunta.",
        messages=[
            {"role": "user", "content": f"Contexto: {documentos[0]}\n\nPregunta: {pregunta}"}
        ],
    )
    respuesta = mensaje.content[0].text

    span_gen.end(
        output=respuesta,
        usage={
            "input": mensaje.usage.input_tokens,
            "output": mensaje.usage.output_tokens,
        },
    )

    # Cerrar la traza principal
    traza.update(output={"respuesta": respuesta})

    return respuesta


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Tracing con @observe ===")
    historial = []
    preguntas = [
        "¿Qué es el attention mechanism?",
        "¿Y cómo se diferencia del RNN?",
    ]
    for i, pregunta in enumerate(preguntas):
        print(f"\nTurno {i + 1}: {pregunta}")
        respuesta = responder_usuario(pregunta, historial, user_id="usuario-demo")
        print(f"Respuesta: {respuesta[:120]}...")

    print("\n=== Tracing manual con flujo RAG ===")
    resp_rag = flujo_con_rag("¿Qué es el prompt caching?", user_id="usuario-demo")
    print(f"Respuesta RAG: {resp_rag[:120]}...")

    # Flush antes de terminar el script
    langfuse.flush()
    print("\nTrazas enviadas a Langfuse. Revisa tu dashboard en cloud.langfuse.com")
```

**Ver las trazas:** abre [cloud.langfuse.com](https://cloud.langfuse.com), entra en tu proyecto y verás cada llamada con su árbol de spans, tokens consumidos y coste estimado.

---

## 4. Métricas de negocio — dashboard de costes por usuario

Con los logs en formato JSONL del paso 2, podemos analizar costes y patrones con `pandas`.

```bash
pip install pandas tabulate
```

```python
# observabilidad/dashboard_costes.py
"""
Analiza los logs de llm_calls.log y genera un informe de costes por usuario.
Asume que los logs tienen el formato JSONL generado en la sección 2.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def cargar_logs(ruta: str = "llm_calls.log") -> pd.DataFrame:
    """Carga el fichero JSONL de logs en un DataFrame de pandas."""
    registros = []
    ruta_path = Path(ruta)

    if not ruta_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de logs: {ruta}")

    with open(ruta_path, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            # El logger añade el timestamp y el nivel antes del JSON
            # Extraemos solo la parte JSON
            inicio_json = linea.find("{")
            if inicio_json == -1:
                continue
            try:
                registro = json.loads(linea[inicio_json:])
                registros.append(registro)
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["coste_usd"] = pd.to_numeric(df["coste_usd"], errors="coerce").fillna(0)
    df["tokens_total"] = pd.to_numeric(df["tokens_total"], errors="coerce").fillna(0)
    df["latencia_ms"] = pd.to_numeric(df["latencia_ms"], errors="coerce").fillna(0)
    return df


def informe_general(df: pd.DataFrame) -> None:
    """Muestra métricas generales de uso."""
    exitosas = df[df["error"].isna()]
    con_error = df[~df["error"].isna()]

    print("=" * 60)
    print("INFORME GENERAL")
    print("=" * 60)
    print(f"Período:             {df['timestamp'].min():%Y-%m-%d} — {df['timestamp'].max():%Y-%m-%d}")
    print(f"Llamadas totales:    {len(df)}")
    print(f"Llamadas exitosas:   {len(exitosas)}")
    print(f"Llamadas con error:  {len(con_error)}")
    print(f"Tokens totales:      {exitosas['tokens_total'].sum():,.0f}")
    print(f"Coste total (USD):   ${exitosas['coste_usd'].sum():.4f}")
    print(f"Latencia media:      {exitosas['latencia_ms'].mean():.0f} ms")
    print(f"Latencia p95:        {exitosas['latencia_ms'].quantile(0.95):.0f} ms")


def informe_por_modelo(df: pd.DataFrame) -> None:
    """Desglose de uso por modelo."""
    exitosas = df[df["error"].isna()]
    por_modelo = (
        exitosas.groupby("modelo")
        .agg(
            llamadas=("tokens_total", "count"),
            tokens_total=("tokens_total", "sum"),
            coste_total=("coste_usd", "sum"),
            latencia_media=("latencia_ms", "mean"),
        )
        .round(4)
        .sort_values("coste_total", ascending=False)
    )

    print("\n" + "=" * 60)
    print("DESGLOSE POR MODELO")
    print("=" * 60)
    print(por_modelo.to_string())


def informe_tendencia_diaria(df: pd.DataFrame) -> None:
    """Costes y tokens agrupados por día."""
    exitosas = df[df["error"].isna()].copy()
    exitosas["fecha"] = exitosas["timestamp"].dt.date
    por_dia = (
        exitosas.groupby("fecha")
        .agg(
            llamadas=("tokens_total", "count"),
            tokens=("tokens_total", "sum"),
            coste_usd=("coste_usd", "sum"),
        )
        .round(4)
    )

    print("\n" + "=" * 60)
    print("TENDENCIA DIARIA")
    print("=" * 60)
    print(por_dia.to_string())


def detectar_anomalias(df: pd.DataFrame, umbral_tokens: int = 5000) -> None:
    """Muestra llamadas que consumieron más tokens de lo esperado."""
    anomalias = df[df["tokens_total"] > umbral_tokens].copy()

    if anomalias.empty:
        print(f"\nNo hay llamadas con más de {umbral_tokens} tokens.")
        return

    print(f"\n{'='*60}")
    print(f"LLAMADAS ANOMALAS (> {umbral_tokens} tokens)")
    print("=" * 60)
    cols = ["timestamp", "funcion", "modelo", "tokens_total", "coste_usd", "latencia_ms"]
    print(anomalias[cols].to_string(index=False))


if __name__ == "__main__":
    # Generar logs de ejemplo si no existen
    log_path = Path("llm_calls.log")
    if not log_path.exists():
        print("No se encontró llm_calls.log. Ejecuta primero logging_basico.py")
    else:
        df = cargar_logs()
        informe_general(df)
        informe_por_modelo(df)
        informe_tendencia_diaria(df)
        detectar_anomalias(df, umbral_tokens=2000)
```

---

## 5. Alertas — detectar errores y latencia alta

Un sistema de alertas que monitorea el archivo de logs en tiempo real y notifica cuando se superan los umbrales.

```python
# observabilidad/sistema_alertas.py
"""
Monitor de logs en tiempo real con alertas por consola.
En producción se reemplazaría la función 'enviar_alerta' por
Slack, PagerDuty, email, etc.
"""
import json
import time
import logging
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

logger = logging.getLogger("alertas")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [ALERTA] %(message)s")

# ---------------------------------------------------------------------------
# Configuración de umbrales
# ---------------------------------------------------------------------------

UMBRALES = {
    "latencia_ms": 5_000,       # Alerta si una llamada tarda más de 5 segundos
    "tokens_por_llamada": 8_000, # Alerta si una sola llamada supera 8k tokens
    "coste_por_hora": 1.0,       # Alerta si el coste acumulado supera $1 en una hora
    "tasa_errores_pct": 20.0,    # Alerta si más del 20% de llamadas fallan en 5 min
    "latencia_p95_ms": 3_000,    # Alerta si la latencia p95 de los últimos 100 supera 3s
}

# Ventana deslizante para métricas agregadas
ventana_llamadas = deque(maxlen=100)  # Últimas 100 llamadas
ventana_5min: deque = deque()          # Llamadas en los últimos 5 minutos


def enviar_alerta(tipo: str, mensaje: str, datos: dict) -> None:
    """
    Envía una alerta. Reemplaza esta función con tu canal de notificación.
    Opciones: Slack webhook, email, PagerDuty, SMS, etc.
    """
    logger.warning(f"[{tipo}] {mensaje} | Datos: {datos}")
    # Ejemplo para Slack:
    # import httpx
    # httpx.post(SLACK_WEBHOOK_URL, json={"text": f":warning: [{tipo}] {mensaje}"})


def analizar_registro(registro: dict) -> None:
    """Evalúa un registro de log y lanza alertas si se superan los umbrales."""
    ts = datetime.fromisoformat(registro["timestamp"].replace("Z", "+00:00"))
    ahora = datetime.now(ts.tzinfo)

    # Guardar en ventana deslizante
    ventana_llamadas.append(registro)
    ventana_5min.append((ts, registro))

    # Limpiar registros de más de 5 minutos
    while ventana_5min and (ahora - ventana_5min[0][0]).total_seconds() > 300:
        ventana_5min.popleft()

    # --- Alerta 1: Latencia alta en llamada individual ---
    latencia = registro.get("latencia_ms", 0)
    if latencia > UMBRALES["latencia_ms"]:
        enviar_alerta(
            tipo="LATENCIA_ALTA",
            mensaje=f"Llamada tardó {latencia:.0f}ms (umbral: {UMBRALES['latencia_ms']}ms)",
            datos={"funcion": registro.get("funcion"), "modelo": registro.get("modelo"), "latencia_ms": latencia},
        )

    # --- Alerta 2: Tokens excesivos por llamada ---
    tokens = registro.get("tokens_total", 0)
    if tokens > UMBRALES["tokens_por_llamada"]:
        enviar_alerta(
            tipo="TOKENS_EXCESIVOS",
            mensaje=f"Llamada consumió {tokens} tokens (umbral: {UMBRALES['tokens_por_llamada']})",
            datos={"funcion": registro.get("funcion"), "tokens_total": tokens},
        )

    # --- Alerta 3: Tasa de errores en ventana de 5 minutos ---
    if len(ventana_5min) >= 10:
        errores = sum(1 for _, r in ventana_5min if r.get("error") is not None)
        tasa = (errores / len(ventana_5min)) * 100
        if tasa > UMBRALES["tasa_errores_pct"]:
            enviar_alerta(
                tipo="TASA_ERRORES",
                mensaje=f"Tasa de errores: {tasa:.1f}% en los últimos 5 min (umbral: {UMBRALES['tasa_errores_pct']}%)",
                datos={"errores": errores, "total": len(ventana_5min), "tasa_pct": round(tasa, 1)},
            )

    # --- Alerta 4: Latencia P95 alta en las últimas 100 llamadas ---
    if len(ventana_llamadas) >= 20:
        latencias = sorted(r.get("latencia_ms", 0) for r in ventana_llamadas)
        p95 = latencias[int(len(latencias) * 0.95)]
        if p95 > UMBRALES["latencia_p95_ms"]:
            enviar_alerta(
                tipo="LATENCIA_P95",
                mensaje=f"Latencia P95: {p95:.0f}ms (umbral: {UMBRALES['latencia_p95_ms']}ms)",
                datos={"p95_ms": p95, "muestras": len(latencias)},
            )


def monitorear_log(ruta: str = "llm_calls.log", intervalo_seg: float = 2.0) -> None:
    """
    Monitorea el archivo de logs en tiempo real (tail -f).
    Lee las líneas nuevas cada 'intervalo_seg' segundos.
    """
    ruta_path = Path(ruta)
    print(f"Monitoreando {ruta} (Ctrl+C para detener)...")

    # Ir al final del archivo para leer solo líneas nuevas
    with open(ruta_path, encoding="utf-8") as f:
        f.seek(0, 2)  # Moverse al final

        while True:
            linea = f.readline()
            if not linea:
                time.sleep(intervalo_seg)
                continue

            linea = linea.strip()
            inicio_json = linea.find("{")
            if inicio_json == -1:
                continue

            try:
                registro = json.loads(linea[inicio_json:])
                analizar_registro(registro)
            except (json.JSONDecodeError, KeyError):
                continue


# ---------------------------------------------------------------------------
# Test de alertas con registros simulados
# ---------------------------------------------------------------------------

def test_alertas() -> None:
    """Prueba el sistema de alertas con registros de ejemplo."""
    print("Probando sistema de alertas...\n")

    casos = [
        {
            "descripcion": "Llamada normal",
            "registro": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "funcion": "llamar_claude",
                "modelo": "claude-sonnet-4-6",
                "tokens_total": 800,
                "coste_usd": 0.002,
                "latencia_ms": 1200,
                "error": None,
            },
        },
        {
            "descripcion": "Latencia alta",
            "registro": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "funcion": "llamar_claude",
                "modelo": "claude-sonnet-4-6",
                "tokens_total": 1200,
                "coste_usd": 0.004,
                "latencia_ms": 7500,
                "error": None,
            },
        },
        {
            "descripcion": "Tokens excesivos",
            "registro": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "funcion": "procesar_documento",
                "modelo": "claude-sonnet-4-6",
                "tokens_total": 12000,
                "coste_usd": 0.15,
                "latencia_ms": 4200,
                "error": None,
            },
        },
    ]

    for caso in casos:
        print(f"Caso: {caso['descripcion']}")
        analizar_registro(caso["registro"])
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitorear_log()
    else:
        test_alertas()
        print("\nPara monitoreo en tiempo real: python sistema_alertas.py --monitor")
```

---

## 6. Extensiones sugeridas

- **OpenTelemetry**: exportar trazas a Grafana Tempo, Jaeger o Datadog usando el estándar OTEL para interoperabilidad.
- **Evaluación en producción**: integrar el LLM-as-judge del tutorial anterior para puntuar respuestas en tiempo real y registrar las puntuaciones en Langfuse.
- **Dashboards con Grafana**: exponer las métricas como Prometheus metrics y visualizarlas en dashboards preconfigurados.
- **Sesiones y usuarios**: añadir un `session_id` a cada traza para agrupar la conversación completa de un usuario y calcular el coste por sesión.
- **Muestreo**: en producción con alto volumen, loguear el 100% pero enviar solo el 10% a Langfuse para reducir costes de observabilidad.

---

**Siguiente:** [Optimización de costos](./03-optimizacion-costos.md)
