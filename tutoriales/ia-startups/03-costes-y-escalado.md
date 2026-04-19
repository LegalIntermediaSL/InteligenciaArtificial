# Costes y Escalado de IA en Startups

## El problema de los costes de IA a escala

Una startup que no mide los costes de IA desde el día 1 se encontrará,
a los 6 meses, con un coste por usuario que destruye el margen.

```
ERRORES COMUNES DE COSTES
──────────────────────────
1. Usar Opus/Sonnet para todo (usa Haiku donde es suficiente)
2. No cachear system prompts largos (pagar 5x de más)
3. No comprimir historiales largos (tokens que crecen infinitamente)
4. Sin límites de uso por usuario (1 usuario puede arruinarte el mes)
5. Sin alertas de coste (te enteras cuando recibes la factura)
```

## Monitor de costes en tiempo real

```python
import anthropic
import json
import time
from collections import defaultdict
from datetime import datetime, date

client = anthropic.Anthropic()

# Precios por 1M tokens (aproximados, verificar en anthropic.com/pricing)
PRECIOS_MODELO = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00},
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00}
}

class MonitorCostes:
    """Monitor de costes de API por usuario y por día."""

    def __init__(self):
        self._costes: dict = defaultdict(lambda: defaultdict(float))
        self._llamadas: dict = defaultdict(int)
        self.limite_diario_usd: dict = {}

    def registrar(self, user_id: str, modelo: str, tokens_input: int, tokens_output: int) -> dict:
        """Registra una llamada y devuelve el coste y alertas."""
        precios = PRECIOS_MODELO.get(modelo, {"input": 3.0, "output": 15.0})
        coste = (tokens_input * precios["input"] + tokens_output * precios["output"]) / 1_000_000

        hoy = date.today().isoformat()
        self._costes[user_id][hoy] += coste
        self._llamadas[user_id] += 1

        coste_hoy = self._costes[user_id][hoy]
        limite = self.limite_diario_usd.get(user_id, 2.0)  # $2/día por defecto

        return {
            "coste_llamada_usd": round(coste, 5),
            "coste_hoy_usd": round(coste_hoy, 4),
            "limite_diario_usd": limite,
            "pct_limite_usado": round(coste_hoy / limite * 100, 1),
            "alerta_limite": coste_hoy >= limite * 0.80,
            "bloqueado": coste_hoy >= limite
        }

    def resumen_usuario(self, user_id: str) -> dict:
        costes_7_dias = dict(list(self._costes[user_id].items())[-7:])
        total_7_dias = sum(costes_7_dias.values())
        return {
            "user_id": user_id,
            "llamadas_total": self._llamadas[user_id],
            "coste_7_dias_usd": round(total_7_dias, 4),
            "coste_mensual_estimado_usd": round(total_7_dias / 7 * 30, 2),
            "detalle_por_dia": {k: round(v, 4) for k, v in costes_7_dias.items()}
        }

    def top_usuarios_por_coste(self, n: int = 10) -> list:
        totales = {
            uid: sum(self._costes[uid].values())
            for uid in self._costes
        }
        return sorted(
            [{"user_id": uid, "coste_total_usd": round(coste, 4)}
             for uid, coste in totales.items()],
            key=lambda x: x["coste_total_usd"],
            reverse=True
        )[:n]


monitor = MonitorCostes()
monitor.limite_diario_usd = {"user_123": 1.50, "user_456": 5.00}


def llamada_con_control_costes(user_id: str, mensaje: str, system: str = "") -> dict:
    """Llama a la API con control de costes integrado."""

    # Verificar límite antes de llamar
    if monitor._costes[user_id].get(date.today().isoformat(), 0) >= monitor.limite_diario_usd.get(user_id, 2.0):
        return {"error": "Límite diario de uso alcanzado", "bloqueado": True}

    kwargs = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": mensaje}]
    }
    if system:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)
    texto = resp.content[0].text

    metricas = monitor.registrar(
        user_id, "claude-haiku-4-5-20251001",
        resp.usage.input_tokens, resp.usage.output_tokens
    )

    if metricas["alerta_limite"]:
        print(f"⚠️  {user_id}: {metricas['pct_limite_usado']}% del límite diario usado")

    return {"respuesta": texto, "metricas": metricas}


# Demo
for i in range(3):
    resultado = llamada_con_control_costes(
        "user_123",
        f"Pregunta número {i+1}: ¿Cuál es el artículo {i+1} del GDPR?",
        "Eres un experto en normativa europea de protección de datos."
    )
    m = resultado.get("metricas", {})
    print(f"Llamada {i+1}: ${m.get('coste_llamada_usd', 0):.5f} | Acumulado hoy: ${m.get('coste_hoy_usd', 0):.4f}")
```

## Optimización 1: Prompt Caching para system prompts largos

```python
def llamada_con_cache(system_largo: str, mensaje: str) -> dict:
    """
    Usa prompt caching para system prompts de >1024 tokens.
    El primer uso tiene coste de escritura (25% más), pero
    los siguientes tienen coste de lectura (10% del normal).
    """
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=[
            {
                "type": "text",
                "text": system_largo,
                "cache_control": {"type": "ephemeral"}  # activa el caché
            }
        ],
        messages=[{"role": "user", "content": mensaje}]
    )

    cache_write = resp.usage.cache_creation_input_tokens
    cache_read = resp.usage.cache_read_input_tokens
    tokens_normal = resp.usage.input_tokens

    # Calcular ahorro real
    coste_sin_cache = (tokens_normal + cache_write + cache_read) / 1_000_000 * 0.80
    coste_con_cache = (tokens_normal / 1_000_000 * 0.80 +
                       cache_write / 1_000_000 * 1.00 +  # escritura: +25%
                       cache_read / 1_000_000 * 0.08)    # lectura: -90%

    return {
        "respuesta": resp.content[0].text,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "ahorro_pct": round((1 - coste_con_cache / coste_sin_cache) * 100, 1) if cache_read > 0 else 0
    }
```

## Optimización 2: Router de modelos por complejidad

```python
def router_modelo(mensaje: str, historial: list) -> str:
    """
    Selecciona el modelo más económico que puede manejar la tarea.
    Ahorra hasta un 80% en coste sin perder calidad perceptible.
    """
    longitud = len(mensaje)
    tiene_historial_largo = len(historial) > 6

    # Indicadores de complejidad alta
    palabras_clave_sonnet = [
        "analiza", "razona", "compara", "evalúa", "estrategia",
        "diseña", "argumenta", "critica", "sintetiza"
    ]
    requiere_sonnet = any(p in mensaje.lower() for p in palabras_clave_sonnet)

    # Tarea simple → Haiku (80% más barato)
    if longitud < 200 and not requiere_sonnet and not tiene_historial_largo:
        return "claude-haiku-4-5-20251001"

    # Tarea compleja → Sonnet
    if requiere_sonnet or longitud > 1000:
        return "claude-sonnet-4-6"

    # Por defecto → Haiku
    return "claude-haiku-4-5-20251001"


def chat_con_router(mensaje: str, historial: list = None) -> dict:
    """Chat que selecciona el modelo según la complejidad."""
    historial = historial or []
    modelo = router_modelo(mensaje, historial)

    mensajes = historial + [{"role": "user", "content": mensaje}]

    resp = client.messages.create(
        model=modelo,
        max_tokens=800,
        messages=mensajes
    )

    return {
        "respuesta": resp.content[0].text,
        "modelo_usado": modelo,
        "tokens": resp.usage.input_tokens + resp.usage.output_tokens
    }


# Demo del router
consultas = [
    "¿Qué hora es en Madrid?",
    "Analiza las implicaciones estratégicas de implementar IA generativa en una empresa de logística de 500 empleados considerando los aspectos de change management, ROI y riesgos regulatorios",
    "Hola, ¿cómo estás?",
    "Compara y razona sobre las diferencias fundamentales entre RLHF y DPO como técnicas de alineamiento de LLMs"
]

print("ROUTER DE MODELOS")
print("-" * 50)
for consulta in consultas:
    resultado = chat_con_router(consulta)
    print(f"\n[{resultado['modelo_usado'].split('-')[1].upper()}] {consulta[:60]}...")
    print(f"  Tokens: {resultado['tokens']}")
```

## Optimización 3: Comprimir historial antes de que crezca

```python
def comprimir_si_necesario(historial: list, umbral_tokens: int = 4000) -> list:
    """Comprime el historial cuando supera el umbral para controlar costes."""

    # Estimación rápida (~4 chars/token)
    tokens_estimados = sum(len(str(m.get("content", ""))) // 4 for m in historial)

    if tokens_estimados <= umbral_tokens:
        return historial

    print(f"⚡ Comprimiendo historial: ~{tokens_estimados} tokens → objetivo {umbral_tokens}")

    # Mantener los últimos 4 turnos
    recientes = historial[-8:]
    antiguos = historial[:-8]

    texto_antiguo = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in antiguos
        if isinstance(m.get("content"), str)
    )

    resumen = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Resume en 3 bullets los puntos clave:\n{texto_antiguo[:2000]}"
        }]
    ).content[0].text

    return [
        {"role": "user", "content": f"[Contexto previo]: {resumen}"},
        {"role": "assistant", "content": "Entendido."}
    ] + recientes
```

## Dashboard de costes para inversores

```python
def generar_reporte_costes(monitor: MonitorCostes, periodo: str = "mes") -> str:
    """Genera un resumen de costes para el equipo/inversores."""

    top_usuarios = monitor.top_usuarios_por_coste(5)
    coste_total = sum(u["coste_total_usd"] for u in top_usuarios)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Genera un resumen ejecutivo de costes de IA para una startup:

Periodo: {periodo}
Coste total API (muestra de usuarios): ${coste_total:.2f}
Top usuarios por coste:
{json.dumps(top_usuarios, indent=2)}

El resumen debe incluir:
1. Coste por usuario activo (promedio y mediana)
2. Proyección mensual si escala a 1000 usuarios
3. 2 recomendaciones de optimización
4. ¿Es el modelo de negocio sostenible con estos costes?

Máximo 200 palabras, tono ejecutivo."""
        }]
    )
    return resp.content[0].text


reporte = generar_reporte_costes(monitor)
print("\n=== REPORTE EJECUTIVO DE COSTES ===")
print(reporte)
```

## Alertas automáticas por Slack/email

```python
import httpx  # pip install httpx

def enviar_alerta_slack(webhook_url: str, mensaje: str):
    """Envía alerta al canal de Slack del equipo."""
    if not webhook_url or webhook_url == "TU_WEBHOOK_URL":
        print(f"[ALERTA SIMULADA] {mensaje}")
        return

    httpx.post(webhook_url, json={"text": f"🚨 Alerta de costes IA\n{mensaje}"})

def verificar_y_alertar(monitor: MonitorCostes, webhook_url: str = "TU_WEBHOOK_URL"):
    """Verifica umbrales y envía alertas proactivas."""
    hoy = date.today().isoformat()

    for user_id, dias in monitor._costes.items():
        coste_hoy = dias.get(hoy, 0)
        limite = monitor.limite_diario_usd.get(user_id, 2.0)

        if coste_hoy >= limite:
            enviar_alerta_slack(
                webhook_url,
                f"Usuario {user_id} ha alcanzado su límite diario de ${limite}"
            )
        elif coste_hoy >= limite * 0.80:
            enviar_alerta_slack(
                webhook_url,
                f"Usuario {user_id} al {round(coste_hoy/limite*100)}% de su límite diario"
            )

# Demo
verificar_y_alertar(monitor)
```

## Recursos

- [Notebook interactivo](../notebooks/ia-startups/03-costes-y-escalado.ipynb)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [Langfuse — Observabilidad y costes](https://langfuse.com/docs/model-usage-and-cost)
