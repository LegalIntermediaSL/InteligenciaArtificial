# 04 — Auditoría y trazabilidad en sistemas de IA

> **Bloque:** Seguridad en IA · **Nivel:** Avanzado · **Tiempo estimado:** 35 min

---

## Índice

1. [Por qué auditar sistemas de IA](#1-por-qué-auditar-sistemas-de-ia)
2. [Qué registrar: esquema completo de log](#2-qué-registrar-esquema-completo-de-log)
3. [Logger seguro para IA](#3-logger-seguro-para-ia)
4. [Detección de anomalías en uso](#4-detección-de-anomalías-en-uso)
5. [Dashboard de auditoría con pandas](#5-dashboard-de-auditoría-con-pandas)
6. [Alertas automáticas](#6-alertas-automáticas)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Por qué auditar sistemas de IA

Desplegar un modelo de IA en producción sin auditoría es como operar un sistema financiero sin contabilidad: todo funciona hasta que algo falla, y entonces no hay forma de saber qué ocurrió, cuándo ni por qué.

### 1.1 Responsabilidad legal y regulatoria

La normativa evoluciona rápidamente. El **AI Act europeo** (vigente desde 2024) exige trazabilidad en sistemas de alto riesgo. La **GDPR** impone obligaciones sobre el tratamiento de datos personales, incluidos los inputs a modelos. Sin logs auditables, demostrar cumplimiento es imposible.

### 1.2 Debugging y mejora continua

Los sistemas de IA fallan de formas silenciosas: respuestas incorrectas, alucinaciones, degradación gradual de calidad. Sin registros detallados es imposible:

- Reproducir un error específico reportado por un usuario.
- Identificar qué tipo de prompts generan más fallos.
- Medir si un cambio de modelo o prompt mejora o empeora los resultados.

### 1.3 Detección de abusos

Los sistemas de IA son objetivos frecuentes de:

- **Prompt injection**: intentos de saltarse las instrucciones del sistema.
- **Scraping masivo**: uso automatizado para extraer conocimiento propietario.
- **Generación de contenido prohibido**: intentos de eludir los filtros de seguridad.

Sin auditoría, estos ataques pueden pasar desapercibidos durante semanas.

### 1.4 Control de costes

Los modelos de lenguaje se facturan por token. Un usuario malintencionado o un bug en el código pueden disparar el gasto en horas. El monitoreo de tokens consumidos por usuario y sesión permite detectar anomalías antes de que el impacto sea significativo.

---

## 2. Qué registrar: esquema completo de log

Cada interacción con el modelo debe generar un registro estructurado. El formato recomendado es **JSONL** (JSON Lines): una línea por evento, fácil de procesar con pandas o herramientas de análisis de logs.

### 2.1 Campos obligatorios

| Campo | Tipo | Descripción |
|---|---|---|
| `timestamp` | ISO 8601 string | Momento exacto de la llamada al modelo |
| `usuario_id` | string | Identificador del usuario (anonimizable) |
| `session_id` | string | UUID de la sesión actual |
| `request_id` | string | UUID único de esta petición |
| `modelo` | string | Nombre del modelo usado (ej. `claude-haiku-4-5-20251001`) |
| `input_tokens` | int | Tokens enviados al modelo |
| `output_tokens` | int | Tokens generados por el modelo |
| `coste_estimado` | float | Coste en USD calculado según tarifa del modelo |
| `latencia_ms` | int | Tiempo de respuesta en milisegundos |
| `flags_seguridad` | list[str] | Lista de alertas activadas (puede ser vacía) |
| `ip` | string | Dirección IP del cliente (hasheada o enmascarada) |

### 2.2 Campos recomendados

| Campo | Tipo | Descripción |
|---|---|---|
| `endpoint` | string | Ruta de la API que procesó la petición |
| `status` | string | `"ok"`, `"error"`, `"bloqueado"` |
| `error_tipo` | string | Tipo de error si aplica |
| `prompt_hash` | string | SHA-256 del prompt (sin PII) para detectar duplicados |
| `sistema_version` | string | Versión del sistema que generó la llamada |

### 2.3 Ejemplo de registro

```json
{
  "timestamp": "2025-04-15T14:32:11.453Z",
  "usuario_id": "usr_a3f9b2",
  "session_id": "sess_7d1c4e8a-2b3f-4a9d-b8c1-3f2e1a4b5c6d",
  "request_id": "req_9e8d7c6b-5a4f-3e2d-1c0b-9a8f7e6d5c4b",
  "modelo": "claude-haiku-4-5-20251001",
  "input_tokens": 312,
  "output_tokens": 87,
  "coste_estimado": 0.000087,
  "latencia_ms": 1243,
  "flags_seguridad": [],
  "ip": "192.168.1.***",
  "endpoint": "/api/chat",
  "status": "ok",
  "prompt_hash": "a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5"
}
```

### 2.4 Qué NO registrar

Nunca almacenes en los logs:

- El contenido completo del prompt si contiene datos personales identificables.
- Contraseñas, tokens de API o credenciales que el usuario pudiera incluir accidentalmente.
- Datos de tarjetas de crédito o información financiera sensible.

La solución: **redactar automáticamente** estos datos antes de escribir el log (ver sección 3).

---

## 3. Logger seguro para IA

### 3.1 Diseño de la clase `AuditLogger`

El logger cumple tres funciones:

1. Serializar el registro en formato JSONL.
2. Redactar automáticamente PII (Información Personal Identificable) del contenido antes de guardarlo.
3. Proporcionar una interfaz simple para el resto del sistema.

```python
import json
import re
import hashlib
import uuid
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    """
    Logger de auditoría para sistemas de IA.
    Redacta PII automáticamente antes de persistir los registros.
    """

    # Patrones de PII a redactar
    PATRONES_PII = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "telefono_es": r"\b(?:\+34|0034)?[ -]?[6789]\d{2}[ -]?\d{3}[ -]?\d{3}\b",
        "telefono_intl": r"\b\+\d{1,3}[ -]?\d{4,14}\b",
        "dni_es": r"\b\d{8}[A-HJ-NP-TV-Z]\b",
        "tarjeta": r"\b(?:\d{4}[ -]?){3}\d{4}\b",
        "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b",
        "ip_completa": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    # Tarifas aproximadas por millón de tokens (USD)
    TARIFAS = {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-opus-4": {"input": 15.00, "output": 75.00},
    }

    def __init__(self, ruta_log: str = "audit_ia.jsonl"):
        self.ruta_log = Path(ruta_log)
        self.ruta_log.parent.mkdir(parents=True, exist_ok=True)

    def _redactar_pii(self, texto: str) -> str:
        """Sustituye PII detectada por etiquetas de redacción."""
        if not isinstance(texto, str):
            return texto
        for tipo, patron in self.PATRONES_PII.items():
            texto = re.sub(patron, f"[REDACTADO:{tipo.upper()}]", texto, flags=re.IGNORECASE)
        return texto

    def _calcular_coste(self, modelo: str, input_tokens: int, output_tokens: int) -> float:
        """Calcula coste estimado en USD."""
        tarifa = self.TARIFAS.get(modelo, {"input": 3.00, "output": 15.00})
        return (input_tokens * tarifa["input"] + output_tokens * tarifa["output"]) / 1_000_000

    def _hash_prompt(self, texto: str) -> str:
        """Genera hash SHA-256 del texto para detectar duplicados sin almacenar contenido."""
        return hashlib.sha256(texto.encode()).hexdigest()[:16]

    def _enmascarar_ip(self, ip: str) -> str:
        """Enmascara el último octeto de una IPv4."""
        partes = ip.split(".")
        if len(partes) == 4:
            return ".".join(partes[:3]) + ".***"
        return "***"

    def registrar(
        self,
        request: dict,
        response: dict,
        metadata: dict,
    ) -> dict:
        """
        Registra una interacción con el modelo de IA.

        Args:
            request: Diccionario con los datos de la petición.
                     Campos esperados: usuario_id, session_id, prompt, modelo, ip
            response: Diccionario con la respuesta del modelo.
                      Campos esperados: content, input_tokens, output_tokens, status
            metadata: Diccionario con metadatos adicionales.
                      Campos opcionales: endpoint, flags_seguridad, error_tipo, latencia_ms

        Returns:
            El registro tal como fue almacenado.
        """
        # Extraer y redactar prompt
        prompt_original = request.get("prompt", "")
        prompt_redactado = self._redactar_pii(prompt_original)

        # Tokens y coste
        modelo = request.get("modelo", "desconocido")
        input_tokens = response.get("input_tokens", 0)
        output_tokens = response.get("output_tokens", 0)
        coste = self._calcular_coste(modelo, input_tokens, output_tokens)

        # Construir registro
        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": str(uuid.uuid4()),
            "usuario_id": request.get("usuario_id", "anonimo"),
            "session_id": request.get("session_id", str(uuid.uuid4())),
            "modelo": modelo,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "coste_estimado": round(coste, 8),
            "latencia_ms": metadata.get("latencia_ms", 0),
            "flags_seguridad": metadata.get("flags_seguridad", []),
            "ip": self._enmascarar_ip(request.get("ip", "0.0.0.0")),
            "endpoint": metadata.get("endpoint", "/api/chat"),
            "status": response.get("status", "ok"),
            "error_tipo": metadata.get("error_tipo"),
            "prompt_hash": self._hash_prompt(prompt_original),
            "prompt_preview": prompt_redactado[:100] + "..." if len(prompt_redactado) > 100 else prompt_redactado,
        }

        # Escribir en JSONL
        with open(self.ruta_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

        return registro


# --- Uso básico ---

if __name__ == "__main__":
    logger = AuditLogger("logs/audit_ia.jsonl")

    registro = logger.registrar(
        request={
            "usuario_id": "usr_abc123",
            "session_id": "sess_xyz789",
            "prompt": "Analiza el contrato de Juan García, DNI 12345678Z, email juan@ejemplo.com",
            "modelo": "claude-haiku-4-5-20251001",
            "ip": "192.168.1.42",
        },
        response={
            "content": "El contrato presenta las siguientes cláusulas...",
            "input_tokens": 250,
            "output_tokens": 120,
            "status": "ok",
        },
        metadata={
            "latencia_ms": 1340,
            "flags_seguridad": [],
            "endpoint": "/api/analizar-contrato",
        },
    )

    print(json.dumps(registro, indent=2, ensure_ascii=False))
```

### 3.2 Verificar la redacción

```python
# Probar que la redacción funciona correctamente
logger = AuditLogger()

casos_prueba = [
    "Llámame al +34 612 345 678",
    "Mi email es usuario@empresa.com",
    "DNI: 87654321A",
    "Tarjeta: 4111 1111 1111 1111",
    "IBAN: ES91 2100 0418 4502 0005 1332",
]

for caso in casos_prueba:
    redactado = logger._redactar_pii(caso)
    print(f"Original : {caso}")
    print(f"Redactado: {redactado}")
    print()
```

---

## 4. Detección de anomalías en uso

Una vez que los logs se acumulan, podemos analizarlos con pandas para detectar patrones sospechosos.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone


def cargar_logs(ruta: str = "logs/audit_ia.jsonl") -> pd.DataFrame:
    """Carga logs JSONL en un DataFrame de pandas."""
    registros = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                registros.append(json.loads(linea))
    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def detectar_anomalias(logs_df: pd.DataFrame) -> dict:
    """
    Analiza un DataFrame de logs y detecta patrones anómalos.

    Returns:
        Diccionario con las anomalías detectadas por categoría.
    """
    anomalias = {
        "volumen_inusual": [],
        "inputs_repetidos": [],
        "horario_fuera_negocio": [],
        "ratio_error_alto": [],
    }

    if logs_df.empty:
        return anomalias

    # --- 1. Volumen inusual por usuario ---
    # Detectar usuarios que superan 3 desviaciones estándar del promedio
    peticiones_por_usuario = logs_df.groupby("usuario_id").size()
    media = peticiones_por_usuario.mean()
    std = peticiones_por_usuario.std()
    umbral_volumen = media + 3 * std

    usuarios_anomalos = peticiones_por_usuario[peticiones_por_usuario > umbral_volumen]
    for usuario, total in usuarios_anomalos.items():
        anomalias["volumen_inusual"].append({
            "usuario_id": usuario,
            "peticiones": int(total),
            "umbral": round(umbral_volumen, 1),
            "gravedad": "alta" if total > umbral_volumen * 2 else "media",
        })

    # --- 2. Inputs repetidos idénticos ---
    # Mismo usuario enviando el mismo prompt hash más de 5 veces
    repeticiones = (
        logs_df.groupby(["usuario_id", "prompt_hash"])
        .size()
        .reset_index(name="repeticiones")
    )
    repeticiones_altas = repeticiones[repeticiones["repeticiones"] > 5]
    for _, fila in repeticiones_altas.iterrows():
        anomalias["inputs_repetidos"].append({
            "usuario_id": fila["usuario_id"],
            "prompt_hash": fila["prompt_hash"],
            "repeticiones": int(fila["repeticiones"]),
            "gravedad": "alta" if fila["repeticiones"] > 20 else "media",
        })

    # --- 3. Horario fuera de negocio (antes de las 7h o después de las 23h, hora local) ---
    logs_df = logs_df.copy()
    logs_df["hora"] = logs_df["timestamp"].dt.hour
    fuera_horario = logs_df[(logs_df["hora"] < 7) | (logs_df["hora"] >= 23)]

    if not fuera_horario.empty:
        por_usuario_horario = fuera_horario.groupby("usuario_id").size().reset_index(name="peticiones")
        for _, fila in por_usuario_horario.iterrows():
            if fila["peticiones"] > 3:  # más de 3 peticiones fuera de horario es sospechoso
                anomalias["horario_fuera_negocio"].append({
                    "usuario_id": fila["usuario_id"],
                    "peticiones_fuera_horario": int(fila["peticiones"]),
                    "gravedad": "baja",
                })

    # --- 4. Ratio de error alto por usuario ---
    if "status" in logs_df.columns:
        total_por_usuario = logs_df.groupby("usuario_id").size()
        errores_por_usuario = (
            logs_df[logs_df["status"].isin(["error", "bloqueado"])]
            .groupby("usuario_id")
            .size()
        )
        ratio_error = (errores_por_usuario / total_por_usuario).dropna()
        usuarios_muchos_errores = ratio_error[ratio_error > 0.3]  # más del 30% de errores

        for usuario, ratio in usuarios_muchos_errores.items():
            anomalias["ratio_error_alto"].append({
                "usuario_id": usuario,
                "ratio_error": round(float(ratio), 3),
                "total_peticiones": int(total_por_usuario[usuario]),
                "gravedad": "alta" if ratio > 0.5 else "media",
            })

    return anomalias


# --- Ejemplo de uso ---
if __name__ == "__main__":
    df = cargar_logs("logs/audit_ia.jsonl")
    anomalias = detectar_anomalias(df)

    total_anomalias = sum(len(v) for v in anomalias.values())
    print(f"Anomalías detectadas: {total_anomalias}")

    for categoria, items in anomalias.items():
        if items:
            print(f"\n[{categoria.upper()}]")
            for item in items:
                print(f"  {item}")
```

---

## 5. Dashboard de auditoría con pandas

Este módulo genera un informe completo de uso a partir de los logs acumulados.

```python
import pandas as pd
import json
from pathlib import Path


def generar_informe_auditoria(
    ruta_logs: str = "logs/audit_ia.jsonl",
    dias: int = 7,
) -> None:
    """
    Genera un informe de auditoría en consola con las métricas principales.

    Args:
        ruta_logs: Ruta al archivo JSONL de logs.
        dias: Número de días hacia atrás a analizar.
    """
    # Cargar logs
    registros = []
    with open(ruta_logs, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                registros.append(json.loads(linea))

    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["fecha"] = df["timestamp"].dt.date

    # Filtrar por período
    fecha_inicio = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=dias)
    df = df[df["timestamp"] >= fecha_inicio]

    if df.empty:
        print("No hay datos para el período especificado.")
        return

    separador = "=" * 60

    print(separador)
    print(f"  INFORME DE AUDITORÍA — Últimos {dias} días")
    print(separador)

    # --- Resumen general ---
    print("\n📊 RESUMEN GENERAL")
    print(f"  Total de peticiones : {len(df):,}")
    print(f"  Usuarios únicos     : {df['usuario_id'].nunique():,}")
    print(f"  Sesiones únicas     : {df['session_id'].nunique():,}")
    print(f"  Coste total (USD)   : ${df['coste_estimado'].sum():.4f}")

    if "status" in df.columns:
        bloqueados = (df["status"] == "bloqueado").sum()
        errores = (df["status"] == "error").sum()
        print(f"  Peticiones OK       : {(df['status'] == 'ok').sum():,}")
        print(f"  Errores             : {errores:,}")
        print(f"  Bloqueadas          : {bloqueados:,}")

    # --- Top usuarios por peticiones ---
    print("\n👥 TOP 10 USUARIOS POR PETICIONES")
    top_usuarios = (
        df.groupby("usuario_id")
        .agg(
            peticiones=("request_id", "count"),
            tokens_input=("input_tokens", "sum"),
            tokens_output=("output_tokens", "sum"),
            coste_total=("coste_estimado", "sum"),
        )
        .sort_values("peticiones", ascending=False)
        .head(10)
    )
    top_usuarios["coste_total"] = top_usuarios["coste_total"].map("${:.4f}".format)
    print(top_usuarios.to_string())

    # --- Prompts más largos (por tokens de input) ---
    print("\n📝 TOP 5 PETICIONES POR TOKENS DE INPUT")
    cols = ["timestamp", "usuario_id", "modelo", "input_tokens", "prompt_hash"]
    cols_disponibles = [c for c in cols if c in df.columns]
    prompts_largos = (
        df[cols_disponibles]
        .sort_values("input_tokens", ascending=False)
        .head(5)
    )
    print(prompts_largos.to_string(index=False))

    # --- Intentos bloqueados ---
    if "status" in df.columns and "bloqueado" in df["status"].values:
        print("\n🚫 INTENTOS BLOQUEADOS")
        bloqueados_df = df[df["status"] == "bloqueado"][
            ["timestamp", "usuario_id", "ip", "flags_seguridad"]
        ].sort_values("timestamp", ascending=False).head(10)
        print(bloqueados_df.to_string(index=False))

    # --- Coste por día ---
    print("\n💰 COSTE POR DÍA (USD)")
    coste_dia = (
        df.groupby("fecha")
        .agg(
            peticiones=("request_id", "count"),
            tokens_totales=("input_tokens", "sum"),
            coste=("coste_estimado", "sum"),
        )
        .sort_index()
    )
    coste_dia["coste"] = coste_dia["coste"].map("${:.4f}".format)
    print(coste_dia.to_string())

    # --- Uso por modelo ---
    print("\n🤖 USO POR MODELO")
    por_modelo = (
        df.groupby("modelo")
        .agg(
            peticiones=("request_id", "count"),
            coste_total=("coste_estimado", "sum"),
        )
        .sort_values("peticiones", ascending=False)
    )
    por_modelo["coste_total"] = por_modelo["coste_total"].map("${:.4f}".format)
    print(por_modelo.to_string())

    print(f"\n{separador}")
    print("  Fin del informe")
    print(separador)


# --- Ejecutar ---
if __name__ == "__main__":
    generar_informe_auditoria("logs/audit_ia.jsonl", dias=30)
```

---

## 6. Alertas automáticas

El monitor corre en un hilo secundario y analiza los logs periódicamente, disparando callbacks cuando se detectan condiciones de alerta.

```python
import threading
import time
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Callable, Optional


class Monitor:
    """
    Monitor de auditoría en tiempo real.
    Analiza logs cada N segundos y dispara alertas cuando se superan umbrales.
    """

    def __init__(
        self,
        ruta_logs: str = "logs/audit_ia.jsonl",
        intervalo_segundos: int = 60,
        callback_alerta: Optional[Callable] = None,
    ):
        """
        Args:
            ruta_logs: Ruta al archivo JSONL de logs.
            intervalo_segundos: Frecuencia de análisis en segundos.
            callback_alerta: Función a llamar cuando se detecta una alerta.
                             Recibe (tipo_alerta: str, datos: dict).
        """
        self.ruta_logs = ruta_logs
        self.intervalo = intervalo_segundos
        self.callback_alerta = callback_alerta or self._alerta_por_defecto
        self._hilo: Optional[threading.Thread] = None
        self._activo = False

        # Umbrales configurables
        self.max_fallos_por_ventana = 5       # N intentos fallidos
        self.ventana_minutos = 10             # en X minutos
        self.max_peticiones_por_minuto = 60  # límite de rate

    @staticmethod
    def _alerta_por_defecto(tipo_alerta: str, datos: dict) -> None:
        """Callback por defecto: imprime la alerta en consola."""
        timestamp = datetime.now(timezone.utc).isoformat()
        print(f"[ALERTA][{timestamp}] {tipo_alerta}: {json.dumps(datos, ensure_ascii=False)}")

    def _leer_logs_recientes(self, ventana_minutos: int) -> list[dict]:
        """Lee los registros de los últimos N minutos del archivo JSONL."""
        registros = []
        limite = datetime.now(timezone.utc) - timedelta(minutes=ventana_minutos)

        try:
            with open(self.ruta_logs, "r", encoding="utf-8") as f:
                for linea in f:
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        registro = json.loads(linea)
                        ts = datetime.fromisoformat(registro.get("timestamp", ""))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts >= limite:
                            registros.append(registro)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except FileNotFoundError:
            pass

        return registros

    def _analizar(self) -> None:
        """Analiza los logs recientes y dispara alertas si es necesario."""
        registros = self._leer_logs_recientes(self.ventana_minutos)

        if not registros:
            return

        # --- Detectar N intentos fallidos en X minutos por usuario ---
        fallos_por_usuario: dict[str, list] = defaultdict(list)
        for registro in registros:
            if registro.get("status") in ("error", "bloqueado"):
                usuario = registro.get("usuario_id", "desconocido")
                fallos_por_usuario[usuario].append(registro.get("timestamp"))

        for usuario, timestamps in fallos_por_usuario.items():
            if len(timestamps) >= self.max_fallos_por_ventana:
                self.callback_alerta("FALLOS_REPETIDOS", {
                    "usuario_id": usuario,
                    "fallos": len(timestamps),
                    "ventana_minutos": self.ventana_minutos,
                    "umbral": self.max_fallos_por_ventana,
                    "primer_fallo": timestamps[0],
                    "ultimo_fallo": timestamps[-1],
                })

        # --- Detectar rate elevado (peticiones por minuto) ---
        peticiones_por_usuario: dict[str, int] = defaultdict(int)
        for registro in registros:
            usuario = registro.get("usuario_id", "desconocido")
            peticiones_por_usuario[usuario] += 1

        tasa_por_minuto = self.max_peticiones_por_minuto * (self.ventana_minutos / 1)
        for usuario, total in peticiones_por_usuario.items():
            if total > tasa_por_minuto:
                self.callback_alerta("RATE_ELEVADO", {
                    "usuario_id": usuario,
                    "peticiones_en_ventana": total,
                    "ventana_minutos": self.ventana_minutos,
                    "limite": tasa_por_minuto,
                })

        # --- Detectar coste acumulado elevado ---
        coste_total = sum(r.get("coste_estimado", 0) for r in registros)
        if coste_total > 10.0:  # más de 10 USD en la ventana
            self.callback_alerta("COSTE_ELEVADO", {
                "coste_usd": round(coste_total, 4),
                "ventana_minutos": self.ventana_minutos,
                "num_peticiones": len(registros),
            })

    def _bucle(self) -> None:
        """Bucle principal del hilo de monitoreo."""
        while self._activo:
            try:
                self._analizar()
            except Exception as e:
                print(f"[Monitor] Error durante análisis: {e}")
            time.sleep(self.intervalo)

    def iniciar(self) -> None:
        """Inicia el monitor en un hilo de fondo."""
        if self._activo:
            return
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, daemon=True, name="AuditMonitor")
        self._hilo.start()
        print(f"[Monitor] Iniciado. Intervalo: {self.intervalo}s. Logs: {self.ruta_logs}")

    def detener(self) -> None:
        """Detiene el monitor."""
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=5)
        print("[Monitor] Detenido.")


# --- Uso con callback personalizado ---

def mi_alerta(tipo: str, datos: dict) -> None:
    """Ejemplo de callback: podría enviar un email, Slack, etc."""
    print(f"🚨 ALERTA [{tipo}]: {datos}")
    # Aquí se integraría con sistemas de notificación:
    # send_slack_message(f"Alerta de seguridad: {tipo}")
    # send_email(destinatario="admin@empresa.com", asunto=tipo, cuerpo=str(datos))


if __name__ == "__main__":
    monitor = Monitor(
        ruta_logs="logs/audit_ia.jsonl",
        intervalo_segundos=60,
        callback_alerta=mi_alerta,
    )

    monitor.iniciar()

    try:
        # El monitor corre en segundo plano
        print("Monitor activo. Presiona Ctrl+C para detener.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.detener()
```

---

## 7. Extensiones sugeridas

Una vez implementada la auditoría básica, estos son los siguientes pasos naturales:

### 7.1 Almacenamiento escalable

Reemplazar el archivo JSONL local por:
- **Elasticsearch + Kibana**: búsqueda y visualización en tiempo real, ideal para grandes volúmenes.
- **BigQuery / Snowflake**: análisis histórico a escala de terabytes.
- **Loki + Grafana**: stack open source orientado a logs.

### 7.2 Alertas multi-canal

Integrar el `Monitor` con:
- **Slack / Teams**: via webhooks para alertas en tiempo real al equipo.
- **PagerDuty / OpsGenie**: para alertas críticas fuera de horario.
- **Email**: para informes periódicos automáticos.

### 7.3 Cumplimiento normativo automatizado

- Implementar retención automática de logs (borrado o anonimización tras N días según GDPR).
- Generar informes de auditoría firmados digitalmente para auditorías externas.
- Integrar con sistemas SIEM (Security Information and Event Management) corporativos.

### 7.4 Mejoras en la detección

- **Machine learning para anomalías**: modelos de detección de outliers (Isolation Forest, LSTM) entrenados sobre los propios logs.
- **Fingerprinting de prompts**: agrupar prompts semánticamente similares para detectar variantes de ataques de inyección.
- **Análisis de secuencias**: detectar patrones de comportamiento sospechoso que se distribuyen en el tiempo.

### 7.5 Integración con el pipeline de IA

Convertir el logger en un **middleware** que se inserta transparentemente entre la aplicación y la API de Anthropic, capturando todos los parámetros sin modificar el código existente.

---

**Fin del bloque.**
