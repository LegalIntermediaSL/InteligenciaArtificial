# Monitorización de Cumplimiento Normativo con IA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/03-cumplimiento-normativo.ipynb)

Una empresa mediana opera bajo decenas de marcos normativos: protección de datos, seguridad alimentaria, servicios financieros, ciberseguridad, inteligencia artificial... La legislación cambia constantemente y una modificación de un reglamento puede generar nuevas obligaciones en días. Este artículo construye un sistema de monitorización que detecta automáticamente qué cambios normativos afectan al perfil específico de cada empresa.

> **Aviso:** Los ejemplos son ilustrativos; no constituyen asesoramiento jurídico.

---

## 1. El reto: legislación dinámica y fragmentada

| Problema | Impacto | Solución con IA |
|----------|---------|-----------------|
| Volumen de normas | Una empresa puede estar sujeta a 20+ marcos regulatorios | Clasificación automática por perfil de empresa |
| Velocidad de cambios | Modificaciones frecuentes en GDPR, AI Act, NIS2... | Detección de diffs entre versiones de normas |
| Interpretación | Textos legales densos y ambiguos | Síntesis ejecutiva con ejemplos prácticos |
| Dispersión | Normas en BOE, DOUE, reguladores sectoriales | Ingestión desde múltiples fuentes |
| Trazabilidad | Demostrar que se ha cumplido la diligencia debida | Registro de análisis con timestamp y fuente |

El modelo `claude-sonnet-4-6` tiene capacidad para leer y razonar sobre textos legales complejos manteniendo consistencia a lo largo de documentos extensos.

---

## 2. Clase MonitorNormativo

```python
from __future__ import annotations
import anthropic
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

client = anthropic.Anthropic()

class NivelObligacion(str, Enum):
    OBLIGATORIO = "obligatorio"
    RECOMENDADO = "recomendado"
    INFORMATIVO = "informativo"

class EstadoCompliance(str, Enum):
    CUMPLE = "cumple"
    CUMPLE_PARCIAL = "cumple_parcialmente"
    NO_CUMPLE = "no_cumple"
    NO_APLICA = "no_aplica"
    SIN_EVALUAR = "sin_evaluar"

@dataclass
class PerfilEmpresa:
    """Describe el perfil de la empresa para filtrar obligaciones aplicables."""
    nombre: str
    sector: str  # "tecnologia", "salud", "finanzas", "retail", etc.
    paises_operacion: list[str]
    empleados: int
    procesa_datos_personales: bool
    tiene_ia_en_produccion: bool
    es_operador_servicio_esencial: bool = False
    descripcion_actividad: str = ""

@dataclass
class ObligacionNormativa:
    norma: str
    articulo: str
    descripcion: str
    nivel: NivelObligacion
    plazo_cumplimiento: Optional[str]
    aplica_a_perfil: bool
    justificacion_aplicabilidad: str

@dataclass
class ResultadoAnalisisNorma:
    norma: str
    version: str
    fecha_analisis: datetime
    perfil_empresa: str
    obligaciones_aplicables: list[ObligacionNormativa]
    resumen_ejecutivo: str
    acciones_prioritarias: list[str]

class MonitorNormativo:
    """Analiza textos legales y detecta obligaciones aplicables a un perfil de empresa."""

    def __init__(self, perfil: PerfilEmpresa):
        self.perfil = perfil
        self.historial: list[ResultadoAnalisisNorma] = []

    def analizar_norma(self, texto_norma: str, nombre_norma: str, version: str = "vigente") -> ResultadoAnalisisNorma:
        """Analiza una norma y extrae las obligaciones aplicables al perfil."""
        perfil_json = json.dumps({
            "nombre": self.perfil.nombre,
            "sector": self.perfil.sector,
            "paises": self.perfil.paises_operacion,
            "empleados": self.perfil.empleados,
            "procesa_datos_personales": self.perfil.procesa_datos_personales,
            "tiene_ia_produccion": self.perfil.tiene_ia_en_produccion,
            "operador_servicio_esencial": self.perfil.es_operador_servicio_esencial,
            "actividad": self.perfil.descripcion_actividad,
        }, ensure_ascii=False)

        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""Eres un experto en compliance y regulación empresarial.
                Analiza este texto normativo e identifica todas las obligaciones que aplican
                a la empresa descrita.

                PERFIL DE EMPRESA:
                {perfil_json}

                NORMA: {nombre_norma} ({version})

                Para cada obligación:
                1. Evalúa si aplica a esta empresa específica
                2. Indica el artículo de referencia
                3. Clasifica como obligatorio/recomendado/informativo
                4. Indica el plazo si está especificado

                Devuelve JSON con esta estructura:
                {{
                  "obligaciones": [
                    {{
                      "norma": "{nombre_norma}",
                      "articulo": "Art. X",
                      "descripcion": "descripción de la obligación",
                      "nivel": "obligatorio|recomendado|informativo",
                      "plazo_cumplimiento": "fecha o descripción o null",
                      "aplica_a_perfil": true|false,
                      "justificacion_aplicabilidad": "por qué sí/no aplica a esta empresa"
                    }}
                  ],
                  "resumen_ejecutivo": "resumen en 3-4 frases para dirección",
                  "acciones_prioritarias": ["acción 1", "acción 2", ...]
                }}

                TEXTO NORMATIVO:
                {texto_norma[:15000]}"""
            }]
        )

        datos = json.loads(respuesta.content[0].text)
        obligaciones = [ObligacionNormativa(**o) for o in datos["obligaciones"]]
        resultado = ResultadoAnalisisNorma(
            norma=nombre_norma,
            version=version,
            fecha_analisis=datetime.now(),
            perfil_empresa=self.perfil.nombre,
            obligaciones_aplicables=[o for o in obligaciones if o.aplica_a_perfil],
            resumen_ejecutivo=datos["resumen_ejecutivo"],
            acciones_prioritarias=datos["acciones_prioritarias"],
        )
        self.historial.append(resultado)
        return resultado
```

---

## 3. Checklist de compliance automatizado

Dado un documento interno (política, procedimiento, contrato), verifica si cumple una norma:

```python
@dataclass
class ItemChecklist:
    requisito: str
    articulo_referencia: str
    estado: EstadoCompliance
    evidencia: str  # Fragmento del documento que lo demuestra
    gap: str  # Qué falta si no cumple
    prioridad: int  # 1-3

def evaluar_compliance_documento(
    documento_interno: str,
    nombre_documento: str,
    norma_referencia: str,
    texto_norma: str | None = None,
    requisitos_clave: list[str] | None = None,
) -> list[ItemChecklist]:
    """
    Evalúa si un documento interno cumple con una norma.
    Puede usar un texto normativo completo o una lista de requisitos conocidos.
    """
    if requisitos_clave:
        contexto_norma = f"Requisitos a verificar:\n" + "\n".join(f"- {r}" for r in requisitos_clave)
    elif texto_norma:
        contexto_norma = f"Texto normativo de referencia:\n{texto_norma[:6000]}"
    else:
        contexto_norma = f"Norma de referencia: {norma_referencia} (usa tu conocimiento)"

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"""Evalúa si el documento interno cumple con {norma_referencia}.
            Para cada requisito relevante, indica el estado de cumplimiento.

            {contexto_norma}

            DOCUMENTO A EVALUAR ({nombre_documento}):
            {documento_interno[:8000]}

            Devuelve JSON:
            {{
              "checklist": [
                {{
                  "requisito": "descripción del requisito normativo",
                  "articulo_referencia": "Art. X de la norma",
                  "estado": "cumple|cumple_parcialmente|no_cumple|no_aplica",
                  "evidencia": "fragmento del documento que acredita (o vacío si no cumple)",
                  "gap": "qué falta o debe mejorarse (vacío si cumple)",
                  "prioridad": 1-3
                }}
              ]
            }}"""
        }]
    )

    datos = json.loads(respuesta.content[0].text)
    return [ItemChecklist(**item) for item in datos["checklist"]]

def imprimir_checklist(checklist: list[ItemChecklist]) -> None:
    """Imprime el checklist en formato tabla."""
    iconos = {
        EstadoCompliance.CUMPLE: "✅",
        EstadoCompliance.CUMPLE_PARCIAL: "⚠️",
        EstadoCompliance.NO_CUMPLE: "❌",
        EstadoCompliance.NO_APLICA: "➖",
        EstadoCompliance.SIN_EVALUAR: "❓",
    }
    print(f"{'Estado':<5} {'P':<3} {'Requisito':<50} {'Gap'}")
    print("-" * 100)
    for item in sorted(checklist, key=lambda x: x.prioridad):
        icono = iconos.get(item.estado, "?")
        gap = item.gap[:40] + "..." if len(item.gap) > 40 else item.gap
        req = item.requisito[:48] + ".." if len(item.requisito) > 48 else item.requisito
        print(f"{icono:<5} {item.prioridad:<3} {req:<50} {gap}")
```

---

## 4. Alertas de cambios normativos: versión nueva vs. antigua

```python
@dataclass
class CambioNormativo:
    articulo: str
    tipo_cambio: str  # 'nuevo', 'modificado', 'derogado'
    descripcion: str
    impacto_empresa: str  # 'alto', 'medio', 'bajo', 'sin_impacto'
    accion_requerida: str
    plazo: Optional[str]

def detectar_cambios_normativos(
    norma_antigua: str,
    norma_nueva: str,
    nombre_norma: str,
    perfil: PerfilEmpresa,
) -> list[CambioNormativo]:
    """
    Compara dos versiones de una norma y extrae los cambios relevantes
    para el perfil de empresa especificado.
    """
    perfil_resumen = (
        f"{perfil.nombre} — sector {perfil.sector}, "
        f"{perfil.empleados} empleados, "
        f"{'procesa datos personales' if perfil.procesa_datos_personales else 'no procesa datos personales'}"
    )

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"""Compara estas dos versiones de {nombre_norma} e identifica
            los cambios relevantes para la empresa: {perfil_resumen}

            Céntrate en:
            1. Artículos nuevos o modificados que generen nuevas obligaciones
            2. Cambios en plazos o umbrales
            3. Nuevas sanciones o su modificación
            4. Derogaciones que eliminen obligaciones previas

            Ignora cambios puramente formales (numeración, erratas, formato).

            Devuelve JSON:
            {{
              "cambios": [
                {{
                  "articulo": "Art. X",
                  "tipo_cambio": "nuevo|modificado|derogado",
                  "descripcion": "qué ha cambiado exactamente",
                  "impacto_empresa": "alto|medio|bajo|sin_impacto",
                  "accion_requerida": "qué debe hacer la empresa",
                  "plazo": "plazo de adaptación si se especifica o null"
                }}
              ]
            }}

            VERSIÓN ANTERIOR:
            {norma_antigua[:6000]}

            VERSIÓN NUEVA:
            {norma_nueva[:6000]}"""
        }]
    )

    datos = json.loads(respuesta.content[0].text)
    return [CambioNormativo(**c) for c in datos["cambios"]]
```

---

## 5. Aplicación a GDPR, EU AI Act y Ley de Servicios Digitales

Ejemplos concretos de obligaciones que el monitor detectaría para diferentes perfiles:

```python
# Configuración de normas principales con sus requisitos clave conocidos
NORMAS_PRINCIPALES = {
    "GDPR": {
        "nombre_oficial": "Reglamento (UE) 2016/679",
        "requisitos_clave": [
            "Designación de DPO si se tratan datos a gran escala o datos sensibles",
            "Registro de actividades de tratamiento (Art. 30)",
            "Evaluación de impacto (EIPD) para tratamientos de alto riesgo (Art. 35)",
            "Procedimiento de notificación de brechas en 72h (Art. 33)",
            "Atención de derechos ARCO+ en 30 días (Art. 12-22)",
            "Base jurídica documentada para cada tratamiento (Art. 6)",
        ]
    },
    "EU_AI_ACT": {
        "nombre_oficial": "Reglamento (UE) 2024/1689",
        "requisitos_clave": [
            "Clasificación de sistemas de IA por nivel de riesgo",
            "Prohibición de sistemas de IA de riesgo inaceptable (Art. 5)",
            "Evaluación de conformidad para IA de alto riesgo (Art. 43)",
            "Registro en base de datos UE para IA de alto riesgo (Art. 71)",
            "Documentación técnica y logs de sistemas de alto riesgo (Art. 12-13)",
            "Supervisión humana obligatoria en sistemas de alto riesgo (Art. 14)",
            "Transparencia hacia usuarios de sistemas de IA (Art. 50)",
        ]
    },
    "LSD": {
        "nombre_oficial": "Digital Services Act — Reglamento (UE) 2022/2065",
        "requisitos_clave": [
            "Mecanismos de notificación y acción para contenidos ilegales",
            "Transparencia sobre sistemas de recomendación algorítmica",
            "Informe anual de transparencia para plataformas intermedias",
            "Condiciones de servicio claras y accesibles",
            "Protección de menores en plataformas online",
            "Obligaciones reforzadas para plataformas muy grandes (VLOPs)",
        ]
    }
}

def evaluar_perfil_contra_normas(perfil: PerfilEmpresa) -> dict[str, list[str]]:
    """Determina qué normas aplican y qué acciones prioritarias requieren."""
    monitor = MonitorNormativo(perfil)
    acciones_por_norma = {}

    for clave, norma_info in NORMAS_PRINCIPALES.items():
        # Evaluar con requisitos conocidos (sin necesitar el texto completo)
        checklist = evaluar_compliance_documento(
            documento_interno=f"Empresa: {perfil.descripcion_actividad}",
            nombre_documento="Perfil de empresa",
            norma_referencia=norma_info["nombre_oficial"],
            requisitos_clave=norma_info["requisitos_clave"],
        )
        gaps = [item.gap for item in checklist if item.estado in (
            EstadoCompliance.NO_CUMPLE, EstadoCompliance.CUMPLE_PARCIAL
        ) and item.gap]
        if gaps:
            acciones_por_norma[clave] = gaps

    return acciones_por_norma
```

---

## 6. Pipeline de monitorización semanal

```python
import hashlib
from pathlib import Path

def pipeline_monitorización_semanal(
    perfil: PerfilEmpresa,
    directorio_normas: str,
    archivo_estado: str = "estado_normas.json"
) -> dict:
    """
    Ejecuta el pipeline semanal de monitorización normativa.
    Detecta cambios en los archivos de normas y genera el resumen de novedades.
    """
    ruta_estado = Path(archivo_estado)
    estado_anterior = {}
    if ruta_estado.exists():
        estado_anterior = json.loads(ruta_estado.read_text())

    monitor = MonitorNormativo(perfil)
    novedades = []

    for ruta_norma in Path(directorio_normas).glob("*.txt"):
        contenido = ruta_norma.read_text(encoding="utf-8")
        hash_actual = hashlib.sha256(contenido.encode()).hexdigest()
        hash_previo = estado_anterior.get(ruta_norma.name, {}).get("hash")

        if hash_actual != hash_previo:
            print(f"Cambio detectado en: {ruta_norma.name}")
            resultado = monitor.analizar_norma(
                contenido,
                nombre_norma=ruta_norma.stem,
                version=f"actualización-{date.today()}"
            )
            novedades.append({
                "norma": ruta_norma.name,
                "acciones": resultado.acciones_prioritarias,
                "resumen": resultado.resumen_ejecutivo,
            })
            estado_anterior[ruta_norma.name] = {"hash": hash_actual, "ultima_revision": str(date.today())}

    # Persistir el nuevo estado
    ruta_estado.write_text(json.dumps(estado_anterior, ensure_ascii=False, indent=2))

    # Generar resumen consolidado de novedades
    if novedades:
        resumen = generar_resumen_novedades(novedades, perfil)
    else:
        resumen = f"Sin novedades normativas relevantes para {perfil.nombre} esta semana."

    return {"novedades": novedades, "resumen": resumen, "fecha": str(date.today())}

def generar_resumen_novedades(novedades: list[dict], perfil: PerfilEmpresa) -> str:
    """Genera un resumen ejecutivo de todas las novedades normativas de la semana."""
    contexto = json.dumps(novedades, ensure_ascii=False, indent=2)
    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Redacta el resumen semanal de novedades normativas para {perfil.nombre}
            (sector: {perfil.sector}). Sé conciso y práctico. Incluye:
            1. Qué normas han cambiado y en qué afectan a la empresa
            2. Acciones que debe tomar (ordenadas por urgencia)
            3. Plazos si los hay

            NOVEDADES DETECTADAS:
            {contexto}"""
        }]
    )
    return respuesta.content[0].text
```

---

## 7. Integración con Slack y email para alertas al equipo legal

```python
import urllib.request

def enviar_alerta_slack(
    webhook_url: str,
    resumen_novedades: str,
    perfil: PerfilEmpresa,
    nivel_urgencia: str = "normal"  # "normal", "urgente", "informativo"
) -> bool:
    """Envía el resumen de novedades normativas al canal de Slack del equipo legal."""
    iconos = {"urgente": "🚨", "normal": "⚖️", "informativo": "ℹ️"}
    icono = iconos.get(nivel_urgencia, "⚖️")

    payload = {
        "text": f"{icono} *Novedades Normativas — {perfil.nombre}* ({date.today()})",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{icono} Novedades Normativas — {date.today()}"
                }
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": resumen_novedades[:3000]}
            },
            {
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"_Generado automáticamente para {perfil.nombre} | Requiere revisión del equipo legal_"
                }]
            }
        ]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Error enviando a Slack: {e}")
        return False

def enviar_alerta_email_compliance(
    destinatarios: list[str],
    resumen: str,
    novedades: list[dict],
    perfil: PerfilEmpresa,
    smtp_config: dict | None = None,
) -> None:
    """Envía el boletín semanal de compliance por email."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    config = smtp_config or {"host": "localhost", "port": 25, "from": "compliance@empresa.com"}

    cuerpo = f"""# Boletín Semanal de Compliance — {date.today()}

**Empresa:** {perfil.nombre}
**Sector:** {perfil.sector}

## Resumen Ejecutivo

{resumen}

## Normas Actualizadas

"""
    for nov in novedades:
        cuerpo += f"\n### {nov['norma']}\n{nov['resumen']}\n"
        if nov.get("acciones"):
            cuerpo += "\n**Acciones requeridas:**\n"
            cuerpo += "\n".join(f"- {a}" for a in nov["acciones"])
        cuerpo += "\n"

    cuerpo += "\n---\n*Generado automáticamente. Requiere revisión del equipo legal antes de cualquier acción.*"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Compliance] Novedades Normativas — {date.today()}"
    msg["From"] = config["from"]
    msg["To"] = ", ".join(destinatarios)
    msg.attach(MIMEText(cuerpo, "plain", "utf-8"))

    with smtplib.SMTP(config["host"], config.get("port", 25)) as servidor:
        if config.get("tls"):
            servidor.starttls()
        if config.get("user"):
            servidor.login(config["user"], config["password"])
        servidor.sendmail(config["from"], destinatarios, msg.as_string())
        print(f"Alerta enviada a {len(destinatarios)} destinatarios.")
```

---

## Resumen

En este artículo hemos construido un sistema completo de monitorización normativa que:

1. Modela el **perfil de la empresa** para filtrar obligaciones aplicables
2. Analiza textos legales y extrae obligaciones con la clase `MonitorNormativo`
3. Evalúa documentos internos contra **checklists de compliance** con estados semáforo
4. Detecta **cambios materiales** entre versiones de una norma
5. Aplica el sistema a **GDPR, EU AI Act y LSD** con requisitos clave predefinidos
6. Ejecuta un **pipeline semanal** con detección de cambios por hash
7. Distribuye alertas por **Slack y email** al equipo legal

---

*Anterior: [Due Diligence ←](02-due-diligence-ia.md) | Siguiente: [Redacción de Documentos Legales →](04-redaccion-documentos-legales.md)*
