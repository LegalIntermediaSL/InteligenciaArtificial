# Due Diligence Asistida por IA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/02-due-diligence-ia.ipynb)

La due diligence es el proceso de investigación exhaustiva que precede a una inversión, adquisición o alianza estratégica. Un proceso tradicional puede durar semanas y movilizar a decenas de personas. Con IA, podemos acelerar la revisión documental, detectar red flags automáticamente y responder preguntas sobre el dataroom en segundos.

> **Aviso:** Los ejemplos son ilustrativos; no constituyen asesoramiento jurídico.

---

## 1. Qué es due diligence y qué documentos incluye

La due diligence cubre múltiples áreas de una empresa objetivo. Cada área tiene sus propios documentos y sus propias señales de alerta:

| Área | Documentos típicos | Red flags principales |
|------|-------------------|----------------------|
| **Legal** | Estatutos, contratos mercantiles, litigios, propiedad intelectual | Litigios activos, cláusulas de cambio de control, PI no registrada |
| **Financiero** | Balances, P&L, flujo de caja, auditorías, deudas | Deuda oculta, contingencias no provisionadas, manipulación contable |
| **Laboral** | Contratos de empleados, convenio aplicable, sanciones | Contratos irregulares, deudas a Seguridad Social, EREs pendientes |
| **Técnico** | Arquitectura de sistemas, licencias de software, código fuente | Dependencias obsoletas, licencias GPL contaminantes, deuda técnica |
| **Fiscal** | Declaraciones tributarias, acuerdos con AEAT, inspecciones | Inspecciones abiertas, diferencias con criterios AEAT, operaciones vinculadas |
| **Regulatorio** | Licencias, permisos, comunicaciones con reguladores | Expedientes sancionadores, licencias caducadas |

---

## 2. Clase DDProcessor: índice del dataroom

```python
from __future__ import annotations
import anthropic
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

client = anthropic.Anthropic()

class AreaDD(str, Enum):
    LEGAL = "legal"
    FINANCIERO = "financiero"
    LABORAL = "laboral"
    TECNICO = "tecnico"
    FISCAL = "fiscal"
    REGULATORIO = "regulatorio"
    OTRO = "otro"

@dataclass
class DocumentoIndexado:
    nombre: str
    ruta: str
    area: AreaDD
    resumen: str
    fecha_documento: str | None
    red_flags: list[str]
    relevancia: int  # 1-5

@dataclass
class DataroomIndex:
    empresa: str
    total_documentos: int
    documentos: list[DocumentoIndexado] = field(default_factory=list)
    red_flags_globales: list[str] = field(default_factory=list)

class DDProcessor:
    """Procesa un directorio de documentos y construye un índice de dataroom."""

    def __init__(self, empresa: str):
        self.empresa = empresa
        self.index = DataroomIndex(empresa=empresa, total_documentos=0)
        self._textos: dict[str, str] = {}  # Para RAG posterior

    def _clasificar_documento(self, nombre: str, texto_preview: str) -> AreaDD:
        """Clasifica el documento en un área de DD."""
        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"""Clasifica este documento en una de estas áreas de due diligence:
                legal, financiero, laboral, tecnico, fiscal, regulatorio, otro.

                Nombre del archivo: {nombre}
                Primeras líneas: {texto_preview[:500]}

                Responde ÚNICAMENTE con una de las palabras clave del área."""
            }]
        )
        area_str = respuesta.content[0].text.strip().lower()
        try:
            return AreaDD(area_str)
        except ValueError:
            return AreaDD.OTRO

    def _indexar_documento(self, ruta: Path) -> DocumentoIndexado:
        """Lee y analiza un documento individual."""
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        # Guardar para RAG
        self._textos[ruta.name] = texto

        area = self._clasificar_documento(ruta.name, texto)

        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Analiza este documento en el contexto de una due diligence.
                Empresa objetivo: {self.empresa}
                Área: {area.value}

                Devuelve JSON con:
                {{
                  "resumen": "descripción del documento en 2-3 frases",
                  "fecha_documento": "YYYY-MM-DD o null",
                  "red_flags": ["lista de alertas o riesgos detectados"],
                  "relevancia": 1-5
                }}

                DOCUMENTO ({ruta.name}):
                {texto[:8000]}"""
            }]
        )

        datos = json.loads(respuesta.content[0].text)
        return DocumentoIndexado(
            nombre=ruta.name,
            ruta=str(ruta),
            area=area,
            resumen=datos["resumen"],
            fecha_documento=datos.get("fecha_documento"),
            red_flags=datos.get("red_flags", []),
            relevancia=datos.get("relevancia", 3),
        )

    def procesar_directorio(self, directorio: str, extensiones: list[str] | None = None) -> DataroomIndex:
        """Procesa todos los documentos del directorio."""
        extensiones = extensiones or [".txt", ".md", ".pdf"]
        directorio_path = Path(directorio)

        archivos = [
            f for f in directorio_path.rglob("*")
            if f.suffix.lower() in extensiones
        ]

        print(f"Procesando {len(archivos)} documentos del dataroom de {self.empresa}...")

        for archivo in archivos:
            try:
                doc = self._indexar_documento(archivo)
                self.index.documentos.append(doc)
                if doc.red_flags:
                    print(f"  ⚠ {archivo.name}: {len(doc.red_flags)} red flags")
            except Exception as e:
                print(f"  Error procesando {archivo.name}: {e}")

        self.index.total_documentos = len(self.index.documentos)
        self._consolidar_red_flags_globales()
        return self.index

    def _consolidar_red_flags_globales(self) -> None:
        """Consolida y prioriza los red flags de todos los documentos."""
        todos_flags = []
        for doc in self.index.documentos:
            todos_flags.extend([f"[{doc.area.value}] {flag}" for flag in doc.red_flags])

        if not todos_flags:
            return

        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Consolida y prioriza estos red flags de due diligence.
                Elimina duplicados, agrupa los relacionados y ordena de mayor a menor impacto.
                Devuelve una lista JSON de strings (máximo 10 items).

                RED FLAGS DETECTADOS:
                {json.dumps(todos_flags, ensure_ascii=False)}"""
            }]
        )
        self.index.red_flags_globales = json.loads(respuesta.content[0].text)
```

---

## 3. Extracción de red flags específicos

Algunos red flags requieren búsqueda activa y razonamiento especializado:

```python
RED_FLAGS_CRITICOS = {
    "cambio_de_control": "Cláusulas que dan derecho a resolver o modificar condiciones en caso de cambio accionarial",
    "litigios_activos": "Procedimientos judiciales o arbitrales en curso o amenazados",
    "deuda_oculta": "Obligaciones financieras no reflejadas en el balance o fuera de hoja",
    "ip_contaminada": "Propiedad intelectual con licencias restrictivas (GPL) o titularidad disputada",
    "clausulas_competencia": "Pactos de no competencia, exclusividad o restricciones post-cierre",
}

def detectar_red_flags_criticos(texto: str, empresa_compradora: str) -> dict[str, list[str]]:
    """Busca activamente los red flags más críticos en un documento."""
    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Analiza este documento buscando específicamente los siguientes
            tipos de red flags para una operación de adquisición.
            Empresa compradora: {empresa_compradora}

            TIPOS A DETECTAR:
            {json.dumps(RED_FLAGS_CRITICOS, ensure_ascii=False, indent=2)}

            Para cada tipo, indica si hay evidencia (con cita textual) o ausencia confirmada.
            Devuelve JSON:
            {{
              "tipo_red_flag": ["cita o evidencia encontrada", ...] o []
            }}

            DOCUMENTO:
            {texto[:12000]}"""
        }]
    )
    return json.loads(respuesta.content[0].text)
```

---

## 4. Resumen ejecutivo por área

Cada área requiere una página de resumen para la dirección:

```python
def generar_resumen_area(
    index: DataroomIndex,
    area: AreaDD,
    destinatario: str = "Comité de inversión"
) -> str:
    """Genera un resumen ejecutivo de una página para el área especificada."""
    docs_area = [d for d in index.documentos if d.area == area]

    if not docs_area:
        return f"## {area.value.upper()}\n\n*Sin documentos disponibles para esta área.*"

    contexto = "\n\n".join([
        f"**{d.nombre}** (relevancia {d.relevancia}/5)\n{d.resumen}\nRed flags: {', '.join(d.red_flags) or 'ninguno'}"
        for d in sorted(docs_area, key=lambda x: x.relevancia, reverse=True)
    ])

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Redacta un resumen ejecutivo de UNA PÁGINA sobre el área {area.value}
            de la due diligence de {index.empresa} para {destinatario}.

            Estructura:
            1. Situación general (2-3 frases)
            2. Hallazgos principales (bullets)
            3. Red flags y riesgos (bullets priorizados)
            4. Recomendaciones (bullets)
            5. Veredicto: FAVORABLE / CONDICIONADO / DESFAVORABLE con justificación

            Usa tono profesional y directo. Evita jerga técnica innecesaria.

            DOCUMENTOS ANALIZADOS:
            {contexto}"""
        }]
    )
    return f"## {area.value.upper()}\n\n{respuesta.content[0].text}"
```

---

## 5. Q&A sobre el dataroom con RAG y citaciones

La Citations API de Claude permite responder preguntas indicando exactamente de qué documento proviene cada afirmación:

```python
def qa_dataroom(
    procesador: DDProcessor,
    pregunta: str,
    max_documentos: int = 5
) -> dict:
    """
    Responde preguntas sobre el dataroom con citas a documentos fuente.
    Usa la Citations API de Claude para trazabilidad completa.
    """
    # Recuperar los documentos más relevantes (búsqueda por keywords simplificada)
    palabras_clave = set(pregunta.lower().split())
    docs_puntuados = []
    for nombre, texto in procesador._textos.items():
        texto_lower = texto.lower()
        puntuacion = sum(1 for p in palabras_clave if p in texto_lower)
        if puntuacion > 0:
            docs_puntuados.append((puntuacion, nombre, texto))

    docs_puntuados.sort(reverse=True)
    top_docs = docs_puntuados[:max_documentos]

    if not top_docs:
        return {"respuesta": "No se encontraron documentos relevantes.", "citas": []}

    # Construir el contenido con fuentes para Citations API
    fuentes = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": texto[:6000]},
            "title": nombre,
            "citations": {"enabled": True}
        }
        for _, nombre, texto in top_docs
    ]

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": fuentes + [{
                "type": "text",
                "text": f"""Responde esta pregunta sobre el dataroom de {procesador.empresa}
                basándote ÚNICAMENTE en los documentos proporcionados.
                Cita siempre el documento fuente de cada afirmación.

                PREGUNTA: {pregunta}"""
            }]
        }]
    )

    # Extraer citas de la respuesta
    texto_respuesta = ""
    citas = []
    for bloque in respuesta.content:
        if hasattr(bloque, "text"):
            texto_respuesta += bloque.text
        if hasattr(bloque, "type") and bloque.type == "citations":
            citas.extend(bloque.citations if hasattr(bloque, "citations") else [])

    return {
        "pregunta": pregunta,
        "respuesta": texto_respuesta,
        "documentos_consultados": [n for _, n, _ in top_docs],
        "citas": citas
    }
```

---

## 6. Informe DD exportado a Markdown con tabla de semáforos

```python
SEMAFORO = {1: "🟢", 2: "🟡", 3: "🔴"}

def calcular_semaforo_area(docs: list[DocumentoIndexado]) -> int:
    """1=verde, 2=amarillo, 3=rojo según red flags y relevancia."""
    if not docs:
        return 1
    total_flags = sum(len(d.red_flags) for d in docs)
    flags_criticos = sum(1 for d in docs for f in d.red_flags
                        if any(k in f.lower() for k in ["litigio", "fraude", "irregularidad", "sanción"]))
    if flags_criticos > 0 or total_flags > 5:
        return 3
    if total_flags > 2:
        return 2
    return 1

def exportar_informe_dd(index: DataroomIndex, empresa_compradora: str) -> str:
    """Genera el informe completo de DD en Markdown."""
    from datetime import date
    hoy = date.today().strftime("%d/%m/%Y")

    lineas = [
        f"# Informe de Due Diligence — {index.empresa}",
        f"\n**Empresa compradora:** {empresa_compradora}",
        f"**Fecha:** {hoy}",
        f"**Documentos revisados:** {index.total_documentos}",
        "\n---\n",
        "## Resumen Ejecutivo — Tabla de Semáforos\n",
        "| Área | Estado | Red Flags | Documentos |",
        "|------|--------|-----------|------------|",
    ]

    resúmenes_por_area = {}
    for area in AreaDD:
        if area == AreaDD.OTRO:
            continue
        docs = [d for d in index.documentos if d.area == area]
        semaforo = calcular_semaforo_area(docs)
        total_flags = sum(len(d.red_flags) for d in docs)
        icono = SEMAFORO[semaforo]
        resúmenes_por_area[area] = (semaforo, docs)
        lineas.append(f"| {area.value.capitalize()} | {icono} | {total_flags} | {len(docs)} |")

    if index.red_flags_globales:
        lineas.extend([
            "\n## Red Flags Prioritarios\n",
            *[f"- {flag}" for flag in index.red_flags_globales],
        ])

    lineas.append("\n---\n")
    lineas.append("## Detalle por Área\n")

    for area, (semaforo, docs) in resúmenes_por_area.items():
        if not docs:
            continue
        icono = SEMAFORO[semaforo]
        lineas.extend([
            f"### {area.value.capitalize()} {icono}\n",
            *[f"**{d.nombre}** — {d.resumen}" for d in docs[:5]],
        ])
        flags_area = [f for d in docs for f in d.red_flags]
        if flags_area:
            lineas.append("\n**Alertas:**")
            lineas.extend([f"- {f}" for f in flags_area[:5]])
        lineas.append("")

    return "\n".join(lineas)
```

---

## 7. Limitaciones y responsabilidad del abogado revisor

La IA acelera la due diligence pero no la reemplaza. Es fundamental entender sus límites:

**Lo que la IA hace bien:**
- Lectura rápida de grandes volúmenes de documentos
- Detección de patrones y red flags conocidos
- Síntesis y estructuración de información
- Generación de listas de verificación y preguntas adicionales

**Lo que la IA NO puede hacer:**
- Verificar la autenticidad de documentos (firma, sello, registro)
- Contrastar información con registros externos (Registro Mercantil, catastro, AEAT)
- Evaluar el contexto de negocio y el sector específico
- Asumir responsabilidad profesional por sus conclusiones
- Detectar documentos que faltan en el dataroom

**Protocolo recomendado:**
1. IA genera el índice y el primer borrador de red flags
2. Abogado senior revisa y prioriza los hallazgos
3. IA genera listas de preguntas adicionales para el vendedor
4. Abogado valida con fuentes externas (registros, bases de datos jurídicas)
5. IA redacta el informe final a partir de las notas del abogado
6. Abogado firma y asume responsabilidad del informe definitivo

---

*Anterior: [Análisis de Contratos ←](01-analisis-contratos.md) | Siguiente: [Cumplimiento Normativo →](03-cumplimiento-normativo.md)*
