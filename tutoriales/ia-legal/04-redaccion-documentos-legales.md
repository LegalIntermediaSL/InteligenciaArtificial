# Redacción Asistida de Documentos Legales

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/04-redaccion-documentos-legales.ipynb)

Redactar un NDA, una política de privacidad o un contrato de servicios desde cero consume horas de trabajo cualificado. Claude puede generar borradores sólidos en segundos, adaptados a la jurisdicción y al perfil de las partes, y luego incorporar los comentarios del abogado revisor de forma iterativa. Este artículo construye un flujo completo de redacción asistida con revisión humana integrada.

> **Aviso:** Los ejemplos son ilustrativos; no constituyen asesoramiento jurídico.

---

## 1. Casos de uso: tipología de documentos

| Documento | Complejidad | Tiempo tradicional | Tiempo con IA |
|-----------|-------------|-------------------|---------------|
| **NDA** (confidencialidad) | Baja-Media | 1-2h | 5-10 min |
| **Carta de intenciones** (LOI) | Media | 2-4h | 15-30 min |
| **Contrato de servicios** | Media-Alta | 4-8h | 30-60 min |
| **Política de privacidad** | Media | 3-5h | 20-40 min |
| **SLA** (nivel de servicio) | Alta | 4-6h | 30-60 min |
| **Pacto de socios** | Muy alta | 8-16h | 2-4h |

La IA es más eficaz en documentos estándar con variables predecibles. Los documentos muy complejos o con estructuras de propiedad atípicas siguen requiriendo redacción original especializada.

---

## 2. Sistema de plantillas inteligentes

Las plantillas no son texto fijo: incluyen variables, condiciones y adaptaciones de contexto:

```python
from __future__ import annotations
import anthropic
from dataclasses import dataclass, field
from typing import Optional
from datetime import date
from enum import Enum

client = anthropic.Anthropic()

class Jurisdiccion(str, Enum):
    ESPANA = "España"
    MEXICO = "México"
    ARGENTINA = "Argentina"
    COLOMBIA = "Colombia"
    UNION_EUROPEA = "Unión Europea"

@dataclass
class ParteLegal:
    nombre: str
    tipo: str  # "persona_fisica", "sociedad_limitada", "sociedad_anonima"
    cif_nif: Optional[str] = None
    domicilio: Optional[str] = None
    representante: Optional[str] = None
    cargo_representante: Optional[str] = None

@dataclass
class ParametrosNDA:
    parte_divulgante: ParteLegal
    parte_receptora: ParteLegal
    objeto_confidencial: str  # Qué información se protege
    finalidad: str  # Para qué se comparte la información
    duracion_anos: int = 3
    exclusiones: list[str] = field(default_factory=list)
    jurisdiccion: Jurisdiccion = Jurisdiccion.ESPANA
    incluir_penalizacion: bool = False
    penalizacion_importe: Optional[str] = None
    idioma: str = "español"

@dataclass
class ParametrosContrato:
    proveedor: ParteLegal
    cliente: ParteLegal
    descripcion_servicio: str
    importe: str
    forma_pago: str
    fecha_inicio: date
    duracion: str  # "6 meses", "1 año", "indefinido"
    lugar_prestacion: str
    jurisdiccion: Jurisdiccion = Jurisdiccion.ESPANA
    incluir_penalizacion: bool = False
    incluir_propiedad_intelectual: bool = True
    incluir_subcontratacion: bool = False

@dataclass
class ParametrosPoliticaPrivacidad:
    empresa: ParteLegal
    url_web: str
    tipos_datos: list[str]  # ["nombre", "email", "IP", "cookies", ...]
    finalidades: list[str]
    terceros_receptores: list[str]
    transferencias_internacionales: bool
    dpo_email: Optional[str] = None
    jurisdiccion: Jurisdiccion = Jurisdiccion.ESPANA
```

---

## 3. Clase RedactorLegal

```python
class RedactorLegal:
    """Genera y revisa documentos legales adaptados a jurisdicción y perfil."""

    def __init__(self, jurisdiccion_default: Jurisdiccion = Jurisdiccion.ESPANA):
        self.jurisdiccion_default = jurisdiccion_default
        self._historial_versiones: dict[str, list[str]] = {}

    def _instrucciones_jurisdiccion(self, jurisdiccion: Jurisdiccion) -> str:
        """Instrucciones específicas por jurisdicción."""
        instrucciones = {
            Jurisdiccion.ESPANA: (
                "Usa derecho español. Cita el Código Civil (CC), Código de Comercio "
                "y legislación específica vigente. Incluye referencias al Registro Mercantil "
                "si aplica. Foro: Juzgados de lo Mercantil de la ciudad del demandado."
            ),
            Jurisdiccion.MEXICO: (
                "Usa derecho mexicano. Cita el Código Civil Federal, Código de Comercio "
                "y legislación aplicable de la CDMX salvo que se especifique otro estado. "
                "Foro: Tribunales del fuero federal o local según la materia."
            ),
            Jurisdiccion.ARGENTINA: (
                "Usa derecho argentino. Cita el Código Civil y Comercial de la Nación "
                "(Ley 26.994) vigente. Foro: Juzgados Nacionales en lo Comercial de Buenos Aires "
                "salvo acuerdo en contrario."
            ),
            Jurisdiccion.UNION_EUROPEA: (
                "Usa derecho de la Unión Europea. Incluye cláusulas de tratamiento de datos "
                "conforme al RGPD. Permite elección de ley aplicable dentro de la UE."
            ),
        }
        return instrucciones.get(jurisdiccion, "Derecho local aplicable")

    def generar_nda(self, params: ParametrosNDA) -> str:
        """Genera un Acuerdo de Confidencialidad (NDA) completo."""
        exclusiones_str = "\n".join(f"- {e}" for e in params.exclusiones) if params.exclusiones else "- No se establecen exclusiones adicionales"
        penalizacion_str = (
            f"Penalización pactada por incumplimiento: {params.penalizacion_importe}"
            if params.incluir_penalizacion and params.penalizacion_importe
            else "Sin cláusula penal específica"
        )

        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": f"""Redacta un Acuerdo de Confidencialidad (NDA) profesional y completo.

                INSTRUCCIONES DE JURISDICCIÓN:
                {self._instrucciones_jurisdiccion(params.jurisdiccion)}

                PARTES:
                - Parte divulgante: {params.parte_divulgante.nombre} ({params.parte_divulgante.tipo})
                  CIF/NIF: {params.parte_divulgante.cif_nif or 'A completar'}
                  Domicilio: {params.parte_divulgante.domicilio or 'A completar'}
                  Representante: {params.parte_divulgante.representante or 'A completar'}

                - Parte receptora: {params.parte_receptora.nombre} ({params.parte_receptora.tipo})
                  CIF/NIF: {params.parte_receptora.cif_nif or 'A completar'}
                  Domicilio: {params.parte_receptora.domicilio or 'A completar'}

                OBJETO DE CONFIDENCIALIDAD: {params.objeto_confidencial}
                FINALIDAD DE LA DIVULGACIÓN: {params.finalidad}
                DURACIÓN: {params.duracion_anos} años desde la firma
                EXCLUSIONES DE CONFIDENCIALIDAD:
                {exclusiones_str}
                CLÁUSULA PENAL: {penalizacion_str}

                El documento debe incluir: encabezado, partes, expositivos, cláusulas numeradas
                (objeto, definición de información confidencial, obligaciones, exclusiones,
                duración, devolución de información, remedio, ley aplicable y jurisdicción),
                y espacio para firmas con fecha.

                Usa un tono formal y técnico-jurídico propio del {params.jurisdiccion.value}."""
            }]
        )

        texto = respuesta.content[0].text
        id_doc = f"NDA_{params.parte_divulgante.nombre}_{params.parte_receptora.nombre}"
        self._historial_versiones.setdefault(id_doc, []).append(texto)
        return texto

    def generar_politica_privacidad(self, params: ParametrosPoliticaPrivacidad) -> str:
        """Genera una Política de Privacidad conforme a RGPD."""
        tipos_datos_str = ", ".join(params.tipos_datos)
        finalidades_str = "\n".join(f"- {f}" for f in params.finalidades)
        terceros_str = "\n".join(f"- {t}" for t in params.terceros_receptores) if params.terceros_receptores else "- No se ceden datos a terceros salvo obligación legal"

        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=5000,
            messages=[{
                "role": "user",
                "content": f"""Redacta una Política de Privacidad completa y conforme al RGPD/LOPDGDD
                para el siguiente sitio web.

                RESPONSABLE DEL TRATAMIENTO:
                {params.empresa.nombre}
                CIF: {params.empresa.cif_nif or 'A completar'}
                Domicilio: {params.empresa.domicilio or 'A completar'}
                URL: {params.url_web}
                {f'DPO: {params.dpo_email}' if params.dpo_email else 'Sin DPO designado'}

                DATOS TRATADOS: {tipos_datos_str}

                FINALIDADES:
                {finalidades_str}

                CESIÓN A TERCEROS:
                {terceros_str}

                TRANSFERENCIAS INTERNACIONALES: {'Sí — incluir salvaguardas adecuadas' if params.transferencias_internacionales else 'No'}

                Incluye obligatoriamente: identidad del responsable, datos de contacto,
                finalidades y base jurídica, destinatarios, derechos del interesado
                (ARCO+ y cómo ejercerlos), plazo de conservación, decisiones automatizadas
                si aplica, y datos de contacto de la AEPD.

                Redacta en español, tono claro y accesible para ciudadanos."""
            }]
        )

        texto = respuesta.content[0].text
        id_doc = f"PP_{params.empresa.nombre}_{params.url_web}"
        self._historial_versiones.setdefault(id_doc, []).append(texto)
        return texto

    def adaptar_clausula(
        self,
        clausula_original: str,
        instruccion: str,
        contexto_contrato: str = "",
        jurisdiccion: Jurisdiccion | None = None
    ) -> str:
        """Adapta o reescribe una cláusula específica según instrucciones."""
        juris = jurisdiccion or self.jurisdiccion_default
        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Adapta esta cláusula contractual según las instrucciones dadas.
                Mantén el tono formal y técnico-jurídico del {juris.value}.

                CONTEXTO DEL CONTRATO: {contexto_contrato or 'Contrato mercantil genérico'}
                INSTRUCCIÓN DE ADAPTACIÓN: {instruccion}

                CLÁUSULA ORIGINAL:
                {clausula_original}

                Devuelve únicamente la cláusula adaptada, sin explicaciones adicionales."""
            }]
        )
        return respuesta.content[0].text
```

---

## 4. Revisión automática de borrador

Antes de enviar al abogado, Claude realiza una primera revisión técnica:

```python
@dataclass
class HallazgoRevision:
    tipo: str  # "ambiguedad", "contradiccion", "laguna", "error_tecnico", "mejora"
    fragmento: str
    descripcion: str
    sugerencia: str
    gravedad: str  # "alta", "media", "baja"

def revisar_borrador(
    documento: str,
    tipo_documento: str,
    jurisdiccion: Jurisdiccion = Jurisdiccion.ESPANA
) -> list[HallazgoRevision]:
    """
    Revisa un borrador legal detectando ambigüedades, contradicciones y lagunas.
    Primera capa de control de calidad antes de la revisión humana.
    """
    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"""Eres un abogado revisor experto. Revisa este borrador de {tipo_documento}
            conforme al derecho de {jurisdiccion.value}.

            Detecta y clasifica:
            1. **Ambigüedades**: términos o frases con interpretación múltiple
            2. **Contradicciones**: cláusulas que se contradicen entre sí
            3. **Lagunas**: aspectos importantes no regulados que deberían estarlo
            4. **Errores técnicos**: referencias normativas incorrectas, plazos imposibles, etc.
            5. **Mejoras**: oportunidades de fortalecer la posición del cliente

            Devuelve JSON:
            {{
              "hallazgos": [
                {{
                  "tipo": "ambiguedad|contradiccion|laguna|error_tecnico|mejora",
                  "fragmento": "texto exacto del documento donde se detecta",
                  "descripcion": "explicación del problema",
                  "sugerencia": "cómo resolverlo",
                  "gravedad": "alta|media|baja"
                }}
              ]
            }}

            BORRADOR:
            {documento[:15000]}"""
        }]
    )

    datos = json.loads(respuesta.content[0].text)
    return [HallazgoRevision(**h) for h in datos["hallazgos"]]

def imprimir_revision(hallazgos: list[HallazgoRevision]) -> None:
    iconos = {"alta": "🔴", "media": "🟡", "baja": "🟢"}
    tipos_orden = ["contradiccion", "laguna", "error_tecnico", "ambiguedad", "mejora"]

    for tipo in tipos_orden:
        grupo = [h for h in hallazgos if h.tipo == tipo]
        if not grupo:
            continue
        print(f"\n{'='*60}")
        print(f"  {tipo.upper()} ({len(grupo)} hallazgos)")
        print('='*60)
        for h in grupo:
            icono = iconos.get(h.gravedad, "⚪")
            print(f"\n{icono} [{h.gravedad.upper()}] {h.descripcion}")
            print(f"   Fragmento: «{h.fragmento[:100]}...»")
            print(f"   Sugerencia: {h.sugerencia}")
```

---

## 5. Ajuste de tono y jurisdicción

La misma cláusula puede necesitar variantes para diferentes países:

```python
VARIANTES_JURISDICCION = {
    Jurisdiccion.ESPANA: {
        "saludo_formal": "En Madrid, a",
        "tribunal": "Juzgados de lo Mercantil",
        "registro": "Registro Mercantil",
        "ley_datos": "RGPD y LOPDGDD",
    },
    Jurisdiccion.MEXICO: {
        "saludo_formal": "En la Ciudad de México, a",
        "tribunal": "Tribunales Federales",
        "registro": "Registro Público de Comercio",
        "ley_datos": "Ley Federal de Protección de Datos Personales",
    },
    Jurisdiccion.ARGENTINA: {
        "saludo_formal": "En la Ciudad Autónoma de Buenos Aires, a",
        "tribunal": "Juzgados Nacionales en lo Comercial",
        "registro": "Registro Público de Comercio",
        "ley_datos": "Ley 25.326 de Protección de Datos Personales",
    },
}

def adaptar_jurisdiccion(documento: str, origen: Jurisdiccion, destino: Jurisdiccion) -> str:
    """Adapta un documento de una jurisdicción a otra."""
    variantes_origen = VARIANTES_JURISDICCION.get(origen, {})
    variantes_destino = VARIANTES_JURISDICCION.get(destino, {})

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5000,
        messages=[{
            "role": "user",
            "content": f"""Adapta este documento jurídico de {origen.value} a {destino.value}.

            Cambios necesarios:
            - Sustituir referencias normativas por las equivalentes en {destino.value}
            - Adaptar la terminología jurídica al uso local
            - Ajustar las referencias a registros y organismos
            - Mantener la estructura y el sentido jurídico de cada cláusula

            Referencia de adaptaciones clave:
            {json.dumps({'origen': variantes_origen, 'destino': variantes_destino}, ensure_ascii=False, indent=2)}

            DOCUMENTO ORIGINAL ({origen.value}):
            {documento[:12000]}"""
        }]
    )
    return respuesta.content[0].text

import json
```

---

## 6. Flujo de revisión humana: borrador → revisión → incorporación de comentarios

```python
@dataclass
class ComentarioAbogado:
    clausula_referencia: str  # "Cláusula 3", "Párrafo 2 de cláusula 5", etc.
    texto_afectado: str  # Fragmento del borrador al que se refiere
    instruccion: str  # "Cambiar por...", "Eliminar", "Aclarar que...", "Agregar cláusula..."
    prioridad: str = "normal"  # "bloqueante", "normal", "sugerencia"

def incorporar_comentarios_abogado(
    borrador: str,
    comentarios: list[ComentarioAbogado],
    tipo_documento: str,
    jurisdiccion: Jurisdiccion = Jurisdiccion.ESPANA
) -> tuple[str, str]:
    """
    Incorpora los comentarios del abogado al borrador.
    Devuelve (documento_revisado, registro_cambios).
    """
    comentarios_formateados = "\n\n".join([
        f"[{i+1}] Prioridad: {c.prioridad}\n"
        f"Referencia: {c.clausula_referencia}\n"
        f"Texto afectado: «{c.texto_afectado[:200]}»\n"
        f"Instrucción: {c.instruccion}"
        for i, c in enumerate(comentarios)
    ])

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        messages=[{
            "role": "user",
            "content": f"""Incorpora los siguientes comentarios del abogado revisor al borrador
            de {tipo_documento} conforme al derecho de {jurisdiccion.value}.

            Instrucciones generales:
            1. Incorpora TODOS los comentarios, respetando su prioridad
            2. Mantén la coherencia y el estilo formal del documento
            3. No introduzcas cambios no solicitados
            4. Tras el documento revisado, incluye "---REGISTRO DE CAMBIOS---"
               seguido de la lista de modificaciones realizadas

            COMENTARIOS DEL ABOGADO:
            {comentarios_formateados}

            BORRADOR ACTUAL:
            {borrador[:12000]}"""
        }]
    )

    texto_completo = respuesta.content[0].text
    if "---REGISTRO DE CAMBIOS---" in texto_completo:
        partes = texto_completo.split("---REGISTRO DE CAMBIOS---", 1)
        return partes[0].strip(), partes[1].strip()
    return texto_completo, "Registro de cambios no generado"
```

---

## 7. Control de versiones de documentos legales

```python
from datetime import datetime
import hashlib

@dataclass
class VersionDocumento:
    version: str
    fecha: datetime
    autor: str
    descripcion_cambios: str
    hash_contenido: str
    contenido: str

class ControlVersionesLegal:
    """Gestiona versiones de documentos legales con registro de cambios."""

    def __init__(self, id_documento: str):
        self.id_documento = id_documento
        self.versiones: list[VersionDocumento] = []

    def registrar_version(
        self, contenido: str, autor: str, descripcion: str
    ) -> VersionDocumento:
        """Registra una nueva versión del documento."""
        n = len(self.versiones) + 1
        version = f"v{n}.0"
        hash_c = hashlib.sha256(contenido.encode()).hexdigest()[:12]
        v = VersionDocumento(
            version=version,
            fecha=datetime.now(),
            autor=autor,
            descripcion_cambios=descripcion,
            hash_contenido=hash_c,
            contenido=contenido,
        )
        self.versiones.append(v)
        return v

    def diff_versiones(self, v1: str, v2: str) -> str:
        """Genera un diff semántico entre dos versiones numeradas."""
        ver1 = next((v for v in self.versiones if v.version == v1), None)
        ver2 = next((v for v in self.versiones if v.version == v2), None)
        if not ver1 or not ver2:
            return "Versión no encontrada"

        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""Genera un changelog legal entre estas dos versiones del documento.
                Lista solo los cambios materiales (no de formato).
                Formato: "Cláusula X: [descripción del cambio]"

                VERSIÓN {v1} ({ver1.fecha.strftime('%d/%m/%Y')}):
                {ver1.contenido[:6000]}

                VERSIÓN {v2} ({ver2.fecha.strftime('%d/%m/%Y')}):
                {ver2.contenido[:6000]}"""
            }]
        )
        return respuesta.content[0].text

    def historial(self) -> str:
        """Devuelve el historial de versiones en formato tabla."""
        lineas = [
            f"# Historial de versiones — {self.id_documento}\n",
            "| Versión | Fecha | Autor | Descripción | Hash |",
            "|---------|-------|-------|-------------|------|",
        ]
        for v in self.versiones:
            lineas.append(
                f"| {v.version} | {v.fecha.strftime('%d/%m/%Y %H:%M')} "
                f"| {v.autor} | {v.descripcion_cambios[:50]} | `{v.hash_contenido}` |"
            )
        return "\n".join(lineas)
```

---

## Resumen del flujo completo

```
1. Definir parámetros (partes, objeto, jurisdicción)
        ↓
2. RedactorLegal.generar_nda() / generar_politica_privacidad()
        ↓
3. revisar_borrador() → lista de hallazgos automáticos
        ↓
4. Abogado revisa hallazgos + añade ComentarioAbogado
        ↓
5. incorporar_comentarios_abogado() → v2 del documento
        ↓
6. ControlVersionesLegal.registrar_version() → historial git-like
        ↓
7. Firma final (fuera del scope de la IA)
```

Este flujo reduce el tiempo de redacción en un 60-80% en documentos estándar, liberando al abogado para el trabajo de mayor valor añadido: negociación, estrategia y validación final.

---

*Anterior: [Cumplimiento Normativo ←](03-cumplimiento-normativo.md) | [Volver al índice del Bloque 32 →](README.md)*
