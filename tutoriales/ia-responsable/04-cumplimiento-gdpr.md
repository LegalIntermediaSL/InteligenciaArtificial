# 04 — Cumplimiento GDPR y Regulación de IA

> **Bloque:** IA Responsable · **Nivel:** Intermedio · **Tiempo estimado:** 50 min

---

## Índice

1. [GDPR y sistemas de IA: artículos clave](#1-gdpr-y-sistemas-de-ia-artículos-clave)
2. [EU AI Act: categorías de riesgo](#2-eu-ai-act-categorías-de-riesgo)
3. [Minimización de datos en pipelines de ML](#3-minimización-de-datos-en-pipelines-de-ml)
4. [Anonimización y pseudonimización](#4-anonimización-y-pseudonimización)
5. [Derecho al olvido y machine unlearning](#5-derecho-al-olvido-y-machine-unlearning)
6. [Checklist de cumplimiento con código](#6-checklist-de-cumplimiento-con-código)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. GDPR y sistemas de IA: artículos clave

El GDPR (Reglamento General de Protección de Datos) no fue diseñado específicamente para IA, pero varios de sus artículos tienen implicaciones directas sobre cómo entrenar, desplegar y operar sistemas de aprendizaje automático.

### Artículos de mayor impacto

**Art. 5 — Principios de tratamiento:**
- **Minimización de datos**: recopilar solo los datos estrictamente necesarios
- **Limitación de la finalidad**: no usar datos para fines distintos al declarado
- **Exactitud**: los datos deben ser correctos y actualizados

**Art. 13/14 — Transparencia:**
- El sistema de IA debe ser explicable al usuario afectado
- Se debe informar de la lógica de las decisiones automatizadas significativas

**Art. 22 — Decisiones automatizadas:**
> El interesado tiene derecho a no ser objeto de decisiones basadas **únicamente** en tratamiento automatizado que produzcan efectos jurídicos significativos.

Esto afecta directamente a sistemas de IA que:
- Conceden o deniegan créditos, seguros, contratos
- Toman decisiones de contratación o evaluación de empleados
- Realizan perfilado con consecuencias legales

**Mitigación para Art. 22:**
- Implementar revisión humana significativa (no solo rubber-stamp)
- Documentar y poder explicar cada decisión individual
- Ofrecer mecanismo de reclamación y revisión

**Art. 25 — Privacy by design:**
- La privacidad debe estar integrada en el diseño, no añadida a posteriori
- Aplicado a ML: anonimizar antes de entrenar, no guardar datos de inferencia innecesarios

---

## 2. EU AI Act: categorías de riesgo

El **EU AI Act** (en vigor desde agosto 2024) clasifica los sistemas de IA en cuatro niveles de riesgo:

```python
# Categorías del EU AI Act
categorias = {
    "inaceptable": {
        "descripcion": "Prohibidos — riesgo inaceptable para derechos fundamentales",
        "ejemplos": [
            "Scoring social por gobiernos",
            "Manipulación subliminal de comportamiento",
            "Identificación biométrica en tiempo real en espacios públicos (salvo excepciones)",
            "Sistemas de predicción de emociones en trabajo/educación"
        ],
        "requisitos": "PROHIBIDOS"
    },
    "alto_riesgo": {
        "descripcion": "Permitidos con requisitos estrictos de conformidad",
        "ejemplos": [
            "IA en infraestructuras críticas (energía, agua, transporte)",
            "IA en educación (evaluación de estudiantes)",
            "IA en empleo (contratación, evaluación de rendimiento)",
            "IA en servicios esenciales (scoring crediticio, seguros, servicios públicos)",
            "IA en justicia y administración pública",
            "IA para gestión de migraciones y fronteras"
        ],
        "requisitos": [
            "Sistema de gestión de riesgos documentado",
            "Gobernanza de datos de entrenamiento",
            "Documentación técnica y logs de auditoría",
            "Transparencia e información al usuario",
            "Supervisión humana",
            "Robustez, seguridad y precisión",
            "Registro en base de datos EU (para sistemas de organismos públicos)"
        ]
    },
    "riesgo_limitado": {
        "descripcion": "Requisitos de transparencia",
        "ejemplos": [
            "Chatbots (deben identificarse como IA)",
            "Deepfakes (deben etiquetarse)",
            "Sistemas de recomendación"
        ],
        "requisitos": ["Transparencia hacia el usuario"]
    },
    "riesgo_minimo": {
        "descripcion": "Sin requisitos específicos",
        "ejemplos": [
            "Filtros de spam",
            "IA en videojuegos",
            "Clasificadores de imágenes sin impacto en derechos"
        ],
        "requisitos": "Ninguno obligatorio"
    }
}

for categoria, info in categorias.items():
    print(f"\n{'='*50}")
    print(f"CATEGORÍA: {categoria.upper()}")
    print(f"Descripción: {info['descripcion']}")
    print(f"Ejemplos: {', '.join(info['ejemplos'][:2])}...")
```

### Obligaciones para sistemas de alto riesgo

```python
# checklist_eu_ai_act.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class EUAIActCompliance:
    sistema: str
    categoria_riesgo: str
    fecha_evaluacion: datetime = field(default_factory=datetime.now)

    # Artículo 9 — Sistema de gestión de riesgos
    sistema_gestion_riesgos: bool = False
    riesgos_identificados: list[str] = field(default_factory=list)
    medidas_mitigacion: list[str] = field(default_factory=list)

    # Artículo 10 — Gobernanza de datos
    practicas_datos_documentadas: bool = False
    datos_entrenamiento_evaluados_sesgos: bool = False
    cobertura_geografica_demografica: bool = False

    # Artículo 11 — Documentación técnica
    documentacion_tecnica_completa: bool = False
    arquitectura_documentada: bool = False
    limitaciones_documentadas: bool = False

    # Artículo 13 — Transparencia
    instrucciones_uso_disponibles: bool = False
    capacidades_limitaciones_comunicadas: bool = False

    # Artículo 14 — Supervisión humana
    mecanismos_supervision_humana: bool = False
    capacidad_anulacion_humana: bool = False

    # Artículo 15 — Precisión y robustez
    metricas_rendimiento_documentadas: bool = False
    pruebas_adversariales_realizadas: bool = False

    def puntuacion(self) -> float:
        campos_bool = [
            self.sistema_gestion_riesgos,
            self.practicas_datos_documentadas,
            self.datos_entrenamiento_evaluados_sesgos,
            self.documentacion_tecnica_completa,
            self.instrucciones_uso_disponibles,
            self.mecanismos_supervision_humana,
            self.capacidad_anulacion_humana,
            self.metricas_rendimiento_documentadas,
        ]
        return sum(campos_bool) / len(campos_bool)

    def informe(self) -> str:
        score = self.puntuacion()
        estado = "CONFORME" if score >= 0.85 else "EN PROGRESO" if score >= 0.5 else "NO CONFORME"
        return (
            f"Sistema: {self.sistema}\n"
            f"Categoría: {self.categoria_riesgo}\n"
            f"Puntuación: {score:.0%}\n"
            f"Estado: {estado}\n"
            f"Fecha: {self.fecha_evaluacion.strftime('%Y-%m-%d')}"
        )


# Ejemplo de uso
evaluacion = EUAIActCompliance(
    sistema="SistemaDecisionCrediticia-v3",
    categoria_riesgo="alto_riesgo",
    sistema_gestion_riesgos=True,
    practicas_datos_documentadas=True,
    datos_entrenamiento_evaluados_sesgos=True,
    documentacion_tecnica_completa=True,
    instrucciones_uso_disponibles=True,
    mecanismos_supervision_humana=True,
    capacidad_anulacion_humana=False,  # pendiente
    metricas_rendimiento_documentadas=True,
)
print(evaluacion.informe())
```

---

## 3. Minimización de datos en pipelines de ML

El principio de minimización implica recopilar solo lo necesario. Aplicado a ML:

```python
# data_minimization.py
import pandas as pd
from typing import Optional

class DataMinimizer:
    """
    Filtra un DataFrame para retener solo los campos
    justificados para la finalidad declarada del modelo.
    """

    def __init__(self, finalidad: str, campos_necesarios: list[str]):
        self.finalidad = finalidad
        self.campos_necesarios = campos_necesarios
        self.log_justificaciones = {}

    def registrar_justificacion(self, campo: str, justificacion: str):
        """Documenta por qué se necesita cada campo (requerimiento GDPR)."""
        self.log_justificaciones[campo] = justificacion

    def minimizar(self, df: pd.DataFrame) -> pd.DataFrame:
        campos_a_eliminar = [c for c in df.columns if c not in self.campos_necesarios]
        if campos_a_eliminar:
            print(f"Campos eliminados por minimización: {campos_a_eliminar}")

        # Verificar que todos los campos retenidos tienen justificación
        sin_justificacion = [
            c for c in self.campos_necesarios
            if c not in self.log_justificaciones
        ]
        if sin_justificacion:
            print(f"ADVERTENCIA: Campos sin justificación documentada: {sin_justificacion}")

        return df[self.campos_necesarios]

    def generar_registro(self) -> dict:
        return {
            "finalidad": self.finalidad,
            "campos_retenidos": self.campos_necesarios,
            "justificaciones": self.log_justificaciones
        }


# Ejemplo: dataset de scoring crediticio
minimizer = DataMinimizer(
    finalidad="Modelo de scoring crediticio para préstamos personales",
    campos_necesarios=["historial_pagos", "ratio_deuda_ingresos", "antiguedad_cuenta", "label"]
)

minimizer.registrar_justificacion("historial_pagos", "Predictor principal de riesgo de impago (Art. 5.1.b)")
minimizer.registrar_justificacion("ratio_deuda_ingresos", "Indicador estándar de capacidad de pago")
minimizer.registrar_justificacion("antiguedad_cuenta", "Proxy de estabilidad financiera")
minimizer.registrar_justificacion("label", "Variable objetivo — impago en los 12 meses siguientes")

# Campos que NO se retienen aunque estén disponibles:
# nombre, dni, dirección, código postal exacto, email, teléfono, etc.

import json
print(json.dumps(minimizer.generar_registro(), ensure_ascii=False, indent=2))
```

---

## 4. Anonimización y pseudonimización

```python
# anonymization.py
import hashlib
import re
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def anonimizar_texto(texto: str, idioma: str = "es") -> str:
    """Detecta y reemplaza PII en texto libre usando Presidio."""
    resultados = analyzer.analyze(text=texto, language=idioma)
    anonimizado = anonymizer.anonymize(
        text=texto,
        analyzer_results=resultados,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSONA>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEFONO>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LUGAR>"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "<FECHA>"}),
            "NRP": OperatorConfig("replace", {"new_value": "<DNI>"}),  # DNI/NIE
        }
    )
    return anonimizado.text


def pseudonimizar_id(valor: str, salt: str = "secreto_interno") -> str:
    """
    Pseudonimización: reemplaza identificadores por hash determinista.
    Con el salt se puede revertir (≠ anonimización irreversible).
    """
    return hashlib.sha256(f"{salt}:{valor}".encode()).hexdigest()[:16]


def anonimizar_dataframe(
    df: pd.DataFrame,
    columnas_texto: list[str],
    columnas_id: list[str],
    columnas_eliminar: list[str],
    salt: str = "secreto_interno"
) -> pd.DataFrame:
    """Pipeline completo de anonimización para un DataFrame."""
    df = df.copy()

    # Eliminar columnas directamente identificativas
    df.drop(columns=columnas_eliminar, errors="ignore", inplace=True)

    # Pseudonimizar IDs
    for col in columnas_id:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: pseudonimizar_id(x, salt)
            )

    # Anonimizar texto libre
    for col in columnas_texto:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(anonimizar_texto)

    return df


# Ejemplo
datos = pd.DataFrame({
    "cliente_id": ["C001", "C002", "C003"],
    "nombre": ["Juan García López", "María Rodríguez", "Pedro Martínez"],
    "email": ["juan@example.com", "maria@example.com", "pedro@example.com"],
    "descripcion_incidencia": [
        "El cliente Juan García llamó el 15 de marzo desde Madrid",
        "María Rodríguez con DNI 12345678A solicita devolución",
        "Pedro contactó desde su email pedro@example.com"
    ],
    "importe": [1500.00, 2300.50, 890.00],
    "categoria_riesgo": ["bajo", "medio", "bajo"]
})

datos_anonimizados = anonimizar_dataframe(
    df=datos,
    columnas_texto=["descripcion_incidencia"],
    columnas_id=["cliente_id"],
    columnas_eliminar=["nombre", "email"]  # eliminación directa
)
print(datos_anonimizados.to_string())
```

---

## 5. Derecho al olvido y machine unlearning

El Art. 17 GDPR reconoce el derecho a la supresión. Para modelos de ML esto plantea el problema del **machine unlearning**: cómo "olvidar" que una persona existía en el training set.

### Aproximaciones prácticas

```python
# machine_unlearning.py — Enfoque práctico para modelos de clasificación
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

class UnlearningManager:
    """
    Gestiona el derecho al olvido para modelos de ML.
    Implementa reentrenamiento completo (enfoque más seguro) y 
    actualización del registro de datos usados en entrenamiento.
    """

    def __init__(self, model_path: str, data_path: str, id_column: str):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.id_column = id_column
        self.solicitudes_olvido: list[str] = []

    def registrar_solicitud_olvido(self, user_id: str, motivo: str = "Art. 17 GDPR"):
        """Registra una solicitud de derecho al olvido."""
        self.solicitudes_olvido.append(user_id)
        print(f"Solicitud registrada: {user_id} — {motivo}")
        # En producción: guardar en DB con timestamp y referencia legal

    def ejecutar_olvido(self, features: list[str], target: str) -> dict:
        """
        Reentrenamiento completo excluyendo los datos del usuario.
        Es el método más robusto — alternativas como gradient unlearning
        son más rápidas pero con garantías más débiles.
        """
        if not self.solicitudes_olvido:
            return {"status": "sin_solicitudes_pendientes"}

        # Cargar datos
        df = pd.read_csv(self.data_path)
        n_original = len(df)

        # Eliminar registros de usuarios que solicitaron olvido
        mask = ~df[self.id_column].isin(self.solicitudes_olvido)
        df_limpio = df[mask]
        n_eliminados = n_original - len(df_limpio)

        if n_eliminados == 0:
            return {"status": "usuarios_no_encontrados", "solicitados": self.solicitudes_olvido}

        # Guardar datos actualizados (sin los registros eliminados)
        df_limpio.to_csv(self.data_path, index=False)

        # Reentrenar modelo
        X = df_limpio[features]
        y = df_limpio[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)

        # Guardar modelo reentrenado
        joblib.dump({"model": model, "scaler": scaler}, self.model_path)

        resultado = {
            "status": "completado",
            "registros_eliminados": n_eliminados,
            "usuarios_olvidados": self.solicitudes_olvido,
            "registros_restantes": len(df_limpio),
            "modelo_reentrenado": True
        }
        self.solicitudes_olvido.clear()
        return resultado


# Ejemplo de uso
manager = UnlearningManager(
    model_path="./modelo_scoring.pkl",
    data_path="./datos_entrenamiento.csv",
    id_column="cliente_id_pseudonimizado"
)

manager.registrar_solicitud_olvido("a3f7c2b1e9d4", "Solicitud GDPR Art. 17 — ref #2024-0315")
manager.registrar_solicitud_olvido("b8d2e4a6f1c3", "Solicitud GDPR Art. 17 — ref #2024-0318")

resultado = manager.ejecutar_olvido(
    features=["historial_pagos", "ratio_deuda_ingresos", "antiguedad_cuenta"],
    target="impago_12m"
)
print(resultado)
```

> **Nota:** El reentrenamiento completo garantiza que el modelo no contiene trazas del individuo, pero es costoso. Para LLMs grandes, las técnicas de gradient-based unlearning (SISA, gradient ascent) son activamente investigadas pero aún no tienen garantías formales suficientes para uso en producción con requisitos legales estrictos.

---

## 6. Checklist de cumplimiento con código

```python
# compliance_checker.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Estado(Enum):
    CONFORME = "✅"
    PARCIAL = "⚠️"
    NO_CONFORME = "❌"
    NO_APLICA = "—"


@dataclass
class ItemCompliance:
    id: str
    articulo: str
    descripcion: str
    estado: Estado = Estado.NO_CONFORME
    notas: str = ""
    responsable: str = ""
    fecha_revision: datetime = field(default_factory=datetime.now)


class GDPRComplianceChecker:
    def __init__(self, sistema: str):
        self.sistema = sistema
        self.items: list[ItemCompliance] = self._init_checklist()

    def _init_checklist(self) -> list[ItemCompliance]:
        return [
            ItemCompliance("G01", "Art. 5.1.b", "Base legal documentada para el tratamiento"),
            ItemCompliance("G02", "Art. 5.1.c", "Minimización de datos implementada"),
            ItemCompliance("G03", "Art. 13/14", "Información al usuario sobre decisiones automatizadas"),
            ItemCompliance("G04", "Art. 17", "Mecanismo de derecho al olvido implementado"),
            ItemCompliance("G05", "Art. 22", "Revisión humana disponible para decisiones significativas"),
            ItemCompliance("G06", "Art. 25", "Privacy by design en el pipeline de datos"),
            ItemCompliance("G07", "Art. 32", "Medidas técnicas de seguridad (cifrado, acceso)"),
            ItemCompliance("G08", "Art. 35", "EIPD realizada si alto riesgo"),
            ItemCompliance("A01", "EU AI Act Art. 9", "Sistema de gestión de riesgos documentado"),
            ItemCompliance("A02", "EU AI Act Art. 10", "Gobernanza de datos de entrenamiento"),
            ItemCompliance("A03", "EU AI Act Art. 11", "Documentación técnica completa"),
            ItemCompliance("A04", "EU AI Act Art. 14", "Supervisión humana implementada"),
            ItemCompliance("A05", "EU AI Act Art. 15", "Pruebas de robustez y precisión realizadas"),
        ]

    def actualizar(self, id: str, estado: Estado, notas: str = "", responsable: str = ""):
        for item in self.items:
            if item.id == id:
                item.estado = estado
                item.notas = notas
                item.responsable = responsable
                item.fecha_revision = datetime.now()
                return
        raise ValueError(f"Item {id} no encontrado")

    def informe(self):
        conformes = sum(1 for i in self.items if i.estado == Estado.CONFORME)
        total = len(self.items)
        print(f"\n{'='*60}")
        print(f"INFORME DE CUMPLIMIENTO: {self.sistema}")
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Puntuación: {conformes}/{total} ({conformes/total:.0%})")
        print(f"{'='*60}")
        for item in self.items:
            print(f"{item.estado.value} [{item.id}] {item.articulo}: {item.descripcion}")
            if item.notas:
                print(f"   → {item.notas}")
        print(f"{'='*60}\n")


# Uso
checker = GDPRComplianceChecker("SistemaDecisionCrediticia-v3")
checker.actualizar("G01", Estado.CONFORME, "Base legal: interés legítimo documentado en EIPD")
checker.actualizar("G02", Estado.CONFORME, "DataMinimizer implementado — 4 campos retenidos de 23")
checker.actualizar("G04", Estado.CONFORME, "UnlearningManager en producción — SLA 30 días")
checker.actualizar("G05", Estado.PARCIAL, "Revisión humana disponible pero no obligatoria aún")
checker.actualizar("G08", Estado.CONFORME, "EIPD realizada — ref DPO-2024-007")
checker.actualizar("A01", Estado.CONFORME, "Risk register actualizado trimestralmente")
checker.actualizar("A04", Estado.PARCIAL, "Supervisión humana implementada — falta botón de anulación")
checker.informe()
```

---

## 7. Extensiones sugeridas

- **EIPD automatizada**: generar el borrador de la Evaluación de Impacto en la Protección de Datos desde el model card
- **Consentimiento granular**: rastrear qué datos de cada usuario se usaron en qué versión del modelo
- **Audit trail inmutable**: logging en blockchain o servicio de timestamping para evidencia legal
- **Integración con DPA**: automatizar notificaciones a la Agencia Española de Protección de Datos para incidentes

---

**Anterior:** [03 — Model cards](./03-model-cards-datasheets.md) · **Siguiente bloque:** [Bloque 16 — MLOps](../mlops/)
