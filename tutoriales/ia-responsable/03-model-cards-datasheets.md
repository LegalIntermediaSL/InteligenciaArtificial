# 03 — Model Cards y Documentación Responsable

> **Bloque:** IA Responsable · **Nivel:** Intermedio · **Tiempo estimado:** 45 min

---

## Índice

1. [Qué son los model cards](#1-qué-son-los-model-cards)
2. [Estructura de un model card](#2-estructura-de-un-model-card)
3. [Datasheets for Datasets](#3-datasheets-for-datasets)
4. [model-card-toolkit de Google](#4-model-card-toolkit-de-google)
5. [Ejemplo completo de model card](#5-ejemplo-completo-de-model-card)
6. [Integración con HuggingFace Model Hub](#6-integración-con-huggingface-model-hub)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Qué son los model cards

Los **model cards** son documentos estandarizados que acompañan a un modelo de machine learning y describen su propósito, capacidades, limitaciones y consideraciones éticas. El término fue acuñado por Mitchell et al. en el paper *"Model Cards for Model Reporting"* (2019, Google).

La idea central es que un modelo de IA no puede evaluarse en abstracto: su comportamiento depende del contexto de uso. Un clasificador de imágenes médicas que funciona bien en población adulta puede ser inadecuado para pacientes pediátricos. Sin documentación explícita, estos problemas son invisibles para quien despliega el modelo.

### Por qué son necesarios

- **Transparencia**: permiten a terceros entender qué hace el modelo antes de usarlo
- **Responsabilidad**: documentan quién tomó qué decisiones de diseño y por qué
- **Auditoría**: facilitan el cumplimiento regulatorio (EU AI Act, GDPR)
- **Mantenimiento**: ayudan al equipo que hereda el modelo a entender su historia
- **Confianza**: los usuarios finales pueden tomar decisiones informadas

---

## 2. Estructura de un model card

Un model card estándar cubre estas secciones:

### 2.1 Información básica

```markdown
# Model Card: ClasificadorContratos-v2

**Desarrollado por:** Equipo de IA Legal — LegalTech SL  
**Fecha de entrenamiento:** 2024-03-15  
**Versión:** 2.1.0  
**Tipo de modelo:** Clasificador de texto (DistilBERT fine-tuneado)  
**Licencia:** Apache 2.0  
**Repositorio:** https://github.com/org/clasificador-contratos
```

### 2.2 Uso previsto (Intended Use)

```markdown
## Uso previsto

### Casos de uso primarios
- Clasificación automática de cláusulas contractuales en 12 categorías
- Prelabeling para revisión humana en flujos de due diligence
- Filtrado inicial de contratos por tipo para asignación a departamentos

### Usuarios objetivo
- Abogados y paralegales con acceso a la plataforma interna
- Sistemas automatizados de ingesta documental

### Casos de uso fuera de alcance
- Decisiones legales autónomas sin supervisión humana
- Procesamiento de contratos en idiomas distintos al español
- Contratos de más de 50,000 tokens (contexto máximo del modelo)
- Uso en jurisdicciones fuera de España y México
```

### 2.3 Factores y condiciones de rendimiento

```markdown
## Factores de rendimiento

### Variables relevantes
- **Tipo de contrato**: el modelo fue entrenado principalmente con contratos mercantiles;
  el rendimiento baja ~8% en contratos laborales y ~15% en contratos administrativos
- **Longitud del texto**: para fragmentos <50 palabras la precisión cae del 94% al 87%
- **Año del documento**: contratos anteriores a 2010 usan terminología diferente;
  se recomienda validación manual para documentos históricos

### Grupos de evaluación
El modelo fue evaluado separadamente en:
- Contratos de compraventa (n=1,200)
- Contratos de arrendamiento (n=800)
- Contratos de servicios (n=950)
- Contratos de confidencialidad (n=600)
```

### 2.4 Métricas de rendimiento

```markdown
## Métricas

| Conjunto | Precisión | Recall | F1 |
|---|---|---|---|
| Test global | 0.941 | 0.938 | 0.939 |
| Compraventa | 0.962 | 0.958 | 0.960 |
| Arrendamiento | 0.933 | 0.929 | 0.931 |
| Servicios | 0.928 | 0.921 | 0.924 |
| Confidencialidad | 0.955 | 0.951 | 0.953 |

**Umbral de decisión:** 0.75 (ajustado para minimizar falsos negativos en cláusulas de responsabilidad)
```

### 2.5 Consideraciones éticas

```markdown
## Consideraciones éticas

### Sesgos conocidos
- El modelo muestra menor rendimiento (F1 -0.04) en contratos con partes identificadas
  como PYMEs vs grandes empresas, probablemente por subrepresentación en el training set
- Las cláusulas en formato de tabla tienen F1 0.12 menor que las cláusulas en prosa

### Riesgos potenciales
- **Falsos negativos en cláusulas críticas**: una cláusula de limitación de responsabilidad
  no detectada podría tener consecuencias legales. El sistema incluye una bandera de
  "baja confianza" para revisión manual cuando score < 0.80
- **No es consejo legal**: el modelo clasifica texto, no interpreta su validez jurídica

### Mitigaciones implementadas
- Revisión humana obligatoria para documentos con score promedio < 0.85
- Auditoría mensual de muestras de producción por el equipo legal
- Panel de monitorización de distribución de predicciones para detectar drift
```

---

## 3. Datasheets for Datasets

El paper *"Datasheets for Datasets"* (Gebru et al., 2021) propone documentar también los datos de entrenamiento, no solo el modelo. Las preguntas clave son:

```python
# Plantilla de datasheet como diccionario Python (para serializar a JSON/YAML)
datasheet = {
    "motivacion": {
        "proposito": "Entrenar un clasificador de cláusulas contractuales en español",
        "creado_por": "Equipo de IA Legal — LegalTech SL",
        "financiado_por": "Proyecto interno I+D 2023",
        "uso_no_previsto": "Análisis de sentimiento, tareas fuera del dominio legal"
    },
    "composicion": {
        "instancias": "47,832 fragmentos de texto de contratos",
        "tipos_datos": "Texto en español extraído de PDFs de contratos",
        "etiquetas": 12,
        "anonimizacion": "Nombres, fechas y montos reemplazados con placeholders",
        "datos_faltantes": "8.3% de fragmentos sin etiqueta de confianza alta (excluidos del entrenamiento)",
        "division": {"train": 0.70, "val": 0.15, "test": 0.15}
    },
    "proceso_recoleccion": {
        "fuentes": "Contratos anonimizados de clientes (con consentimiento), contratos públicos BOE",
        "periodo": "2015-2023",
        "protocolo_anotacion": "3 anotadores + resolución por mayoría + revisión experto legal",
        "acuerdo_anotadores": "Cohen's kappa = 0.87"
    },
    "preprocesamiento": {
        "pasos": [
            "Extracción de texto con pdfplumber",
            "Segmentación en cláusulas por expresiones regulares",
            "Limpieza de caracteres especiales de OCR",
            "Anonimización con Presidio",
            "Tokenización con DistilBERT tokenizer (max_length=512)"
        ],
        "codigo_disponible": True,
        "datos_crudos_disponibles": False
    },
    "consideraciones_eticas": {
        "datos_personales": "No (anonimizados antes de almacenamiento)",
        "consentimiento": "Contratos de clientes: cláusula de anonimización en contrato de servicio",
        "riesgos": "Riesgo bajo — no hay información personal identificable en el dataset"
    },
    "mantenimiento": {
        "propietario": "data-team@legaltech.com",
        "actualizaciones": "Trimestral",
        "politica_errores": "Issue tracker interno — SLA 5 días hábiles"
    }
}

import json
print(json.dumps(datasheet, ensure_ascii=False, indent=2))
```

---

## 4. model-card-toolkit de Google

Google publicó una librería Python para crear model cards de forma programática y exportarlos a HTML o JSON.

```bash
pip install model-card-toolkit
```

```python
import model_card_toolkit as mct
from model_card_toolkit import ModelCard

# Crear toolkit
toolkit = mct.ModelCardToolkit(
    output_dir="./model_card_output"
)

# Inicializar model card
model_card = toolkit.scaffold_assets()

# Rellenar información del modelo
model_card.model_details.name = "ClasificadorContratos-v2"
model_card.model_details.overview = (
    "Clasificador de cláusulas contractuales en español basado en DistilBERT. "
    "Clasifica fragmentos de texto en 12 categorías de cláusulas contractuales."
)
model_card.model_details.version.name = "2.1.0"
model_card.model_details.version.date = "2024-03-15"

# Owners
from model_card_toolkit.proto import model_card_pb2 as mc_pb
owner = mc_pb.Owner(name="Equipo IA Legal", contact="ia@legaltech.com")
model_card.model_details.owners.append(owner)

# Consideraciones
model_card.considerations.use_cases.append(
    mc_pb.UseCase(description="Clasificación automática de cláusulas para prelabeling")
)
model_card.considerations.limitations.append(
    mc_pb.Limitation(
        description="Rendimiento reducido en contratos pre-2010 y en idiomas distintos al español"
    )
)
model_card.considerations.ethical_considerations.append(
    mc_pb.Risk(
        name="Falsos negativos en cláusulas críticas",
        mitigation_strategy="Revisión humana obligatoria para score < 0.80"
    )
)

# Métricas
perf = mc_pb.PerformanceMetric(
    type="F1 Score",
    value="0.939",
    slice="Test global (n=7,174)"
)
model_card.quantitative_analysis.performance_metrics.append(perf)

# Exportar
toolkit.update_model_card(model_card)

# Genera HTML y JSON en output_dir
html_path = toolkit.export_format(model_card=model_card, template_path=None)
print(f"Model card generado en: {html_path}")
```

---

## 5. Ejemplo completo de model card

```markdown
---
# Model Card: ClasificadorContratos-v2

## Información del modelo

| Campo | Valor |
|---|---|
| Nombre | ClasificadorContratos-v2 |
| Versión | 2.1.0 |
| Tipo | Clasificación de texto (12 clases) |
| Base | DistilBERT-base-multilingual-cased |
| Fecha | 2024-03-15 |
| Licencia | Apache 2.0 |
| Contacto | ia-team@legaltech.com |

## Descripción

Modelo de clasificación de cláusulas contractuales en español entrenado sobre 47,832
fragmentos de contratos mercantiles, laborales y administrativos españoles y mexicanos.

## Uso previsto

**Recomendado para:**
- Prelabeling de cláusulas en flujos de due diligence
- Clasificación de contratos para enrutamiento automático
- Análisis estadístico de portfolios contractuales

**No recomendado para:**
- Decisiones legales autónomas
- Contratos en idiomas distintos al español
- Fragmentos de menos de 30 palabras

## Rendimiento

| Conjunto | F1 |
|---|---|
| Test global | 0.939 |
| Compraventa | 0.960 |
| Arrendamiento | 0.931 |
| Servicios | 0.924 |

## Limitaciones conocidas

1. Rendimiento reducido (~-8%) en contratos laborales vs mercantiles
2. Fragmentos cortos (<50 palabras): F1 0.87 vs 0.94 en fragmentos largos
3. Terminología pre-2010: variaciones léxicas no cubiertas en training data

## Consideraciones éticas

- **Sesgo de empresa**: menor rendimiento en contratos de PYMEs (F1 -0.04)
- **Revisión humana requerida** para score < 0.80
- **No es consejo legal**: el modelo clasifica texto, no valida contratos

## Datos de entrenamiento

- 47,832 fragmentos de contratos (2015-2023)
- Anonimizados con Presidio antes del almacenamiento
- Anotados por 3 personas + validación experto legal (κ = 0.87)

## Evaluación

Evaluado en test set estratificado por tipo de contrato y año.
Auditoría mensual en producción por el equipo legal.
---
```

---

## 6. Integración con HuggingFace Model Hub

HuggingFace usa el archivo `README.md` del repositorio como model card, con frontmatter YAML estandarizado:

```python
from huggingface_hub import HfApi, ModelCard, ModelCardData

# Crear model card con metadatos estructurados
card_data = ModelCardData(
    language=["es"],
    license="apache-2.0",
    library_name="transformers",
    tags=["text-classification", "legal", "spanish", "contracts"],
    datasets=["legaltech/contratos-es"],
    metrics=["f1", "precision", "recall"],
    base_model="distilbert-base-multilingual-cased",
    pipeline_tag="text-classification"
)

card_content = """
## Descripción
Clasificador de cláusulas contractuales en español basado en DistilBERT.

## Uso

```python
from transformers import pipeline

clf = pipeline("text-classification", model="legaltech/clasificador-contratos-v2")
result = clf("El arrendatario se obliga a pagar la renta mensualmente.")
print(result)
# [{'label': 'clausula_pago', 'score': 0.97}]
```

## Rendimiento
| Métrica | Valor |
|---|---|
| F1 (test) | 0.939 |
| Precisión | 0.941 |
| Recall | 0.938 |
"""

model_card = ModelCard.from_template(
    card_data=card_data,
    template_str=card_content,
    model_id="legaltech/clasificador-contratos-v2"
)

# Subir al Hub
api = HfApi()
model_card.push_to_hub(
    repo_id="legaltech/clasificador-contratos-v2",
    token="hf_..."
)
print("Model card publicado en HuggingFace Hub")
```

---

## 7. Extensiones sugeridas

- **Versionado automático**: generar model card en el pipeline CI/CD después de cada entrenamiento
- **Comparativa de versiones**: incluir tabla comparativa de métricas entre versiones del modelo
- **Integración con MLflow**: guardar model card como artefacto junto al modelo
- **Datasheet automatizado**: generar el datasheet con `pandas-profiling` para estadísticas del dataset

---

**Anterior:** [02 — Interpretabilidad](./02-interpretabilidad.md) · **Siguiente:** [04 — Cumplimiento GDPR](./04-cumplimiento-gdpr.md)
