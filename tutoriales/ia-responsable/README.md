# Bloque 15 — IA Responsable: Fairness, Interpretabilidad y Cumplimiento

> **Bloque:** IA Responsable · **Nivel:** Avanzado · **Tiempo estimado:** 5 h (bloque completo)

---

## Introducción

Construir sistemas de IA que funcionen bien técnicamente no es suficiente. Un sistema responsable también debe ser **justo** con todos los grupos de personas que afecta, **explicable** para quienes lo usan y supervisan, y **conforme** con el marco legal vigente —en especial el GDPR europeo y el EU AI Act.

Este bloque cubre las herramientas, métricas y prácticas concretas que permiten pasar de modelos que "predicen bien" a modelos que "predicen bien y son confiables". No es filosofía: cada sección incluye código ejecutable.

> **Nota sobre el enfoque:**
> IA responsable no es un checkbox de cumplimiento. Es una práctica de ingeniería. Los conceptos de fairness, interpretabilidad y privacidad deben integrarse desde el diseño del sistema, no añadirse al final. Este bloque muestra cómo hacerlo en Python, con herramientas del ecosistema real.

---

## Requisitos previos

- Python 3.10+ con entorno virtual activo
- Conocimientos sólidos de scikit-learn y pandas (Bloque 4)
- Familiaridad con modelos de clasificación supervisada (Bloque 2)
- Haber completado el Bloque 12 (Seguridad) es recomendable

## Instalación de dependencias

```bash
# Fairness y sesgos
pip install fairlearn aif360 pandas scikit-learn matplotlib seaborn

# Interpretabilidad
pip install shap lime transformers bertviz captum

# Model cards
pip install model-card-toolkit

# Privacidad y cumplimiento
pip install faker hashlib cryptography
```

Para soporte completo de AIF360 con todos los algoritmos:

```bash
pip install 'aif360[all]'
```

---

## Tutoriales del bloque

| # | Tutorial | Tema principal | Tiempo |
|---|----------|---------------|--------|
| 01 | [Sesgos y Fairness](./01-sesgos-fairness.md) | Tipos de sesgo, métricas de equidad, detección y mitigación | 90 min |
| 02 | [Interpretabilidad y Explicabilidad](./02-interpretabilidad.md) | SHAP, LIME, attention viz, saliency maps | 75 min |
| 03 | [Model Cards y Documentación Responsable](./03-model-cards-datasheets.md) | Estándar Mitchell et al., datasheets, HuggingFace | 45 min |
| 04 | [Cumplimiento GDPR y EU AI Act](./04-cumplimiento-gdpr.md) | Regulación, anonimización, machine unlearning, auditoría | 60 min |

---

## Mapa conceptual del bloque

```
Datos                  Modelo                 Despliegue
┌──────────────┐       ┌──────────────────┐   ┌─────────────────────┐
│ Datasheets   │       │ Detección sesgos │   │ Model cards         │
│ Minimización│──────▶│ Métricas fairness│──▶│ Derecho explicación │
│ Anonimización│       │ Mitigación       │   │ Auditoría GDPR      │
│ Consentimiento│      │ Interpretabilidad│   │ EU AI Act           │
└──────────────┘       └──────────────────┘   └─────────────────────┘
        │                      │                        │
        └──────────────────────┴────────────────────────┘
                    IA Responsable por diseño
```

---

## Orden de lectura recomendado

1. **01 → 02**: Empieza por sesgos e interpretabilidad. Son las bases técnicas; van juntos porque detectar un sesgo sin poder explicarlo no permite actuar.
2. **03**: Una vez entiendes los problemas, aprende a documentarlos con model cards. La documentación responsable es el puente entre técnica y gobernanza.
3. **04**: Cierra con el marco legal. Conocer primero los problemas técnicos hace que los requisitos regulatorios tengan mucho más sentido.

---

**Siguiente:** [01 — Sesgos y Fairness](./01-sesgos-fairness.md)
