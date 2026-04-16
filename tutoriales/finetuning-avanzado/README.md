# Bloque 14 — Fine-tuning Avanzado de LLMs

> **Bloque:** Fine-tuning avanzado · **Nivel:** Avanzado · **Tiempo estimado:** 15 min

---

## Índice

1. [¿Qué es el fine-tuning avanzado?](#qué-es-el-fine-tuning-avanzado)
2. [Fine-tuning supervisado vs ajuste por preferencias](#fine-tuning-supervisado-vs-ajuste-por-preferencias)
3. [Tutoriales del bloque](#tutoriales-del-bloque)
4. [Requisitos de instalación](#requisitos-de-instalación)
5. [Cómo está organizado este bloque](#cómo-está-organizado-este-bloque)

---

## ¿Qué es el fine-tuning avanzado?

Los modelos de lenguaje como LLaMA, Mistral o Qwen se preentrenan sobre billones de tokens de texto general. Sin embargo, para tareas específicas —responder con el estilo de tu empresa, seguir instrucciones en un dominio técnico, generar código en un framework privado— es necesario un paso adicional de adaptación.

El **fine-tuning avanzado** va más allá del ajuste básico de parámetros: abarca técnicas que alinean el comportamiento del modelo con preferencias humanas, métodos eficientes que entrenan solo una fracción de los parámetros, pipelines de evaluación rigurosos y estrategias de despliegue optimizadas para producción.

Este bloque cubre las cuatro patas del ciclo completo:

- **RLHF y DPO**: cómo alinear modelos con preferencias humanas.
- **Instruction tuning**: cómo enseñar al modelo a seguir instrucciones de forma confiable.
- **Evaluación**: métricas automáticas, LLM-as-judge y benchmarks estándar.
- **Despliegue**: servir modelos propios con vLLM, TGI y cuantización.

---

## Fine-tuning supervisado vs ajuste por preferencias

| Criterio | Fine-tuning supervisado (SFT) | Ajuste por preferencias (DPO/RLHF) |
|---|---|---|
| **Qué aprende** | A imitar respuestas de ejemplo | A preferir respuestas mejor valoradas sobre peores |
| **Datos necesarios** | Pares `(instrucción, respuesta)` | Tripletes `(instrucción, respuesta_buena, respuesta_mala)` |
| **Coste de datos** | Bajo | Alto (requiere comparaciones humanas o de modelo) |
| **Cuándo usarlo** | Adaptar estilo, dominio o formato | Mejorar alineación, reducir respuestas dañinas, mejorar utilidad general |
| **Riesgo principal** | Overfitting al estilo del dataset | Reward hacking, colapso de diversidad |
| **Complejidad de entrenamiento** | Baja (cross-entropy estándar) | Media-alta (PPO) o media (DPO) |
| **Herramientas principales** | `SFTTrainer` (TRL) | `DPOTrainer` (TRL), `PPOTrainer` (TRL) |

**Regla práctica:**

- Si partes de un modelo base sin instrucciones → empieza con **SFT**.
- Si el modelo ya sigue instrucciones pero quieres mejorar calidad y alineación → añade **DPO**.
- Si necesitas control fino sobre el comportamiento y tienes datos de preferencia de calidad → considera **RLHF completo** con PPO.

---

## Tutoriales del bloque

| # | Archivo | Tema | Tiempo estimado |
|---|---|---|---|
| 01 | [01-dpo-rlhf.md](./01-dpo-rlhf.md) | DPO y RLHF — ajuste por preferencias humanas | 55 min |
| 02 | [02-instruction-tuning.md](./02-instruction-tuning.md) | Instruction tuning con SFTTrainer | 45 min |
| 03 | [03-evaluacion-modelos-finetuneados.md](./03-evaluacion-modelos-finetuneados.md) | Evaluación de modelos fine-tuneados | 50 min |
| 04 | [04-despliegue-modelos-propios.md](./04-despliegue-modelos-propios.md) | Despliegue con vLLM, TGI y cuantización | 60 min |

---

## Requisitos de instalación

Instala las dependencias principales del bloque:

```bash
# Entrenamiento y fine-tuning
pip install trl transformers datasets peft accelerate bitsandbytes

# Evaluación
pip install evaluate rouge-score sacrebleu lm-eval

# Experimentos y métricas
pip install wandb

# Despliegue (instalar en entorno de serving separado)
pip install vllm
pip install text-generation[server]

# API y contenedores
pip install fastapi uvicorn httpx
```

Variables de entorno necesarias:

```bash
export HF_TOKEN="hf_..."                    # Token de Hugging Face (para modelos privados)
export WANDB_API_KEY="..."                  # Para logging de experimentos
export ANTHROPIC_API_KEY="sk-ant-..."       # Para LLM-as-judge con Claude (tutorial 03)
export OPENAI_API_KEY="sk-..."             # Para LLM-as-judge con GPT-4 (tutorial 03, opcional)
```

Requisitos de hardware:

| Tutorial | GPU mínima recomendada | RAM GPU |
|---|---|---|
| 01 — DPO | NVIDIA RTX 3090 / A10G | 24 GB |
| 02 — SFT | NVIDIA RTX 3090 / A10G | 24 GB |
| 03 — Evaluación | CPU suficiente para métricas; GPU para `lm-eval` | 16 GB+ |
| 04 — Despliegue | NVIDIA A10G / A100 para vLLM | 24 GB+ |

> **Nota:** Todos los ejemplos de código están diseñados para funcionar en Google Colab (A100 40 GB) o en un servidor con GPU de al menos 24 GB. Para hardware más limitado, consulta la sección de cuantización del tutorial 04.

---

## Cómo está organizado este bloque

Los tutoriales siguen el ciclo natural de trabajo con LLMs propios:

1. **DPO/RLHF** establece la teoría de alineación y muestra cómo ajustar preferencias. Es el tutorial más conceptual del bloque.
2. **Instruction tuning** es el punto de entrada práctico: cómo preparar datos y lanzar un fine-tuning supervisado desde cero.
3. **Evaluación** abarca cómo medir si el fine-tuning funcionó —imprescindible antes de desplegar en producción.
4. **Despliegue** cierra el ciclo: cómo servir el modelo resultante de forma eficiente y escalable.

Si eres nuevo en fine-tuning, empieza por el tutorial 02 (instruction tuning) antes de abordar el 01 (DPO). Si ya tienes experiencia con SFT, puedes ir directamente al 01 o al 03.

---

**Siguiente:** [01 — DPO y RLHF](./01-dpo-rlhf.md)
