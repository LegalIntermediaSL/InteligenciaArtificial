# Proyecto 3: Pipeline de contenido con IA

## Qué vamos a construir

Un pipeline automático que, dado un tema o URL de fuente, genera contenido
optimizado para múltiples canales: artículo de blog, hilo de Twitter/X,
post de LinkedIn y newsletter. Con versionado de prompts y control de calidad.

```
ARQUITECTURA
────────────────────────────────────────────────────
Input (tema/URL) → Extracción de contenido fuente
                        ↓
                  Investigación con Claude
                        ↓
                  Generación multi-formato
                    ├── Blog (1500 palabras)
                    ├── Twitter/X (hilo 10 tweets)
                    ├── LinkedIn (post 300 palabras)
                    └── Newsletter (500 palabras)
                        ↓
                  Evaluación de calidad automática
                        ↓
                  Guardar + publicar (n8n)
```

**Bloques que combina:** 8 (multimodalidad), 17 (automatización), 21 (startups — versionado prompts).

---

## Estructura del proyecto

```
pipeline-contenido/
├── main.py           ← orquestador del pipeline
├── prompts/          ← prompts versionados por formato
│   ├── blog_v2.txt
│   ├── twitter_v1.txt
│   ├── linkedin_v1.txt
│   └── newsletter_v1.txt
├── extractor.py      ← extraer contenido de URLs
├── generator.py      ← generación multi-formato
├── evaluator.py      ← control de calidad automático
└── publisher.py      ← enviar a n8n/Buffer/Notion
```

---

## Gestión de prompts versionados

```python
# prompts/blog_v2.txt
BLOG_V2 = """Eres un escritor de contenido especializado en tecnología y negocios.
Escribe un artículo de blog completo, optimizado para SEO y para lectores hispanohablantes.

ESTRUCTURA OBLIGATORIA:
1. Título (H1) — atractivo, con keyword principal
2. Introducción (150 palabras) — engancha al lector con un problema o dato sorprendente
3. 3-5 secciones con subtítulos (H2) — cada una con 200-300 palabras
4. Ejemplos concretos o casos reales en cada sección
5. Conclusión + llamada a la acción (100 palabras)

ESTILO:
- Segunda persona (tú/vosotros)
- Párrafos cortos (máx 3 frases)
- Una lista o tabla por sección
- Evitar jerga innecesaria

TEMA: {tema}
PUNTOS CLAVE A CUBRIR: {puntos_clave}
TONO: {tono}"""
```

```python
# prompts/twitter_v1.txt
TWITTER_V1 = """Crea un hilo de Twitter/X de exactamente 8-10 tweets sobre el tema dado.

REGLAS:
- Tweet 1: gancho potente (máx 200 chars) con dato o afirmación sorprendente
- Tweets 2-8: un punto por tweet, con numeración (2/ 3/ etc.)
- Último tweet: CTA + link a [URL_BLOG]
- Cada tweet: máx 240 caracteres
- Usa emojis con moderación (1-2 por tweet)
- Formato de salida: lista numerada, un tweet por línea

TEMA: {tema}
AUDIENCIA: {audiencia}"""
```

---

## Generador multi-formato (generator.py)

```python
import anthropic
import json
from pathlib import Path
from datetime import datetime

client = anthropic.Anthropic()

PROMPTS = {
    "blog": Path("prompts/blog_v2.txt").read_text() if Path("prompts/blog_v2.txt").exists() else """
Escribe un artículo de blog completo (1200-1500 palabras) sobre: {tema}
Puntos clave: {puntos_clave}. Tono: {tono}. Incluye introducción, 4 secciones y conclusión.""",
    
    "twitter": """Crea un hilo de 8 tweets sobre: {tema}. 
Audiencia: {audiencia}. Tweet 1: gancho. Tweets 2-7: puntos clave. Tweet 8: CTA.""",
    
    "linkedin": """Escribe un post de LinkedIn profesional (250-300 palabras) sobre: {tema}.
Empieza con una frase gancho. Usa saltos de línea frecuentes. Termina con pregunta para engagement.""",
    
    "newsletter": """Escribe la sección principal de una newsletter (400-500 palabras) sobre: {tema}.
Tono cercano, como si escribieras a un amigo. Incluye: contexto, qué significa para el lector y un consejo accionable."""
}

def generar_formato(tema: str, formato: str, contexto: dict = None) -> dict:
    """Genera contenido para un formato específico."""
    contexto = contexto or {}
    prompt_template = PROMPTS.get(formato, PROMPTS["blog"])
    
    # Rellenar el template con el contexto
    prompt = prompt_template.format(
        tema=tema,
        puntos_clave=contexto.get("puntos_clave", "los más relevantes"),
        tono=contexto.get("tono", "profesional pero accesible"),
        audiencia=contexto.get("audiencia", "profesionales de tecnología y negocios")
    )
    
    # Blog usa Sonnet (más largo y estructurado), el resto Haiku
    modelo = "claude-sonnet-4-6" if formato == "blog" else "claude-haiku-4-5-20251001"
    max_tok = 2000 if formato == "blog" else 600
    
    t_inicio = datetime.now()
    response = client.messages.create(
        model=modelo,
        max_tokens=max_tok,
        messages=[{"role": "user", "content": prompt}]
    )
    duracion_ms = (datetime.now() - t_inicio).total_seconds() * 1000
    
    contenido = response.content[0].text
    
    return {
        "formato": formato,
        "contenido": contenido,
        "palabras": len(contenido.split()),
        "modelo": modelo,
        "tokens_input": response.usage.input_tokens,
        "tokens_output": response.usage.output_tokens,
        "duracion_ms": round(duracion_ms),
        "generado_en": datetime.now().isoformat()
    }

def pipeline_completo(tema: str, formatos: list[str] = None, contexto: dict = None) -> dict:
    """Genera contenido para todos los formatos solicitados."""
    formatos = formatos or ["blog", "twitter", "linkedin", "newsletter"]
    
    resultados = {}
    coste_total = 0
    
    precios = {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00}
    }
    
    for formato in formatos:
        print(f"  Generando {formato}...")
        resultado = generar_formato(tema, formato, contexto)
        
        p = precios[resultado["modelo"]]
        coste = (resultado["tokens_input"] * p["input"] + resultado["tokens_output"] * p["output"]) / 1_000_000
        resultado["coste_usd"] = round(coste, 5)
        coste_total += coste
        
        resultados[formato] = resultado
    
    return {
        "tema": tema,
        "formatos": resultados,
        "coste_total_usd": round(coste_total, 4),
        "generado_en": datetime.now().isoformat()
    }
```

---

## Evaluador de calidad (evaluator.py)

```python
import anthropic
import json

client = anthropic.Anthropic()

def evaluar_contenido(contenido: str, formato: str, tema: str) -> dict:
    """Evalúa la calidad del contenido generado usando LLM-as-judge."""
    
    criterios = {
        "blog": ["relevancia_tema", "estructura", "claridad", "seo_friendly", "llamada_accion"],
        "twitter": ["gancho_tweet1", "longitud_tweets", "coherencia_hilo", "engagement"],
        "linkedin": ["tono_profesional", "gancho_inicial", "legibilidad", "cta"],
        "newsletter": ["cercanía_tono", "valor_para_lector", "consejo_accionable", "longitud"]
    }
    
    criterios_formato = criterios.get(formato, criterios["blog"])
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Evalúa este contenido de {formato} sobre "{tema}".
Puntúa cada criterio del 1 al 10 y da una nota global. JSON sin markdown:
{{
  "puntuaciones": {{{", ".join(f'"{c}": 0' for c in criterios_formato)}}},
  "nota_global": 0,
  "puntos_fuertes": ["max 2"],
  "puntos_mejora": ["max 2"],
  "listo_publicar": true
}}

CONTENIDO:
{contenido[:2000]}"""
        }]
    )
    
    evaluacion = json.loads(response.content[0].text)
    evaluacion["formato"] = formato
    return evaluacion

def evaluar_pipeline(pipeline_resultado: dict) -> dict:
    """Evalúa todos los formatos del pipeline."""
    tema = pipeline_resultado["tema"]
    evaluaciones = {}
    
    for formato, datos in pipeline_resultado["formatos"].items():
        print(f"  Evaluando {formato}...")
        evaluaciones[formato] = evaluar_contenido(datos["contenido"], formato, tema)
    
    nota_media = sum(e["nota_global"] for e in evaluaciones.values()) / len(evaluaciones)
    todos_listos = all(e["listo_publicar"] for e in evaluaciones.values())
    
    return {
        "evaluaciones": evaluaciones,
        "nota_media": round(nota_media, 1),
        "todos_listos_para_publicar": todos_listos,
        "formatos_rechazados": [f for f, e in evaluaciones.items() if not e["listo_publicar"]]
    }
```

---

## Orquestador (main.py)

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import json
import uuid
from datetime import datetime
from generator import pipeline_completo
from evaluator import evaluar_pipeline

app = FastAPI(title="Pipeline de Contenido IA", version="1.0")

# Almacén en memoria (en producción: Redis o SQLite)
JOBS: dict = {}

class SolicitudContenido(BaseModel):
    tema: str
    formatos: list[str] = ["blog", "twitter", "linkedin", "newsletter"]
    contexto: dict = {}

def ejecutar_pipeline(job_id: str, solicitud: SolicitudContenido):
    """Se ejecuta en background."""
    try:
        JOBS[job_id]["estado"] = "generando"
        
        print(f"[{job_id}] Generando contenido para: {solicitud.tema}")
        resultado = pipeline_completo(solicitud.tema, solicitud.formatos, solicitud.contexto)
        
        JOBS[job_id]["estado"] = "evaluando"
        print(f"[{job_id}] Evaluando calidad...")
        evaluacion = evaluar_pipeline(resultado)
        
        JOBS[job_id].update({
            "estado": "completado",
            "resultado": resultado,
            "evaluacion": evaluacion,
            "completado_en": datetime.now().isoformat()
        })
        print(f"[{job_id}] ✅ Completado. Nota media: {evaluacion['nota_media']}/10")
        
    except Exception as e:
        JOBS[job_id]["estado"] = "error"
        JOBS[job_id]["error"] = str(e)

@app.post("/generar")
async def generar(solicitud: SolicitudContenido, background_tasks: BackgroundTasks):
    """Inicia el pipeline en background y devuelve un job_id para consultar el estado."""
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"estado": "en_cola", "tema": solicitud.tema, "iniciado_en": datetime.now().isoformat()}
    background_tasks.add_task(ejecutar_pipeline, job_id, solicitud)
    return {"job_id": job_id, "estado": "en_cola"}

@app.get("/job/{job_id}")
async def estado_job(job_id: str):
    if job_id not in JOBS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return JOBS[job_id]

@app.get("/jobs")
async def listar_jobs():
    return [{"job_id": jid, "estado": j["estado"], "tema": j["tema"]} 
            for jid, j in JOBS.items()]
```

---

## Uso

```bash
# Arrancar
uvicorn main:app --reload

# Lanzar pipeline
curl -X POST http://localhost:8000/generar \
  -H "Content-Type: application/json" \
  -d '{
    "tema": "Por qué los agentes de IA están transformando el trabajo del conocimiento",
    "formatos": ["blog", "twitter", "linkedin"],
    "contexto": {
      "tono": "experto pero accesible",
      "audiencia": "directores de tecnología y emprendedores"
    }
  }'

# Consultar estado
curl http://localhost:8000/job/JOB_ID

# Coste típico por pipeline completo (4 formatos):
# Blog (Sonnet):    ~$0.015
# Twitter (Haiku):  ~$0.002
# LinkedIn (Haiku): ~$0.002
# Newsletter (Haiku): ~$0.002
# Total: ~$0.021 por pipeline
```

## Recursos

- [Notebook interactivo](../notebooks/proyectos-integradores/03-pipeline-contenido.ipynb)
- [FastAPI — Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
