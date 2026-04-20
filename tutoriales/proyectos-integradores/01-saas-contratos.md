# Proyecto 1: SaaS de análisis de contratos

## Qué vamos a construir

Un SaaS funcional que permite a despachos de abogados y equipos legales subir contratos
en PDF, analizarlos con IA y recibir un informe estructurado con riesgos, cláusulas clave
y comparación contra una base de contratos anteriores.

```
ARQUITECTURA
────────────────────────────────────────────────────
Cliente (PDF) → FastAPI → Files API (Anthropic)
                    ↓
              Claude Sonnet → análisis JSON
                    ↓
              ChromaDB → búsqueda de contratos similares
                    ↓
              Informe final → guardado en SQLite
                    ↓
              Respuesta al cliente con PDF del informe
```

**Bloques que combina:** 3 (APIs), 13 (BD vectoriales), 18 (Files API), 19 (Agent SDK),
seguridad (PII), producción (rate limiting, logs).

---

## Estructura del proyecto

```
saas-contratos/
├── main.py              ← FastAPI app
├── analyzer.py          ← lógica de análisis con Claude
├── vector_store.py      ← ChromaDB para contratos similares
├── models.py            ← esquemas Pydantic
├── database.py          ← SQLite para historial
├── .env                 ← ANTHROPIC_API_KEY
└── requirements.txt
```

---

## Paso 1: Modelos Pydantic (models.py)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ClausulaRiesgo(BaseModel):
    titulo: str
    texto: str
    severidad: str  # "alta" | "media" | "baja"
    recomendacion: str

class AnalisisContrato(BaseModel):
    tipo_contrato: str
    partes: list[str]
    fecha_inicio: Optional[str]
    fecha_fin: Optional[str]
    valor_eur: Optional[float]
    renovacion_automatica: bool
    clausulas_riesgo: list[ClausulaRiesgo]
    clausulas_faltantes: list[str]
    puntuacion_riesgo: int = Field(ge=0, le=100)
    resumen_ejecutivo: str
    procesado_en: str

class SolicitudAnalisis(BaseModel):
    texto_contrato: str
    tipo_esperado: Optional[str] = None

class ResultadoCompleto(BaseModel):
    analisis: AnalisisContrato
    contratos_similares: list[dict]
    advertencia_legal: str = "Este análisis es orientativo. Requiere revisión por abogado cualificado."
```

---

## Paso 2: Analizador con Claude (analyzer.py)

```python
import anthropic
import json
from models import AnalisisContrato, SolicitudAnalisis
from datetime import datetime

client = anthropic.Anthropic()

SYSTEM_LEGAL = """Eres un asistente legal especializado en análisis de contratos.
Tu función es identificar riesgos, cláusulas problemáticas y elementos faltantes.

REGLAS CRÍTICAS:
- Responde SIEMPRE con JSON válido según el esquema solicitado
- Nunca inventes datos que no estén en el contrato
- Marca como alta severidad solo riesgos realmente significativos
- El campo puntuacion_riesgo es 0 (sin riesgo) a 100 (riesgo crítico)
- Incluye siempre advertencia sobre revisión humana en el resumen"""

def analizar_contrato(solicitud: SolicitudAnalisis) -> AnalisisContrato:
    """Analiza un contrato usando claude-sonnet-4-6 con structured output."""
    
    prompt = f"""Analiza este contrato y devuelve un JSON con exactamente esta estructura:
{{
  "tipo_contrato": "prestacion_servicios|compraventa|arrendamiento|laboral|confidencialidad|otro",
  "partes": ["Empresa A", "Empresa B"],
  "fecha_inicio": "YYYY-MM-DD o null",
  "fecha_fin": "YYYY-MM-DD o null",
  "valor_eur": 0.0,
  "renovacion_automatica": false,
  "clausulas_riesgo": [
    {{
      "titulo": "nombre de la cláusula",
      "texto": "extracto relevante",
      "severidad": "alta|media|baja",
      "recomendacion": "qué hacer al respecto"
    }}
  ],
  "clausulas_faltantes": ["lista de cláusulas que deberían estar y no están"],
  "puntuacion_riesgo": 0,
  "resumen_ejecutivo": "2-3 frases con lo más importante",
  "procesado_en": "{datetime.now().isoformat()}"
}}

CONTRATO:
{solicitud.texto_contrato[:8000]}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_LEGAL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    datos = json.loads(response.content[0].text)
    return AnalisisContrato(**datos)


def analizar_contrato_con_files_api(file_id: str) -> AnalisisContrato:
    """Versión usando Files API para PDFs ya subidos (más eficiente en producción)."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_LEGAL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": file_id}
                },
                {
                    "type": "text",
                    "text": "Analiza este contrato y devuelve el JSON de análisis solicitado."
                }
            ]
        }]
    )
    
    datos = json.loads(response.content[0].text)
    datos["procesado_en"] = datetime.now().isoformat()
    return AnalisisContrato(**datos)
```

---

## Paso 3: Base de datos vectorial (vector_store.py)

```python
import chromadb
from chromadb.utils import embedding_functions
import anthropic
import json

client_anthropic = anthropic.Anthropic()

def get_embedding(texto: str) -> list[float]:
    """Genera embedding usando la API de Anthropic (o usa sentence-transformers local)."""
    # Alternativa gratuita: sentence-transformers
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # return model.encode(texto).tolist()
    
    # Con Voyage AI (recomendado por Anthropic para producción):
    # import voyageai; vo = voyageai.Client(); return vo.embed([texto], model="voyage-3").embeddings[0]
    
    # Para la demo: usar el hash como placeholder
    import hashlib
    h = int(hashlib.md5(texto.encode()).hexdigest(), 16)
    return [(h >> i & 0xFF) / 255.0 for i in range(384)]

class AlmacenContratos:
    def __init__(self, ruta: str = "./chroma_contratos"):
        self.client = chromadb.PersistentClient(path=ruta)
        self.coleccion = self.client.get_or_create_collection(
            name="contratos",
            metadata={"hnsw:space": "cosine"}
        )
    
    def guardar_contrato(self, id_contrato: str, texto: str, metadatos: dict):
        """Indexa un contrato para búsqueda semántica futura."""
        # Usar primeros 2000 chars para el embedding (lo más representativo)
        embedding = get_embedding(texto[:2000])
        self.coleccion.add(
            ids=[id_contrato],
            embeddings=[embedding],
            documents=[texto[:500]],  # Fragmento para mostrar en resultados
            metadatas=[metadatos]
        )
    
    def buscar_similares(self, texto: str, n: int = 3) -> list[dict]:
        """Encuentra contratos similares al proporcionado."""
        if self.coleccion.count() == 0:
            return []
        
        embedding = get_embedding(texto[:2000])
        resultados = self.coleccion.query(
            query_embeddings=[embedding],
            n_results=min(n, self.coleccion.count())
        )
        
        similares = []
        for i, (doc_id, distancia, meta) in enumerate(zip(
            resultados["ids"][0],
            resultados["distances"][0],
            resultados["metadatas"][0]
        )):
            similares.append({
                "id": doc_id,
                "similitud": round(1 - distancia, 3),
                "tipo": meta.get("tipo_contrato", "desconocido"),
                "fecha": meta.get("procesado_en", ""),
                "puntuacion_riesgo": meta.get("puntuacion_riesgo", 0)
            })
        
        return similares
```

---

## Paso 4: API principal (main.py)

```python
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
import anthropic
import sqlite3
import json
import uuid
from datetime import datetime
from models import SolicitudAnalisis, ResultadoCompleto
from analyzer import analizar_contrato, analizar_contrato_con_files_api
from vector_store import AlmacenContratos

app = FastAPI(title="SaaS Análisis de Contratos", version="1.0")
cliente_anthropic = anthropic.Anthropic()
almacen = AlmacenContratos()

# Rate limiting sencillo (en producción: Redis + SlowAPI)
LLAMADAS_POR_IP: dict = {}

def verificar_rate_limit(ip: str = "127.0.0.1"):
    from time import time
    ahora = int(time())
    ventana = 60  # 1 minuto
    max_llamadas = 10
    
    historial = LLAMADAS_POR_IP.get(ip, [])
    historial = [t for t in historial if ahora - t < ventana]
    
    if len(historial) >= max_llamadas:
        raise HTTPException(status_code=429, detail="Rate limit: máx 10 análisis/minuto")
    
    historial.append(ahora)
    LLAMADAS_POR_IP[ip] = historial

def init_db():
    con = sqlite3.connect("contratos.db")
    con.execute("""CREATE TABLE IF NOT EXISTS analisis (
        id TEXT PRIMARY KEY,
        tipo_contrato TEXT,
        puntuacion_riesgo INTEGER,
        resultado_json TEXT,
        procesado_en TEXT
    )""")
    con.commit()
    con.close()

init_db()

@app.post("/analizar-texto", response_model=ResultadoCompleto)
async def analizar_texto(solicitud: SolicitudAnalisis):
    """Analiza un contrato enviado como texto plano."""
    verificar_rate_limit()
    
    if len(solicitud.texto_contrato) < 100:
        raise HTTPException(status_code=400, detail="El contrato debe tener al menos 100 caracteres")
    
    # Analizar con Claude
    analisis = analizar_contrato(solicitud)
    
    # Buscar contratos similares en la BD vectorial
    similares = almacen.buscar_similares(solicitud.texto_contrato)
    
    # Guardar en SQLite y vectorial
    id_analisis = str(uuid.uuid4())
    con = sqlite3.connect("contratos.db")
    con.execute(
        "INSERT INTO analisis VALUES (?, ?, ?, ?, ?)",
        (id_analisis, analisis.tipo_contrato, analisis.puntuacion_riesgo,
         analisis.model_dump_json(), analisis.procesado_en)
    )
    con.commit()
    con.close()
    
    almacen.guardar_contrato(
        id_contrato=id_analisis,
        texto=solicitud.texto_contrato,
        metadatos={
            "tipo_contrato": analisis.tipo_contrato,
            "puntuacion_riesgo": analisis.puntuacion_riesgo,
            "procesado_en": analisis.procesado_en
        }
    )
    
    return ResultadoCompleto(analisis=analisis, contratos_similares=similares)


@app.post("/analizar-pdf")
async def analizar_pdf(archivo: UploadFile = File(...)):
    """Sube un PDF y lo analiza usando la Files API de Anthropic."""
    verificar_rate_limit()
    
    if not archivo.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
    
    contenido = await archivo.read()
    
    # Subir a la Files API de Anthropic
    file_response = cliente_anthropic.beta.files.upload(
        file=(archivo.filename, contenido, "application/pdf"),
    )
    
    # Analizar usando el file_id (no se retransmite el PDF)
    analisis = analizar_contrato_con_files_api(file_response.id)
    similares = []  # Necesitaríamos el texto extraído para buscar similares
    
    # Limpiar el archivo de Anthropic (buena práctica)
    cliente_anthropic.beta.files.delete(file_response.id)
    
    return ResultadoCompleto(analisis=analisis, contratos_similares=similares)


@app.get("/historial")
async def obtener_historial(limite: int = 10):
    """Devuelve los últimos análisis realizados."""
    con = sqlite3.connect("contratos.db")
    filas = con.execute(
        "SELECT id, tipo_contrato, puntuacion_riesgo, procesado_en FROM analisis ORDER BY procesado_en DESC LIMIT ?",
        (limite,)
    ).fetchall()
    con.close()
    
    return [
        {"id": f[0], "tipo": f[1], "riesgo": f[2], "fecha": f[3]}
        for f in filas
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "contratos_indexados": almacen.coleccion.count()}
```

---

## Paso 5: Arrancar en local

```bash
pip install anthropic fastapi uvicorn chromadb python-dotenv pydantic

# .env
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Ejecutar
uvicorn main:app --reload

# Probar
curl -X POST http://localhost:8000/analizar-texto \
  -H "Content-Type: application/json" \
  -d '{"texto_contrato": "CONTRATO DE SERVICIOS entre A y B..."}'

# Ver historial
curl http://localhost:8000/historial
```

---

## Checklist de producción

```
Backend:
  ✓ Rate limiting por usuario (no solo por IP) con Redis
  ✓ Autenticación JWT o API keys para cada cliente
  ✓ Logs estructurados con IDs de ejecución
  ✓ Timeouts en llamadas a Claude (max 60s)
  ✓ Validación de tamaño máximo de PDF (< 50MB)

Seguridad:
  ✓ Escanear PDFs con antivirus antes de procesar
  ✓ No guardar texto del contrato sin cifrar si contiene PII
  ✓ Contratos eliminados de Files API tras análisis
  ✓ HTTPS obligatorio en producción

Negocio:
  ✓ Disclaimer legal visible en cada respuesta
  ✓ Auditoría de uso por cliente (para facturación)
  ✓ Límite de análisis por plan (Starter: 20/mes, Pro: 200/mes)
```

## Recursos

- [Notebook interactivo](../notebooks/proyectos-integradores/01-saas-contratos.ipynb)
- [Files API — documentación](https://docs.anthropic.com/en/docs/build-with-claude/files)
- [ChromaDB — getting started](https://docs.trychroma.com/getting-started)
