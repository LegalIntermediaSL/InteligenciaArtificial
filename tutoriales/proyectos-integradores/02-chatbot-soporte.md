# Proyecto 2: Chatbot de soporte con memoria

## Qué vamos a construir

Un chatbot de soporte para SaaS que recuerda el historial del usuario, consulta una
base de conocimiento interna, escala a humano cuando no sabe la respuesta,
y aprende de las interacciones para mejorar.

```
ARQUITECTURA
────────────────────────────────────────────────────
Usuario → FastAPI → Gestión de sesión (SQLite)
                ↓
          Búsqueda en base de conocimiento (ChromaDB)
                ↓
          Claude Haiku (respuesta rápida)
                ↓
          ¿Confianza baja? → Claude Sonnet (reintento)
                ↓
          ¿Sigue sin saber? → Escalar a Slack/email
                ↓
          Guardar feedback → mejora continua
```

**Bloques que combina:** 2 (LLMs), 3 (APIs), 13 (BD vectoriales), 19 (memoria), 22 (n8n).

---

## Estructura del proyecto

```
chatbot-soporte/
├── main.py          ← FastAPI con WebSocket y REST
├── brain.py         ← lógica de razonamiento con Claude
├── knowledge.py     ← base de conocimiento en ChromaDB
├── session.py       ← gestión de sesiones y memoria
├── escalation.py    ← escalar a humano
└── seed_knowledge.py ← poblar la BD con artículos de ayuda
```

---

## Base de conocimiento (knowledge.py)

```python
import chromadb
import json

# Base de conocimiento de ejemplo (en producción: leer de Notion/Confluence/Help Center)
ARTICULOS_AYUDA = [
    {
        "id": "kb_001",
        "titulo": "Cómo resetear tu contraseña",
        "contenido": "Para resetear tu contraseña ve a Configuración → Seguridad → Cambiar contraseña. Recibirás un email con el enlace. El enlace expira en 24 horas.",
        "categoria": "cuenta"
    },
    {
        "id": "kb_002",
        "titulo": "Planes de precios y facturación",
        "contenido": "Ofrecemos tres planes: Starter (9€/mes, 1 usuario), Pro (29€/mes, hasta 5 usuarios), Business (99€/mes, usuarios ilimitados). La facturación es mensual o anual con 20% de descuento.",
        "categoria": "facturacion"
    },
    {
        "id": "kb_003",
        "titulo": "Integración con API",
        "contenido": "La API REST está disponible en todos los planes Pro y Business. Documentación en docs.tuproducto.com. Rate limit: 100 requests/minuto en Pro, 1000 en Business. Autenticación por Bearer token.",
        "categoria": "tecnico"
    },
    {
        "id": "kb_004",
        "titulo": "Exportar datos",
        "contenido": "Puedes exportar todos tus datos en formato CSV o JSON desde Configuración → Datos → Exportar. El archivo se genera en background y recibirás un email con el enlace de descarga.",
        "categoria": "datos"
    },
    {
        "id": "kb_005",
        "titulo": "Cancelar suscripción",
        "contenido": "Puedes cancelar en cualquier momento desde Configuración → Facturación → Cancelar suscripción. Mantendrás acceso hasta el fin del periodo pagado. No hay penalización por cancelación anticipada.",
        "categoria": "facturacion"
    }
]

class BaseConocimiento:
    def __init__(self):
        self.client = chromadb.Client()
        self.col = self.client.get_or_create_collection("kb_soporte")
        self._sembrar()
    
    def _sembrar(self):
        if self.col.count() > 0:
            return
        for art in ARTICULOS_AYUDA:
            self.col.add(
                ids=[art["id"]],
                documents=[f"{art['titulo']}\n{art['contenido']}"],
                metadatas=[{"titulo": art["titulo"], "categoria": art["categoria"]}]
            )
    
    def buscar(self, pregunta: str, n: int = 3) -> list[dict]:
        if self.col.count() == 0:
            return []
        resultados = self.col.query(query_texts=[pregunta], n_results=min(n, self.col.count()))
        
        contexto = []
        for doc, meta, dist in zip(
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0]
        ):
            contexto.append({
                "contenido": doc,
                "titulo": meta["titulo"],
                "relevancia": round(1 - dist, 3)
            })
        return contexto
```

---

## Gestión de sesiones (session.py)

```python
import sqlite3
import json
from datetime import datetime

def init_db():
    con = sqlite3.connect("chatbot.db")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sesiones (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            creada_en TEXT,
            ultima_actividad TEXT
        );
        CREATE TABLE IF NOT EXISTS mensajes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            rol TEXT,
            contenido TEXT,
            timestamp TEXT,
            modelo TEXT,
            escalado INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            mensaje_id INTEGER,
            util INTEGER,  -- 1=útil, 0=no útil
            timestamp TEXT
        );
    """)
    con.commit()
    con.close()

def guardar_mensaje(session_id: str, rol: str, contenido: str, modelo: str = None):
    con = sqlite3.connect("chatbot.db")
    con.execute(
        "INSERT INTO mensajes (session_id, rol, contenido, timestamp, modelo) VALUES (?, ?, ?, ?, ?)",
        (session_id, rol, contenido, datetime.now().isoformat(), modelo)
    )
    con.execute(
        "UPDATE sesiones SET ultima_actividad=? WHERE session_id=?",
        (datetime.now().isoformat(), session_id)
    )
    con.commit()
    con.close()

def obtener_historial(session_id: str, max_mensajes: int = 10) -> list[dict]:
    con = sqlite3.connect("chatbot.db")
    filas = con.execute(
        "SELECT rol, contenido FROM mensajes WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
        (session_id, max_mensajes)
    ).fetchall()
    con.close()
    return [{"role": f[0], "content": f[1]} for f in reversed(filas)]

def crear_sesion(session_id: str, user_id: str):
    con = sqlite3.connect("chatbot.db")
    ahora = datetime.now().isoformat()
    con.execute(
        "INSERT OR IGNORE INTO sesiones VALUES (?, ?, ?, ?)",
        (session_id, user_id, ahora, ahora)
    )
    con.commit()
    con.close()

def guardar_feedback(session_id: str, util: bool):
    con = sqlite3.connect("chatbot.db")
    con.execute(
        "INSERT INTO feedback (session_id, util, timestamp) VALUES (?, ?, ?)",
        (session_id, 1 if util else 0, datetime.now().isoformat())
    )
    con.commit()
    con.close()
```

---

## Motor de razonamiento (brain.py)

```python
import anthropic
from knowledge import BaseConocimiento
from session import obtener_historial, guardar_mensaje

client = anthropic.Anthropic()
kb = BaseConocimiento()

SYSTEM_SOPORTE = """Eres el asistente de soporte de TuProducto SaaS.
Tu objetivo es resolver las dudas del usuario de forma rápida y clara.

REGLAS:
- Usa SOLO la información de la base de conocimiento proporcionada
- Si no sabes la respuesta con seguridad, di: "No tengo esta información. Te voy a conectar con un agente humano."
- Respuestas cortas (máximo 3 párrafos)
- Tono: amigable y profesional
- Si el usuario está frustrado, empatiza primero antes de dar la solución"""

def responder(session_id: str, pregunta: str) -> dict:
    """Genera una respuesta usando memoria de sesión + base de conocimiento."""
    
    # Recuperar historial de la sesión
    historial = obtener_historial(session_id, max_mensajes=8)
    
    # Buscar en base de conocimiento
    contexto_kb = kb.buscar(pregunta, n=3)
    contexto_texto = "\n\n".join([
        f"[{art['titulo']}]\n{art['contenido']}"
        for art in contexto_kb if art["relevancia"] > 0.3
    ])
    
    # Construir mensajes
    mensajes = historial + [{
        "role": "user",
        "content": f"BASE DE CONOCIMIENTO DISPONIBLE:\n{contexto_texto or 'Sin resultados relevantes.'}\n\nPREGUNTA DEL USUARIO: {pregunta}"
    }]
    
    # Intentar primero con Haiku (rápido y económico)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=SYSTEM_SOPORTE,
        messages=mensajes
    )
    respuesta = resp.content[0].text
    modelo_usado = "haiku"
    
    # Si no sabe la respuesta, reintento con Sonnet
    if "No tengo esta información" in respuesta or "agente humano" in respuesta.lower():
        resp2 = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=SYSTEM_SOPORTE,
            messages=mensajes
        )
        respuesta_sonnet = resp2.content[0].text
        
        # ¿Sonnet tampoco sabe? → escalar
        if "No tengo esta información" in respuesta_sonnet:
            return {
                "respuesta": "No tengo suficiente información para responder esta pregunta correctamente. Te voy a conectar con un agente de soporte que podrá ayudarte mejor. Tiempo estimado de respuesta: < 2 horas.",
                "modelo": "escalado",
                "necesita_escalado": True,
                "contexto_kb": contexto_kb
            }
        
        respuesta = respuesta_sonnet
        modelo_usado = "sonnet"
    
    # Guardar en historial
    guardar_mensaje(session_id, "user", pregunta)
    guardar_mensaje(session_id, "assistant", respuesta, modelo_usado)
    
    return {
        "respuesta": respuesta,
        "modelo": modelo_usado,
        "necesita_escalado": False,
        "articulos_usados": [art["titulo"] for art in contexto_kb if art["relevancia"] > 0.3]
    }
```

---

## API principal (main.py)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from brain import responder
from session import init_db, crear_sesion, guardar_feedback

app = FastAPI(title="Chatbot Soporte", version="1.0")
init_db()

class MensajeEntrada(BaseModel):
    session_id: str | None = None
    user_id: str = "anonimo"
    pregunta: str

class FeedbackEntrada(BaseModel):
    session_id: str
    util: bool

@app.post("/chat")
async def chat(entrada: MensajeEntrada):
    if not entrada.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    # Crear sesión si es nueva
    session_id = entrada.session_id or str(uuid.uuid4())
    crear_sesion(session_id, entrada.user_id)
    
    resultado = responder(session_id, entrada.pregunta)
    
    # Si necesita escalado, activar flujo de notificación
    if resultado["necesita_escalado"]:
        # En producción: llamar a n8n webhook que notifica en Slack al equipo de soporte
        # await notificar_escalado(session_id, entrada.user_id, entrada.pregunta)
        pass
    
    return {
        "session_id": session_id,
        **resultado
    }

@app.post("/feedback")
async def recibir_feedback(fb: FeedbackEntrada):
    guardar_feedback(fb.session_id, fb.util)
    return {"ok": True}

@app.get("/stats")
async def estadisticas():
    import sqlite3
    con = sqlite3.connect("chatbot.db")
    total = con.execute("SELECT COUNT(*) FROM mensajes WHERE rol='user'").fetchone()[0]
    escalados = con.execute("SELECT COUNT(*) FROM mensajes WHERE escalado=1").fetchone()[0]
    util = con.execute("SELECT AVG(util) FROM feedback").fetchone()[0] or 0
    con.close()
    return {
        "total_preguntas": total,
        "escalados": escalados,
        "tasa_escalado_pct": round(escalados/total*100, 1) if total else 0,
        "satisfaccion_pct": round(util * 100, 1)
    }
```

---

## Prueba rápida

```bash
# Arrancar
uvicorn main:app --reload

# Primera pregunta (nueva sesión)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Cómo cancelo mi suscripción?"}'

# Segunda pregunta (misma sesión — el bot recuerda)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID_ANTERIOR", "pregunta": "¿Y perderé mis datos?"}'

# Feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "util": true}'

# Estadísticas
curl http://localhost:8000/stats
```

## Recursos

- [Notebook interactivo](../notebooks/proyectos-integradores/02-chatbot-soporte.ipynb)
- [ChromaDB — in-memory client](https://docs.trychroma.com/)
- [FastAPI — WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
