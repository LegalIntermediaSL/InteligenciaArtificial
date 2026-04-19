# Agentes con Memoria: Continuidad entre Sesiones

## Tipos de memoria en agentes

Los agentes sin memoria olvidan todo al terminar una sesión. Para construir
asistentes que **recuerden al usuario y acumulen conocimiento**, implementamos
distintos tipos de memoria:

```
MEMORIA A CORTO PLAZO (in-context)
    └── El historial de mensajes dentro de la sesión actual

MEMORIA A MEDIO PLAZO (caché)
    └── Prompt Caching para reutilizar contexto entre llamadas cercanas

MEMORIA A LARGO PLAZO (persistente)
    ├── Semántica: ChromaDB / pgvector → "recuerdos" buscables por similitud
    ├── Episódica: JSON / SQLite → historial de conversaciones pasadas
    └── Factual: diccionario / Redis → preferencias, datos del usuario
```

## Memoria episódica: guardar y recuperar conversaciones

```python
import json
import os
from datetime import datetime
from pathlib import Path

DIRECTORIO_MEMORIA = Path("memoria_agente")
DIRECTORIO_MEMORIA.mkdir(exist_ok=True)

def guardar_sesion(user_id: str, mensajes: list, resumen: str = ""):
    """Guarda los mensajes de una sesión en disco."""
    sesion = {
        "user_id": user_id,
        "fecha": datetime.now().isoformat(),
        "resumen": resumen,
        "mensajes": mensajes
    }
    ruta = DIRECTORIO_MEMORIA / f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(sesion, f, ensure_ascii=False, indent=2)
    return str(ruta)

def cargar_sesiones_recientes(user_id: str, n: int = 5) -> list:
    """Carga las N sesiones más recientes del usuario."""
    archivos = sorted(
        DIRECTORIO_MEMORIA.glob(f"{user_id}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:n]

    sesiones = []
    for archivo in archivos:
        with open(archivo, encoding="utf-8") as f:
            sesiones.append(json.load(f))
    return sesiones

def resumir_sesion(mensajes: list) -> str:
    """Usa Claude para resumir la sesión en pocas líneas."""
    import anthropic
    client = anthropic.Anthropic()

    texto = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in mensajes
        if isinstance(m.get("content"), str)
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": f"Resume en 2-3 líneas los puntos clave de esta conversación:\n{texto[:2000]}"
        }]
    )
    return resp.content[0].text
```

## Memoria semántica con embeddings

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Inicializar base de datos de memoria semántica
modelo_embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
db = chromadb.PersistentClient(path="./memoria_semantica")
coleccion_memoria = db.get_or_create_collection("recuerdos_usuario")

def guardar_recuerdo(user_id: str, contenido: str, tipo: str = "conversacion"):
    """Guarda un recuerdo como embedding para búsqueda semántica."""
    embedding = modelo_embedding.encode([contenido])[0].tolist()
    coleccion_memoria.add(
        ids=[f"{user_id}_{datetime.now().timestamp()}"],
        embeddings=[embedding],
        documents=[contenido],
        metadatas=[{"user_id": user_id, "tipo": tipo, "fecha": datetime.now().isoformat()}]
    )

def recuperar_recuerdos_relevantes(user_id: str, consulta: str, n: int = 3) -> list:
    """Recupera los N recuerdos más relevantes para una consulta."""
    embedding_consulta = modelo_embedding.encode([consulta])[0].tolist()
    resultados = coleccion_memoria.query(
        query_embeddings=[embedding_consulta],
        n_results=n,
        where={"user_id": user_id}
    )
    return resultados["documents"][0] if resultados["documents"] else []
```

## Memoria factual: preferencias y datos del usuario

```python
class MemoriaFactual:
    """Almacén de hechos y preferencias del usuario."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.ruta = DIRECTORIO_MEMORIA / f"{user_id}_perfil.json"
        self._datos = self._cargar()

    def _cargar(self) -> dict:
        if self.ruta.exists():
            with open(self.ruta, encoding="utf-8") as f:
                return json.load(f)
        return {"preferencias": {}, "datos": {}, "historial_temas": []}

    def _guardar(self):
        with open(self.ruta, "w", encoding="utf-8") as f:
            json.dump(self._datos, f, ensure_ascii=False, indent=2)

    def actualizar(self, clave: str, valor, categoria: str = "datos"):
        self._datos.setdefault(categoria, {})[clave] = valor
        self._guardar()

    def obtener(self, clave: str, categoria: str = "datos", default=None):
        return self._datos.get(categoria, {}).get(clave, default)

    def agregar_tema(self, tema: str):
        self._datos["historial_temas"].append({"tema": tema, "fecha": datetime.now().isoformat()})
        self._guardar()

    def resumen_perfil(self) -> str:
        datos = self._datos.get("datos", {})
        prefs = self._datos.get("preferencias", {})
        temas = [t["tema"] for t in self._datos.get("historial_temas", [])[-5:]]
        return (
            f"Usuario: {datos}\n"
            f"Preferencias: {prefs}\n"
            f"Temas recientes: {', '.join(temas)}"
        )
```

## Agente con memoria completa

```python
import anthropic

client = anthropic.Anthropic()

class AgenteConMemoria:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.historial_sesion = []
        self.memoria_factual = MemoriaFactual(user_id)

    def _construir_system(self) -> str:
        """System prompt enriquecido con memoria del usuario."""
        perfil = self.memoria_factual.resumen_perfil()
        recuerdos = recuperar_recuerdos_relevantes(
            self.user_id,
            " ".join(m["content"] for m in self.historial_sesion[-3:]
                     if isinstance(m.get("content"), str))
        ) if self.historial_sesion else []

        return f"""Eres un asistente personal que recuerda al usuario entre sesiones.

PERFIL DEL USUARIO:
{perfil}

RECUERDOS RELEVANTES DE SESIONES ANTERIORES:
{chr(10).join(f'- {r}' for r in recuerdos) if recuerdos else 'Sin recuerdos previos'}

INSTRUCCIONES:
- Usa el perfil y recuerdos para personalizar tus respuestas
- Si aprendes algo nuevo sobre el usuario (nombre, preferencias, datos), recuérdalo
- Sé consistente con lo que has dicho en sesiones anteriores"""

    def _extraer_y_guardar_hechos(self, mensaje_usuario: str, respuesta: str):
        """Claude extrae hechos del usuario y los guarda en memoria factual."""
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Del siguiente intercambio, extrae datos del usuario para recordar.
Si no hay datos nuevos, responde con {{}}.
Responde SOLO con JSON: {{"nombre": "...", "empresa": "...", "preferencia": "...", "dato_clave": "..."}}
Solo incluye los campos que aparezcan explícitamente.

Usuario dijo: {mensaje_usuario}
Asistente respondió: {respuesta[:200]}"""
            }]
        ).content[0].text.strip()

        try:
            if "```" in resp:
                resp = resp.split("```")[1].lstrip("json")
            hechos = json.loads(resp)
            for clave, valor in hechos.items():
                if valor:
                    self.memoria_factual.actualizar(clave, valor)
        except (json.JSONDecodeError, KeyError):
            pass

    def chat(self, mensaje: str) -> str:
        self.historial_sesion.append({"role": "user", "content": mensaje})

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=self._construir_system(),
            messages=self.historial_sesion
        )

        respuesta = resp.content[0].text
        self.historial_sesion.append({"role": "assistant", "content": respuesta})

        # Guardar hechos y recuerdos en background
        self._extraer_y_guardar_hechos(mensaje, respuesta)
        guardar_recuerdo(
            self.user_id,
            f"El usuario dijo: {mensaje}. Respuesta: {respuesta[:200]}"
        )

        return respuesta

    def finalizar_sesion(self):
        """Guarda resumen de la sesión al terminar."""
        if self.historial_sesion:
            resumen = resumir_sesion(self.historial_sesion)
            guardar_sesion(self.user_id, self.historial_sesion, resumen)
            print(f"Sesión guardada. Resumen: {resumen}")

# Uso
agente = AgenteConMemoria("usuario_001")

print(agente.chat("Hola, soy Ana García, directora de operaciones en LogiTech SA"))
print(agente.chat("Necesito automatizar los informes semanales de mi equipo"))
print(agente.chat("¿Recuerdas mi nombre y empresa?"))

agente.finalizar_sesion()
```

## Compresión de contexto para sesiones largas

```python
def comprimir_historial(historial: list, max_tokens: int = 3000) -> list:
    """Comprime el historial cuando supera el límite de tokens."""
    # Estimar tokens (~4 chars por token)
    tokens_est = sum(len(str(m.get("content", ""))) // 4 for m in historial)

    if tokens_est <= max_tokens:
        return historial

    # Mantener los últimos 4 turnos completos
    turnos_recientes = historial[-8:]

    # Comprimir el resto con Claude
    historial_antiguo = historial[:-8]
    texto_antiguo = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in historial_antiguo
        if isinstance(m.get("content"), str)
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Resume en 3-4 puntos clave esta conversación:\n{texto_antiguo[:2000]}"
        }]
    )
    resumen = resp.content[0].text

    historial_comprimido = [
        {"role": "user", "content": f"[CONTEXTO PREVIO RESUMIDO]: {resumen}"},
        {"role": "assistant", "content": "Entendido. Continuemos."}
    ] + turnos_recientes

    return historial_comprimido
```

## Recursos

- [Notebook interactivo](../notebooks/agent-sdk/03-agentes-con-memoria.ipynb)
- [ChromaDB Docs](https://docs.trychroma.com)
