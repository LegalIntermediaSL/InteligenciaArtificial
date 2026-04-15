# 04 — Memoria a Largo Plazo en Agentes de IA

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 80 min

---

## Índice

1. El problema de la memoria en LLMs
2. Tipos de memoria
3. Memoria con archivos JSON
4. Memoria semántica con embeddings
5. Resumen automático de conversaciones largas
6. Sistema completo de memoria
7. Extensiones sugeridas

---

## 1. El problema de la memoria en LLMs

Los modelos de lenguaje como Claude son, por diseño, **sin estado**. Cada llamada a la API es independiente: el modelo no recuerda la conversación anterior a menos que se la incluyas en el contexto del siguiente mensaje.

**Las dos limitaciones fundamentales:**

**1. Sin persistencia entre sesiones.** Si cierras tu aplicación y la vuelves a abrir, el agente no sabe quién eres, qué has hecho antes ni cuáles son tus preferencias. Cada conversación empieza desde cero.

**2. Ventana de contexto limitada.** Claude tiene una ventana de contexto grande (cientos de miles de tokens), pero no infinita. Una conversación larga eventualmente excede ese límite. Además, enviar cientos de miles de tokens en cada mensaje es caro y lento.

**¿Qué consecuencias tiene esto en la práctica?**

```
Sin memoria:
  Sesión 1: "Me llamo Pedro y soy diseñador gráfico."
  Sesión 2: "¿Cuál es mi profesión?" → Claude no lo sabe.

Con memoria:
  Sesión 1: "Me llamo Pedro y soy diseñador gráfico." → guardado en memoria
  Sesión 2: "¿Cuál es mi profesión?" → Claude consulta la memoria → "Eres diseñador gráfico, Pedro."
```

**La solución:** implementar una capa de memoria externa que el agente pueda leer y escribir entre sesiones. Esta memoria vive fuera del modelo (en disco, en una base de datos) y se inyecta al contexto de forma selectiva.

---

## 2. Tipos de memoria

La neurociencia ofrece un marco conceptual útil para diseñar sistemas de memoria para IA.

### Memoria episódica

Recuerdos de eventos concretos: qué pasó, cuándo y en qué contexto.

- **Analogía humana:** recordar que ayer cenaste con tu jefe y que te habló del nuevo proyecto.
- **En IA:** historial de conversaciones, acciones realizadas, decisiones tomadas.
- **Implementación:** archivos JSON o bases de datos relacionales con timestamps.

### Memoria semántica

Conocimiento general sobre el mundo y sobre el usuario, sin fecha específica.

- **Analogía humana:** saber que París es la capital de Francia, o que a tu amigo Juan no le gusta el café.
- **En IA:** hechos sobre el usuario (nombre, profesión, preferencias), conocimiento del dominio.
- **Implementación:** embeddings vectoriales en ChromaDB, Pinecone o similar.

### Memoria procedimental

Cómo hacer cosas: habilidades y procedimientos.

- **Analogía humana:** saber montar en bicicleta o escribir a máquina sin pensar en cada tecla.
- **En IA:** instrucciones del sistema, plantillas de respuesta, flujos de trabajo memorizados.
- **Implementación:** prompts del sistema persistentes, colecciones de ejemplos few-shot.

### Comparativa

| Tipo | ¿Qué almacena? | Implementación | Recuperación |
|---|---|---|---|
| **Episódica** | Eventos con contexto temporal | JSON, SQLite | Cronológica o por filtro |
| **Semántica** | Hechos, preferencias, conceptos | Embeddings vectoriales | Por similitud semántica |
| **Procedimental** | Instrucciones, flujos | Archivos de texto, prompts | Carga completa al inicio |

---

## 3. Memoria con archivos JSON

La forma más simple de persistencia: guardar hechos del usuario en un archivo JSON entre sesiones.

```python
# memoria_json.py
import json
import anthropic
from datetime import datetime
from pathlib import Path

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"

ARCHIVO_MEMORIA = Path("memoria_usuario.json")


# ---------------------------------------------------------------------------
# Gestión del archivo de memoria
# ---------------------------------------------------------------------------

def cargar_memoria() -> dict:
    """Carga la memoria desde el archivo JSON. Crea una vacía si no existe."""
    if ARCHIVO_MEMORIA.exists():
        return json.loads(ARCHIVO_MEMORIA.read_text(encoding="utf-8"))
    return {
        "hechos_usuario": {},       # nombre, profesión, preferencias, etc.
        "historial_reciente": [],   # últimas N interacciones
        "notas": [],                # notas libres guardadas por el agente
        "creada_el": datetime.now().isoformat(),
        "actualizada_el": datetime.now().isoformat(),
    }


def guardar_memoria(memoria: dict) -> None:
    """Guarda la memoria en el archivo JSON."""
    memoria["actualizada_el"] = datetime.now().isoformat()
    ARCHIVO_MEMORIA.write_text(
        json.dumps(memoria, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def memoria_a_texto(memoria: dict) -> str:
    """Convierte la memoria a un texto legible para incluir en el prompt."""
    partes = []

    if memoria["hechos_usuario"]:
        hechos = "\n".join(f"- {k}: {v}" for k, v in memoria["hechos_usuario"].items())
        partes.append(f"Hechos conocidos del usuario:\n{hechos}")

    if memoria["notas"]:
        notas = "\n".join(f"- {n}" for n in memoria["notas"][-5:])  # últimas 5 notas
        partes.append(f"Notas relevantes:\n{notas}")

    if memoria["historial_reciente"]:
        ultimas = memoria["historial_reciente"][-3:]  # últimas 3 interacciones
        historial = "\n".join(
            f"[{h['fecha']}] Usuario: {h['usuario'][:80]}... → Respuesta: {h['asistente'][:80]}..."
            for h in ultimas
        )
        partes.append(f"Interacciones recientes:\n{historial}")

    return "\n\n".join(partes) if partes else "No hay memoria previa."


def extraer_hechos_nuevos(
    mensaje_usuario: str,
    respuesta_asistente: str,
    memoria_actual: dict,
) -> dict:
    """
    Usa Claude para extraer hechos nuevos del intercambio y actualizar la memoria.
    Devuelve la memoria actualizada.
    """
    contexto_actual = json.dumps(memoria_actual["hechos_usuario"], ensure_ascii=False)

    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=512,
        system=(
            "Eres un sistema de gestión de memoria. Tu tarea es extraer hechos relevantes "
            "sobre el usuario a partir de una conversación y devolver un JSON con los hechos "
            "actualizados. Solo incluye información objetiva y duradera (nombre, profesión, "
            "preferencias, metas). No incluyas el contenido de la conversación. "
            "Si no hay nuevos hechos relevantes, devuelve el JSON original sin cambios. "
            "Devuelve SOLO el JSON, sin explicaciones."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Hechos actuales en memoria:\n{contexto_actual}\n\n"
                    f"Mensaje del usuario: {mensaje_usuario}\n"
                    f"Respuesta del asistente: {respuesta_asistente}\n\n"
                    f"¿Hay hechos nuevos sobre el usuario que añadir o actualizar?"
                ),
            }
        ],
    )

    texto = respuesta.content[0].text.strip()
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:].strip()

    try:
        nuevos_hechos = json.loads(texto)
        memoria_actual["hechos_usuario"] = nuevos_hechos
    except json.JSONDecodeError:
        pass  # si falla el parsing, mantenemos los hechos actuales

    return memoria_actual


# ---------------------------------------------------------------------------
# Agente con memoria JSON
# ---------------------------------------------------------------------------

def agente_con_memoria_json(mensaje_usuario: str) -> str:
    """
    Procesa un mensaje del usuario usando memoria persistente en JSON.
    Actualiza la memoria automáticamente tras cada interacción.
    """
    memoria = cargar_memoria()
    contexto_memoria = memoria_a_texto(memoria)

    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=1024,
        system=(
            "Eres un asistente personal con memoria persistente. "
            "Usas la información de sesiones anteriores para dar respuestas personalizadas.\n\n"
            f"MEMORIA DE SESIONES ANTERIORES:\n{contexto_memoria}"
        ),
        messages=[{"role": "user", "content": mensaje_usuario}],
    )

    texto_respuesta = respuesta.content[0].text

    # Actualizar memoria
    memoria = extraer_hechos_nuevos(mensaje_usuario, texto_respuesta, memoria)
    memoria["historial_reciente"].append({
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "usuario": mensaje_usuario,
        "asistente": texto_respuesta,
    })
    # Mantener solo las últimas 20 interacciones para no crecer indefinidamente
    memoria["historial_reciente"] = memoria["historial_reciente"][-20:]
    guardar_memoria(memoria)

    return texto_respuesta


# ---------------------------------------------------------------------------
# Demostración: simular dos sesiones distintas
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SESIÓN 1 ===")
    respuesta1 = agente_con_memoria_json("Hola, me llamo Ana y trabajo como arquitecta en Barcelona.")
    print(f"Agente: {respuesta1}\n")

    respuesta2 = agente_con_memoria_json("¿Sabes algo de software BIM para arquitectos?")
    print(f"Agente: {respuesta2}\n")

    print("=== (Simular cierre de sesión) ===\n")
    print("=== SESIÓN 2 ===")
    respuesta3 = agente_con_memoria_json("Hola, ¿te acuerdas de quién soy?")
    print(f"Agente: {respuesta3}\n")

    # Mostrar la memoria guardada
    memoria = cargar_memoria()
    print("Memoria guardada:")
    print(json.dumps(memoria["hechos_usuario"], ensure_ascii=False, indent=2))
```

---

## 4. Memoria semántica con embeddings

La memoria JSON es excelente para hechos estructurados, pero tiene limitaciones: si tienes miles de notas, ¿cómo encuentras las relevantes para el mensaje actual? La respuesta son los **embeddings vectoriales**.

Un embedding convierte texto en un vector de números. Textos con significado similar producen vectores similares (cercanos en el espacio vectorial). ChromaDB almacena estos vectores y permite recuperar los más cercanos a una consulta.

```bash
pip install chromadb sentence-transformers
```

```python
# memoria_semantica.py
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from pathlib import Path

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Configuración de ChromaDB con sentence-transformers para embeddings locales
# (gratuito, sin llamadas adicionales a la API)
# ---------------------------------------------------------------------------

DIRECTORIO_DB = "./chroma_memoria"

# Función de embedding: convierte texto en vectores
# multilingual-MiniLM funciona bien en español sin coste adicional
funcion_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# Cliente ChromaDB persistente (guarda en disco)
chroma_cliente = chromadb.PersistentClient(path=DIRECTORIO_DB)

# Colección de recuerdos
coleccion_recuerdos = chroma_cliente.get_or_create_collection(
    name="recuerdos_usuario",
    embedding_function=funcion_embedding,
    metadata={"descripcion": "Memoria semántica del agente"},
)


# ---------------------------------------------------------------------------
# Operaciones de memoria semántica
# ---------------------------------------------------------------------------

def guardar_recuerdo(contenido: str, metadatos: dict = None) -> str:
    """
    Guarda un recuerdo en ChromaDB.
    Devuelve el ID asignado al recuerdo.
    """
    id_recuerdo = f"recuerdo_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    metadatos_completos = {
        "fecha": datetime.now().isoformat(),
        "tipo": "general",
        **(metadatos or {}),
    }

    coleccion_recuerdos.add(
        documents=[contenido],
        ids=[id_recuerdo],
        metadatas=[metadatos_completos],
    )

    print(f"  [Memoria] Guardado recuerdo: '{contenido[:60]}...' (ID: {id_recuerdo})")
    return id_recuerdo


def recuperar_recuerdos_relevantes(consulta: str, n_resultados: int = 5) -> list[dict]:
    """
    Recupera los N recuerdos más relevantes para la consulta.
    La relevancia se mide por similitud semántica (no palabras exactas).
    """
    total = coleccion_recuerdos.count()
    if total == 0:
        return []

    n_resultados = min(n_resultados, total)
    resultados = coleccion_recuerdos.query(
        query_texts=[consulta],
        n_results=n_resultados,
        include=["documents", "metadatas", "distances"],
    )

    recuerdos = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        # Distancia < 0.5 indica alta relevancia (ChromaDB usa distancia coseno)
        recuerdos.append({
            "contenido": doc,
            "fecha": meta.get("fecha", "desconocida"),
            "tipo": meta.get("tipo", "general"),
            "relevancia": round(1 - dist, 3),  # convertir distancia a similitud
        })

    return recuerdos


def recuerdos_a_texto(recuerdos: list[dict]) -> str:
    """Formatea los recuerdos para incluirlos en el prompt."""
    if not recuerdos:
        return "No hay recuerdos relevantes."

    lineas = []
    for r in recuerdos:
        lineas.append(
            f"[{r['fecha'][:10]} | relevancia: {r['relevancia']}] {r['contenido']}"
        )
    return "\n".join(lineas)


def extraer_y_guardar_recuerdos(mensaje: str, respuesta: str) -> None:
    """
    Usa Claude para identificar qué información del intercambio merece ser recordada
    y la guarda en ChromaDB.
    """
    extraccion = cliente.messages.create(
        model=MODELO,
        max_tokens=512,
        system=(
            "Eres un sistema de extracción de memoria. Analiza el intercambio y "
            "devuelve una lista JSON de strings, donde cada string es un hecho concreto "
            "y memorable sobre el usuario o el contexto. "
            "Incluye solo información duradera y útil para futuras conversaciones. "
            "Si no hay nada memorable, devuelve una lista vacía: []. "
            "Devuelve SOLO el JSON array."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Usuario dijo: {mensaje}\nAsistente respondió: {respuesta}",
            }
        ],
    )

    texto = extraccion.content[0].text.strip()
    if texto.startswith("```"):
        partes = texto.split("```")
        for p in partes:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            try:
                hechos = json.loads(p)
                break
            except Exception:
                continue
    else:
        try:
            import json
            hechos = json.loads(texto)
        except Exception:
            hechos = []

    for hecho in hechos:
        if isinstance(hecho, str) and len(hecho) > 10:
            guardar_recuerdo(hecho, {"tipo": "hecho_usuario"})


import json


# ---------------------------------------------------------------------------
# Agente con memoria semántica
# ---------------------------------------------------------------------------

def agente_con_memoria_semantica(mensaje_usuario: str) -> str:
    """
    Agente que recupera recuerdos relevantes para cada mensaje
    y actualiza la memoria con información nueva.
    """
    # Recuperar recuerdos relevantes para este mensaje
    recuerdos = recuperar_recuerdos_relevantes(mensaje_usuario, n_resultados=5)
    contexto_memoria = recuerdos_a_texto(recuerdos)

    print(f"\n[Memoria] Recuerdos recuperados para: '{mensaje_usuario[:50]}'")
    for r in recuerdos:
        print(f"  → [{r['relevancia']}] {r['contenido'][:70]}")

    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=1024,
        system=(
            "Eres un asistente personal con memoria semántica. "
            "Recuerdas hechos relevantes de conversaciones anteriores y los usas "
            "para personalizar tus respuestas.\n\n"
            f"RECUERDOS RELEVANTES:\n{contexto_memoria}"
        ),
        messages=[{"role": "user", "content": mensaje_usuario}],
    )

    texto_respuesta = respuesta.content[0].text

    # Guardar información nueva en memoria
    extraer_y_guardar_recuerdos(mensaje_usuario, texto_respuesta)

    return texto_respuesta


# ---------------------------------------------------------------------------
# Demostración
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Poblar la memoria con algunos hechos iniciales
    print("=== Cargando memoria inicial ===")
    guardar_recuerdo("El usuario se llama Carlos y vive en Madrid.", {"tipo": "perfil"})
    guardar_recuerdo("Carlos trabaja como ingeniero de software en una startup de fintech.", {"tipo": "perfil"})
    guardar_recuerdo("Carlos prefiere respuestas concisas y técnicas.", {"tipo": "preferencia"})
    guardar_recuerdo("Carlos está aprendiendo sobre agentes de IA y LLMs.", {"tipo": "intereses"})

    print("\n=== Conversación con memoria semántica ===\n")

    respuesta = agente_con_memoria_semantica(
        "¿Qué herramientas de Python me recomiendas para mi área de trabajo?"
    )
    print(f"\nAgente: {respuesta}\n")

    respuesta2 = agente_con_memoria_semantica(
        "Estoy pensando en crear un agente que automatice procesos en mi empresa."
    )
    print(f"\nAgente: {respuesta2}")
```

---

## 5. Resumen automático de conversaciones largas

Cuando el historial de una conversación supera un umbral de tokens, hay que comprimir el pasado sin perder información importante.

```python
# compresion_contexto.py
import anthropic
import json
from typing import TypedDict

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"

# Límite de tokens antes de comprimir (ajustar según el modelo y presupuesto)
LIMITE_TOKENS_HISTORIAL = 4000
TOKENS_APROX_POR_CARACTER = 0.25  # estimación: 1 token ≈ 4 caracteres


class Mensaje(TypedDict):
    role: str
    content: str


def estimar_tokens(historial: list[Mensaje]) -> int:
    """Estima el número de tokens en el historial."""
    total_chars = sum(len(m["content"]) for m in historial)
    return int(total_chars * TOKENS_APROX_POR_CARACTER)


def comprimir_historial(historial: list[Mensaje]) -> tuple[str, list[Mensaje]]:
    """
    Comprime el historial antiguo en un resumen y mantiene los mensajes recientes.
    Devuelve (resumen, mensajes_recientes).
    """
    if len(historial) < 4:
        return "", historial

    # Mantener los últimos 4 mensajes intactos (2 pares usuario-asistente)
    mensajes_a_comprimir = historial[:-4]
    mensajes_recientes = historial[-4:]

    if not mensajes_a_comprimir:
        return "", historial

    # Formatear el historial antiguo para el resumen
    texto_historial = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in mensajes_a_comprimir
    )

    # Pedir a Claude que resuma
    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=512,
        system=(
            "Eres un sistema de compresión de memoria. Tu tarea es resumir una "
            "conversación preservando los hechos importantes, decisiones tomadas, "
            "preferencias expresadas y contexto relevante para el futuro. "
            "Escribe el resumen en tercera persona y de forma concisa."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Resume esta conversación preservando lo esencial:\n\n{texto_historial}",
            }
        ],
    )

    resumen = respuesta.content[0].text
    print(f"\n[Compresión] {len(mensajes_a_comprimir)} mensajes → resumen de {len(resumen)} caracteres")

    return resumen, mensajes_recientes


class ConversacionConMemoria:
    """
    Gestor de conversación con compresión automática cuando el historial crece.
    """

    def __init__(self, system_prompt: str = ""):
        self.historial: list[Mensaje] = []
        self.resumen_comprimido: str = ""
        self.system_base = system_prompt

    def _construir_system_prompt(self) -> str:
        """Construye el prompt del sistema incluyendo el resumen de conversaciones previas."""
        partes = [self.system_base] if self.system_base else []

        if self.resumen_comprimido:
            partes.append(
                f"RESUMEN DE LA CONVERSACIÓN ANTERIOR:\n{self.resumen_comprimido}"
            )

        return "\n\n".join(partes) if partes else "Eres un asistente útil."

    def _comprimir_si_necesario(self) -> None:
        """Comprime el historial si supera el límite de tokens."""
        tokens_estimados = estimar_tokens(self.historial)

        if tokens_estimados > LIMITE_TOKENS_HISTORIAL:
            print(f"\n[Auto-compresión] Historial con ~{tokens_estimados} tokens. Comprimiendo...")
            nuevo_resumen, mensajes_recientes = comprimir_historial(self.historial)

            # Combinar resumen anterior con el nuevo
            if self.resumen_comprimido:
                self.resumen_comprimido = (
                    f"{self.resumen_comprimido}\n\n[Actualización]: {nuevo_resumen}"
                )
            else:
                self.resumen_comprimido = nuevo_resumen

            self.historial = mensajes_recientes
            print(f"[Auto-compresión] Historial reducido a {len(self.historial)} mensajes.")

    def chat(self, mensaje_usuario: str) -> str:
        """Procesa un mensaje y devuelve la respuesta, comprimiendo si es necesario."""
        self.historial.append({"role": "user", "content": mensaje_usuario})

        respuesta = cliente.messages.create(
            model=MODELO,
            max_tokens=1024,
            system=self._construir_system_prompt(),
            messages=self.historial,
        )

        texto_respuesta = respuesta.content[0].text
        self.historial.append({"role": "assistant", "content": texto_respuesta})

        # Comprimir si el historial ha crecido demasiado
        self._comprimir_si_necesario()

        return texto_respuesta

    def estado(self) -> dict:
        return {
            "mensajes_en_historial": len(self.historial),
            "tokens_estimados": estimar_tokens(self.historial),
            "tiene_resumen": bool(self.resumen_comprimido),
            "longitud_resumen": len(self.resumen_comprimido),
        }


if __name__ == "__main__":
    conversacion = ConversacionConMemoria(
        system_prompt="Eres un tutor de programación Python paciente y detallado."
    )

    temas = [
        "¿Qué es una lista en Python?",
        "¿Cómo funciona la comprensión de listas?",
        "Explícame los diccionarios.",
        "¿Para qué sirven los sets?",
        "¿Qué diferencia hay entre tuplas y listas?",
        "Explícame las funciones lambda.",
        "¿Qué es un generador?",
        "¿Para qué sirve el decorador @property?",
    ]

    for pregunta in temas:
        print(f"\nUsuario: {pregunta}")
        respuesta = conversacion.chat(pregunta)
        print(f"Tutor: {respuesta[:150]}...")
        print(f"Estado: {conversacion.estado()}")
```

---

## 6. Sistema completo de memoria

La clase `MemoryManager` combina los tres enfoques anteriores en un sistema cohesivo y listo para usar en producción.

```python
# memory_manager.py
import json
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from pathlib import Path
from typing import Optional


MODELO = "claude-sonnet-4-6"
LIMITE_TOKENS = 6000
TOKENS_POR_CHAR = 0.25


class MemoryManager:
    """
    Sistema completo de memoria que combina:
    - Memoria episódica: hechos estructurados en JSON
    - Memoria semántica: recuerdos indexados con embeddings en ChromaDB
    - Compresión automática: resumen de historial cuando supera el límite

    Uso:
        mm = MemoryManager(usuario_id="usuario_123")
        respuesta = mm.chat("¿Recuerdas cómo me llamo?")
    """

    def __init__(self, usuario_id: str = "default", directorio_base: str = "./memoria"):
        self.usuario_id = usuario_id
        self.cliente = anthropic.Anthropic()

        # Directorios de almacenamiento
        self.directorio = Path(directorio_base) / usuario_id
        self.directorio.mkdir(parents=True, exist_ok=True)
        self.archivo_episodica = self.directorio / "episodica.json"
        self.archivo_chroma = str(self.directorio / "chroma")

        # Estado en memoria RAM (para la sesión actual)
        self.historial_sesion: list[dict] = []
        self.resumen_comprimido: str = ""

        # Cargar memoria episódica
        self.memoria_episodica = self._cargar_episodica()

        # Inicializar ChromaDB para memoria semántica
        self.chroma = chromadb.PersistentClient(path=self.archivo_chroma)
        self.funcion_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.coleccion = self.chroma.get_or_create_collection(
            name="recuerdos",
            embedding_function=self.funcion_embedding,
        )

        print(f"[MemoryManager] Iniciado para usuario: {usuario_id}")
        print(f"  Hechos episódicos: {len(self.memoria_episodica.get('hechos', {}))}")
        print(f"  Recuerdos semánticos: {self.coleccion.count()}")

    # -----------------------------------------------------------------------
    # Memoria episódica (JSON)
    # -----------------------------------------------------------------------

    def _cargar_episodica(self) -> dict:
        if self.archivo_episodica.exists():
            return json.loads(self.archivo_episodica.read_text(encoding="utf-8"))
        return {
            "usuario_id": self.usuario_id,
            "hechos": {},
            "sesiones": [],
            "creada": datetime.now().isoformat(),
        }

    def _guardar_episodica(self) -> None:
        self.memoria_episodica["ultima_actualizacion"] = datetime.now().isoformat()
        self.archivo_episodica.write_text(
            json.dumps(self.memoria_episodica, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def actualizar_hecho(self, clave: str, valor: str) -> None:
        """Guarda un hecho estructurado sobre el usuario."""
        self.memoria_episodica["hechos"][clave] = valor
        self._guardar_episodica()

    # -----------------------------------------------------------------------
    # Memoria semántica (ChromaDB)
    # -----------------------------------------------------------------------

    def recordar(self, contenido: str, tipo: str = "general") -> None:
        """Guarda un recuerdo en la base de datos vectorial."""
        id_recuerdo = f"{self.usuario_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.coleccion.add(
            documents=[contenido],
            ids=[id_recuerdo],
            metadatas=[{"tipo": tipo, "fecha": datetime.now().isoformat()}],
        )

    def recuperar(self, consulta: str, n: int = 4) -> list[str]:
        """Recupera los N recuerdos más relevantes por similitud semántica."""
        total = self.coleccion.count()
        if total == 0:
            return []
        resultados = self.coleccion.query(
            query_texts=[consulta],
            n_results=min(n, total),
            include=["documents", "distances"],
        )
        # Filtrar por relevancia mínima (distancia coseno < 0.6)
        recuerdos = []
        for doc, dist in zip(resultados["documents"][0], resultados["distances"][0]):
            if dist < 0.6:
                recuerdos.append(doc)
        return recuerdos

    # -----------------------------------------------------------------------
    # Compresión de historial
    # -----------------------------------------------------------------------

    def _tokens_estimados(self) -> int:
        return int(sum(len(m["content"]) for m in self.historial_sesion) * TOKENS_POR_CHAR)

    def _comprimir_historial(self) -> None:
        if len(self.historial_sesion) < 4:
            return

        a_comprimir = self.historial_sesion[:-4]
        self.historial_sesion = self.historial_sesion[-4:]

        texto = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in a_comprimir)

        respuesta = self.cliente.messages.create(
            model=MODELO,
            max_tokens=400,
            system="Resume la conversación de forma concisa preservando hechos y decisiones clave.",
            messages=[{"role": "user", "content": texto}],
        )

        nuevo_resumen = respuesta.content[0].text
        self.resumen_comprimido = (
            f"{self.resumen_comprimido}\n\n{nuevo_resumen}" if self.resumen_comprimido
            else nuevo_resumen
        )

        # También guardar el resumen como recuerdo semántico
        self.recordar(nuevo_resumen, tipo="resumen_sesion")
        print(f"[MemoryManager] Historial comprimido. Resumen: {len(nuevo_resumen)} chars.")

    # -----------------------------------------------------------------------
    # Construcción del contexto
    # -----------------------------------------------------------------------

    def _construir_system_prompt(self, consulta_actual: str) -> str:
        partes = [
            "Eres un asistente personal con memoria persistente entre sesiones.",
            "Usas la información de la memoria para personalizar tus respuestas.",
        ]

        # Hechos episódicos
        if self.memoria_episodica["hechos"]:
            hechos = "\n".join(
                f"- {k}: {v}" for k, v in self.memoria_episodica["hechos"].items()
            )
            partes.append(f"HECHOS DEL USUARIO:\n{hechos}")

        # Recuerdos semánticos relevantes
        recuerdos = self.recuperar(consulta_actual)
        if recuerdos:
            partes.append("RECUERDOS RELEVANTES:\n" + "\n".join(f"- {r}" for r in recuerdos))

        # Resumen de conversaciones anteriores
        if self.resumen_comprimido:
            partes.append(f"RESUMEN DE CONVERSACIONES ANTERIORES:\n{self.resumen_comprimido}")

        return "\n\n".join(partes)

    # -----------------------------------------------------------------------
    # Actualización automática de memoria tras cada intercambio
    # -----------------------------------------------------------------------

    def _actualizar_memoria_automaticamente(self, mensaje: str, respuesta: str) -> None:
        """Extrae y guarda hechos nuevos del intercambio."""
        extraccion = self.cliente.messages.create(
            model=MODELO,
            max_tokens=300,
            system=(
                "Extrae hechos memorables sobre el usuario del intercambio. "
                "Devuelve un JSON: {\"hechos\": {\"clave\": \"valor\"}, \"recuerdos\": [\"string\"]}. "
                "hechos: información estructurada (nombre, profesión, ciudad...). "
                "recuerdos: frases cortas con información relevante para el futuro. "
                "Si no hay nada nuevo, devuelve {\"hechos\": {}, \"recuerdos\": []}."
            ),
            messages=[
                {"role": "user", "content": f"Usuario: {mensaje}\nAsistente: {respuesta}"}
            ],
        )

        texto = extraccion.content[0].text.strip()
        if "```" in texto:
            for parte in texto.split("```"):
                parte = parte.strip().lstrip("json").strip()
                try:
                    datos = json.loads(parte)
                    break
                except Exception:
                    continue
        else:
            try:
                datos = json.loads(texto)
            except Exception:
                datos = {"hechos": {}, "recuerdos": []}

        # Actualizar hechos episódicos
        for clave, valor in datos.get("hechos", {}).items():
            self.actualizar_hecho(clave, str(valor))

        # Guardar recuerdos semánticos
        for recuerdo in datos.get("recuerdos", []):
            if isinstance(recuerdo, str) and len(recuerdo) > 15:
                self.recordar(recuerdo, tipo="auto_extraido")

    # -----------------------------------------------------------------------
    # Interfaz principal de chat
    # -----------------------------------------------------------------------

    def chat(self, mensaje: str) -> str:
        """
        Procesa un mensaje con soporte completo de memoria.
        Recupera contexto relevante, responde y actualiza la memoria.
        """
        self.historial_sesion.append({"role": "user", "content": mensaje})

        system_prompt = self._construir_system_prompt(mensaje)

        respuesta = self.cliente.messages.create(
            model=MODELO,
            max_tokens=1024,
            system=system_prompt,
            messages=self.historial_sesion,
        )

        texto_respuesta = respuesta.content[0].text
        self.historial_sesion.append({"role": "assistant", "content": texto_respuesta})

        # Comprimir si es necesario
        if self._tokens_estimados() > LIMITE_TOKENS:
            self._comprimir_historial()

        # Actualizar memoria en segundo plano
        self._actualizar_memoria_automaticamente(mensaje, texto_respuesta)

        return texto_respuesta

    def estadisticas(self) -> dict:
        """Devuelve estadísticas del sistema de memoria."""
        return {
            "usuario": self.usuario_id,
            "hechos_episodicos": len(self.memoria_episodica["hechos"]),
            "recuerdos_semanticos": self.coleccion.count(),
            "mensajes_sesion_actual": len(self.historial_sesion),
            "tokens_estimados_sesion": self._tokens_estimados(),
            "tiene_resumen_comprimido": bool(self.resumen_comprimido),
        }


# ---------------------------------------------------------------------------
# Demostración completa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DEMOSTRACIÓN: Sistema completo de memoria")
    print("=" * 60)

    # Crear el gestor de memoria para un usuario específico
    mm = MemoryManager(usuario_id="usuario_demo")

    # Simular una primera sesión
    print("\n--- Primera sesión ---")

    r1 = mm.chat("Hola, me llamo Laura. Soy médica especialista en neurología.")
    print(f"Asistente: {r1[:200]}...\n")

    r2 = mm.chat("Estoy interesada en aplicaciones de IA en diagnóstico médico.")
    print(f"Asistente: {r2[:200]}...\n")

    r3 = mm.chat("¿Sabes algo sobre el uso de LLMs en informes clínicos?")
    print(f"Asistente: {r3[:200]}...\n")

    print(f"\nEstadísticas tras primera sesión: {mm.estadisticas()}")

    # Simular segunda sesión con un nuevo MemoryManager (como si se reiniciara la app)
    print("\n--- Segunda sesión (nueva instancia, misma memoria) ---")
    mm2 = MemoryManager(usuario_id="usuario_demo")

    r4 = mm2.chat("Buenos días, ¿recuerdas cuál es mi especialidad médica?")
    print(f"Asistente: {r4[:300]}")

    r5 = mm2.chat("¿Qué proyectos de IA en salud me recomendarías explorar dado mi perfil?")
    print(f"\nAsistente: {r5[:300]}")

    print(f"\nEstadísticas finales: {mm2.estadisticas()}")
```

---

## 7. Extensiones sugeridas

- **Memoria compartida entre agentes:** conecta varios agentes al mismo `MemoryManager` para que compartan conocimiento. Un agente investigador guarda lo que aprende; un agente redactor lo recupera.
- **Memoria con PostgreSQL + pgvector:** sustituye ChromaDB por PostgreSQL con la extensión pgvector para producción con alta concurrencia y backups automáticos.
- **Olvido selectivo:** implementa un sistema de "decay" donde los recuerdos más antiguos y menos accedidos se comprimen o eliminan para mantener la base de datos eficiente.
- **Perfil de usuario enriquecido:** además de hechos, guarda el estilo de comunicación preferido, el nivel técnico y los temas de mayor interés para personalizar el tono de las respuestas.
- **Exportación e importación:** permite al usuario exportar su memoria completa en JSON y restaurarla en otro dispositivo o cuenta.
- **Memoria multiusuario:** escala el `MemoryManager` a un sistema multi-tenant con aislamiento entre usuarios, ideado para aplicaciones SaaS.
- **Evaluación de la memoria:** mide la efectividad de la memoria comprobando si el agente recuerda correctamente hechos guardados en sesiones anteriores.

---

**Fin del bloque** — Has completado el Bloque 9: Agentes Avanzados.

[Volver al índice del bloque](./README.md)
