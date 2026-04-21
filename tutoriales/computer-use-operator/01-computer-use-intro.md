# Computer Use con Claude: Claude como operador de ordenador

## ¿Qué es Computer Use?

Computer Use es una capacidad de Claude que le permite **ver la pantalla y controlar el ordenador** como lo haría un humano: mover el ratón, hacer clic, escribir texto y ejecutar comandos. A diferencia de la RPA tradicional (que depende de selectores CSS o coordenadas fijas), Claude entiende visualmente lo que hay en pantalla y puede adaptarse a interfaces dinámicas.

### ¿Qué lo diferencia de la RPA tradicional?

| RPA tradicional | Computer Use con Claude |
|-----------------|------------------------|
| Selectores CSS/XPath frágiles | Visión semántica de la pantalla |
| Se rompe al cambiar el diseño | Se adapta a cambios visuales |
| Requiere scripting explícito | Entiende instrucciones en lenguaje natural |
| No puede manejar CAPTCHAs ni excepciones | Razona sobre situaciones inesperadas |
| Barato pero rígido | Más flexible pero más costoso |

## Herramientas disponibles

Claude dispone de tres herramientas en modo Computer Use:

| Herramienta | Acciones |
|-------------|----------|
| `computer` | screenshot, click, type, key, mouse_move, scroll, drag |
| `text_editor` | view, create, str_replace, insert en archivos de texto |
| `bash` | Ejecutar comandos de shell |

## Modelos compatibles

- **`claude-sonnet-4-6`** — recomendado por Anthropic para Computer Use (mejor balance velocidad/calidad)
- `claude-opus-4-7` — para tareas más complejas donde la velocidad no es prioritaria

## Arquitectura del bucle de control

```
┌─────────────────────────────────────────────┐
│  1. Instrucción en lenguaje natural          │
│     "Busca el correo de Pedro y archívalo"   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  2. Claude toma un screenshot               │
│     (herramienta: computer → screenshot)    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  3. Claude analiza la imagen y decide       │
│     qué acción tomar                        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  4. Ejecuta la acción (click, type, etc.)   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
              ¿Tarea completada?
              ├── No → volver al paso 2
              └── Sí → respuesta final
```

## Entorno de ejecución

Computer Use **requiere un entorno con display virtual**. Para desarrollo y testing se usa Docker:

```yaml
# docker-compose.yml
version: "3.8"
services:
  computer-use:
    image: anthropics/anthropic-quickstarts:computer-use-demo-latest
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - WIDTH=1920
      - HEIGHT=1080
    ports:
      - "6080:6080"   # noVNC (interfaz web para ver la pantalla)
      - "5900:5900"   # VNC directo
      - "8501:8501"   # Streamlit demo
    volumes:
      - ./output:/home/user/output
```

```bash
# Iniciar entorno
docker-compose up -d

# Ver la pantalla en el navegador
open http://localhost:6080
```

## Primer agente: ejemplo básico

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

def tomar_screenshot_local() -> str:
    """Toma un screenshot de la pantalla y devuelve base64."""
    import subprocess
    subprocess.run(["scrot", "/tmp/screenshot.png"], check=True)
    imagen = Path("/tmp/screenshot.png").read_bytes()
    return base64.standard_b64encode(imagen).decode("utf-8")

def ejecutar_accion(tool_name: str, tool_input: dict) -> str:
    """Ejecuta una acción de computer use y devuelve el resultado."""
    import subprocess

    if tool_name == "computer":
        accion = tool_input.get("action")
        if accion == "screenshot":
            imagen_b64 = tomar_screenshot_local()
            return imagen_b64  # En el bucle real se devuelve como imagen
        elif accion == "left_click":
            x, y = tool_input["coordinate"]
            subprocess.run(["xdotool", "click", "--clearmodifiers", str(x), str(y)])
            return "Click realizado"
        elif accion == "type":
            subprocess.run(["xdotool", "type", "--clearmodifiers", tool_input["text"]])
            return "Texto escrito"
        elif accion == "key":
            subprocess.run(["xdotool", "key", tool_input["key"]])
            return "Tecla pulsada"
    elif tool_name == "bash":
        resultado = subprocess.run(
            tool_input["command"], shell=True, capture_output=True, text=True
        )
        return resultado.stdout + resultado.stderr

    return "Acción desconocida"

def agente_computer_use(instruccion: str, max_iteraciones: int = 20):
    """Bucle de control principal del agente Computer Use."""

    tools = [
        {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080,
            "display_number": 1,
        },
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250429", "name": "str_replace_based_edit_tool"},
    ]

    mensajes = [{"role": "user", "content": instruccion}]

    for iteracion in range(max_iteraciones):
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=mensajes,
            betas=["computer-use-2025-01-24"],
        )

        # Añadir respuesta del asistente al historial
        mensajes.append({"role": "assistant", "content": response.content})

        # Si no hay más tool_use, la tarea ha terminado
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            texto_final = next((b.text for b in response.content if b.type == "text"), "")
            print(f"[Completado en {iteracion+1} iteraciones]\n{texto_final}")
            return texto_final

        # Ejecutar cada herramienta y recoger resultados
        resultados = []
        for tool_use in tool_uses:
            print(f"[Iteración {iteracion+1}] Ejecutando: {tool_use.name} → {tool_use.input.get('action', tool_use.input)}")
            resultado = ejecutar_accion(tool_use.name, tool_use.input)
            resultados.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": resultado,
            })

        mensajes.append({"role": "user", "content": resultados})

    print("Alcanzado el límite de iteraciones")

# Ejemplo de uso
agente_computer_use("Abre el navegador y ve a google.com")
```

## Seguridad básica

```python
URLS_PERMITIDAS = {"google.com", "github.com", "docs.anthropic.com"}

def validar_url(url: str) -> bool:
    from urllib.parse import urlparse
    dominio = urlparse(url).netloc.lower().removeprefix("www.")
    return dominio in URLS_PERMITIDAS
```

## Resumen

- Computer Use permite a Claude controlar visualmente el ordenador
- El bucle es: screenshot → análisis → acción → repetir
- Requiere entorno con display virtual (Docker + xvfb)
- Modelo recomendado: `claude-sonnet-4-6`
- Siempre ejecuta en entornos aislados, nunca directamente en producción
