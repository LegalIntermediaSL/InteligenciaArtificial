# Automatización web con Computer Use: scraping visual y formularios

## ¿Cuándo usar Computer Use en lugar de Playwright/Selenium?

| Situación | Herramienta recomendada |
|-----------|------------------------|
| SPA con DOM bien estructurado | Playwright/Selenium |
| Formulario con reCAPTCHA visual | Computer Use |
| Portal ERP legacy sin API | Computer Use |
| Web scraping estándar | BeautifulSoup + requests |
| Aplicación que requiere razonamiento visual | Computer Use |
| Flujo que cambia el diseño frecuentemente | Computer Use |

Computer Use no es la solución por defecto para todo scraping — es la solución cuando las herramientas tradicionales fallan.

## Scraping visual de un portal legacy

```python
import anthropic
import base64
import subprocess
from pathlib import Path

client = anthropic.Anthropic()

def screenshot_a_base64(ruta: str = "/tmp/screenshot.png") -> str:
    subprocess.run(["scrot", ruta], check=True)
    return base64.standard_b64encode(Path(ruta).read_bytes()).decode()

def extraer_datos_visual(url: str, campo_a_extraer: str) -> str:
    """
    Navega a una URL y extrae un campo usando visión.
    Útil para portales que no exponen API ni DOM limpio.
    """
    tools = [
        {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": 1280,
            "display_height_px": 800,
            "display_number": 1,
        },
        {"type": "bash_20250124", "name": "bash"},
    ]

    instruccion = f"""
    1. Abre el navegador y ve a {url}
    2. Espera a que cargue completamente
    3. Localiza y extrae: {campo_a_extraer}
    4. Devuelve solo el valor encontrado, sin texto adicional
    """

    mensajes = [{"role": "user", "content": instruccion}]

    for _ in range(15):
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=tools,
            messages=mensajes,
            betas=["computer-use-2025-01-24"],
        )

        mensajes.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return next((b.text for b in response.content if b.type == "text"), "")

        resultados = []
        for tu in tool_uses:
            if tu.name == "computer" and tu.input.get("action") == "screenshot":
                img = screenshot_a_base64()
                resultados.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}],
                })
            elif tu.name == "bash":
                r = subprocess.run(tu.input["command"], shell=True, capture_output=True, text=True)
                resultados.append({"type": "tool_result", "tool_use_id": tu.id, "content": r.stdout + r.stderr})
            else:
                resultados.append({"type": "tool_result", "tool_use_id": tu.id, "content": "OK"})

        mensajes.append({"role": "user", "content": resultados})

    return "Tiempo máximo alcanzado"
```

## Rellenar formularios complejos

```python
def rellenar_formulario(url: str, datos: dict[str, str]) -> bool:
    """
    Rellena y envía un formulario web.
    datos: {'campo': 'valor', ...}
    """
    instruccion = f"""
    Ve a {url} y rellena el formulario con estos datos:
    {chr(10).join(f'- {k}: {v}' for k, v in datos.items())}
    
    Una vez rellenado, haz clic en el botón de envío.
    Si aparece algún error de validación, intenta corregirlo.
    Cuando el formulario se envíe con éxito, responde "ENVIADO".
    Si no puedes enviarlo después de 3 intentos, responde "ERROR: motivo".
    """

    # [bucle de control estándar]
    # ...
    return True  # o False según resultado


# Ejemplo: formulario de contacto
exito = rellenar_formulario(
    "https://ejemplo.com/contacto",
    {
        "nombre": "María García",
        "email": "maria@empresa.com",
        "asunto": "Solicitud de información",
        "mensaje": "Me gustaría recibir más información sobre sus servicios.",
    }
)
```

## Manejo de páginas dinámicas y SPAs

```python
def esperar_elemento_visual(descripcion: str, timeout_iteraciones: int = 10) -> bool:
    """
    Espera hasta que un elemento visual aparezca en pantalla.
    Devuelve True si aparece, False si se agota el tiempo.
    """
    tools = [{"type": "computer_20250124", "name": "computer", "display_width_px": 1280, "display_height_px": 800, "display_number": 1}]

    for i in range(timeout_iteraciones):
        img = screenshot_a_base64()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku para verificaciones rápidas
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                    {"type": "text", "text": f"¿Está visible '{descripcion}' en la pantalla? Responde solo SÍ o NO."},
                ],
            }],
        )
        if "SÍ" in response.content[0].text.upper():
            return True

        import time
        time.sleep(1)  # Esperar antes de reintentar

    return False
```

## Extracción estructurada de tablas visuales

```python
def extraer_tabla_visual(descripcion_tabla: str) -> list[dict]:
    """Extrae datos de una tabla que solo existe visualmente (no en el DOM)."""
    img = screenshot_a_base64()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                {"type": "text", "text": f"""
Extrae los datos de la tabla '{descripcion_tabla}' en la pantalla.
Devuelve SOLO un JSON array con los datos de cada fila.
Ejemplo: [{{"columna1": "valor1", "columna2": "valor2"}}]
"""},
            ],
        }],
    )

    import json
    try:
        texto = response.content[0].text
        # Extraer JSON del texto
        inicio = texto.find("[")
        fin = texto.rfind("]") + 1
        return json.loads(texto[inicio:fin])
    except Exception:
        return []
```

## Patrones de retry y manejo de errores

```python
import time
from typing import Callable, TypeVar

T = TypeVar("T")

def con_reintentos(
    fn: Callable[[], T],
    max_intentos: int = 3,
    espera_base: float = 2.0,
    descripcion: str = "operación",
) -> T:
    """Ejecuta una función con reintentos exponenciales."""
    ultimo_error = None
    for intento in range(max_intentos):
        try:
            return fn()
        except Exception as e:
            ultimo_error = e
            espera = espera_base ** intento
            print(f"[{descripcion}] Intento {intento+1} fallido: {e}. Esperando {espera}s...")
            time.sleep(espera)
    raise RuntimeError(f"[{descripcion}] Fallido tras {max_intentos} intentos: {ultimo_error}")
```

## Resumen

- Computer Use es ideal para webs legacy, SPAs complejas y formularios con CAPTCHA
- Para scraping estándar, usa BeautifulSoup o Playwright (más barato y rápido)
- Los patrones clave son: screenshot → análisis → acción → verificación
- Usa Haiku para verificaciones simples (¿está visible X?) para reducir costes
- Implementa siempre retry logic para manejar cargas lentas y errores transitorios
