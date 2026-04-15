# 03 — Computer Use

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 70 min

---

## Índice

1. Qué es Computer Use y cómo funciona
2. Habilitar Computer Use en la API
3. Implementación del bucle de control
4. Caso práctico — automatizar un formulario web
5. Caso práctico — agente de scraping visual
6. Seguridad y consideraciones éticas
7. Alternativas ligeras
8. Extensiones sugeridas

---

## 1. Qué es Computer Use y cómo funciona

Computer Use es una capacidad de Claude que le permite **ver** lo que hay en la pantalla (a través de capturas de pantalla) y **actuar** sobre ella (hacer clic, escribir, desplazarse). Es como darle un ratón y un teclado a la IA.

Esto es fundamentalmente distinto a los agentes con herramientas convencionales: en lugar de llamar a una API bien definida, Claude interactúa con interfaces visuales diseñadas para humanos.

**¿Por qué esto importa?**

La mayoría del software empresarial no tiene API. Hay miles de aplicaciones legacy, portales de gobierno, sistemas internos y webs que solo pueden usarse a través de su interfaz gráfica. Computer Use permite automatizar cualquiera de ellos.

**Las herramientas disponibles:**

| Herramienta | Descripción | Parámetros clave |
|---|---|---|
| `screenshot` | Captura la pantalla actual | — |
| `left_click` | Clic izquierdo en coordenadas | `coordinate: [x, y]` |
| `right_click` | Clic derecho | `coordinate: [x, y]` |
| `double_click` | Doble clic | `coordinate: [x, y]` |
| `type` | Escribe texto | `text: "..."` |
| `key` | Pulsa una tecla o combinación | `text: "Return"`, `"ctrl+a"` |
| `scroll` | Desplaza la rueda del ratón | `coordinate`, `direction`, `amount` |
| `mouse_move` | Mueve el ratón sin clic | `coordinate: [x, y]` |
| `left_click_drag` | Arrastra desde un punto a otro | `start_coordinate`, `coordinate` |

**Cómo percibe la pantalla Claude:**

Claude recibe capturas de pantalla como imágenes (base64) y las analiza con su capacidad de visión. Identifica elementos de la UI: botones, campos de texto, menús, enlaces, y calcula sus coordenadas aproximadas para interactuar con ellos.

---

## 2. Habilitar Computer Use en la API

Computer Use está disponible en Claude a través de la API con un parámetro `betas` especial. No requiere configuración adicional más allá de una clave de API válida.

```python
import anthropic
import base64
from pathlib import Path

cliente = anthropic.Anthropic()

# Definición de la herramienta de computadora
# Claude espera estas dimensiones para escalar correctamente las coordenadas
ANCHO_PANTALLA = 1280
ALTO_PANTALLA = 800

HERRAMIENTA_COMPUTADORA = {
    "type": "computer_20241022",  # versión de la herramienta
    "name": "computer",
    "display_width_px": ANCHO_PANTALLA,
    "display_height_px": ALTO_PANTALLA,
    "display_number": 1,  # número de monitor (1 = principal)
}


def enviar_con_screenshot(screenshot_base64: str, instruccion: str):
    """
    Envía una captura de pantalla a Claude junto con una instrucción.
    Devuelve la respuesta completa del modelo.
    """
    respuesta = cliente.beta.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        tools=[HERRAMIENTA_COMPUTADORA],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": instruccion,
                    },
                ],
            }
        ],
        betas=["computer-use-2024-10-22"],  # habilitar Computer Use
    )
    return respuesta


def cargar_screenshot_local(ruta: str) -> str:
    """Carga una imagen PNG local y la devuelve en base64."""
    datos = Path(ruta).read_bytes()
    return base64.standard_b64encode(datos).decode("utf-8")


# Ejemplo de uso básico: mostrar qué haría Claude con una captura de pantalla
if __name__ == "__main__":
    # Para probar sin un entorno real, usamos una imagen de ejemplo
    # En producción, esto sería una captura de pantalla real
    import urllib.request
    import tempfile
    import os

    # Descargamos una imagen de prueba simple
    url_imagen_prueba = "https://via.placeholder.com/1280x800.png"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        try:
            urllib.request.urlretrieve(url_imagen_prueba, tmp.name)
            screenshot_b64 = cargar_screenshot_local(tmp.name)
        except Exception:
            # Si no hay conexión, crear una imagen PNG mínima válida (1x1 pixel blanco)
            import struct
            import zlib
            def crear_png_minimo():
                def chunk(tipo, datos):
                    c = tipo + datos
                    return struct.pack(">I", len(datos)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
                png = b"\x89PNG\r\n\x1a\n"
                png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                png += chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
                png += chunk(b"IEND", b"")
                return base64.standard_b64encode(png).decode()
            screenshot_b64 = crear_png_minimo()
        finally:
            os.unlink(tmp.name)

    respuesta = enviar_con_screenshot(
        screenshot_b64,
        "Describe lo que ves en la pantalla y qué acción realizarías para abrir el navegador.",
    )

    for bloque in respuesta.content:
        if hasattr(bloque, "text"):
            print("Claude dice:", bloque.text)
        elif bloque.type == "tool_use":
            print(f"Claude quiere ejecutar: {bloque.name}")
            print(f"Con parámetros: {bloque.input}")
```

---

## 3. Implementación del bucle de control

El bucle de control es el corazón de Computer Use: captura pantalla → envía a Claude → interpreta la acción → ejecuta → repite.

```python
import anthropic
import base64
import time
from typing import Optional
import subprocess
import platform

# Para capturas de pantalla reales necesitas una de estas bibliotecas:
# pip install pyautogui pillow   (macOS/Windows/Linux con escritorio)
# pip install playwright          (para navegador en headless)

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"

ANCHO = 1280
ALTO = 800

HERRAMIENTAS_COMPUTER_USE = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": ANCHO,
        "display_height_px": ALTO,
        "display_number": 1,
    }
]


# ---------------------------------------------------------------------------
# Capa de ejecución: traduce acciones de Claude a acciones reales del sistema
# ---------------------------------------------------------------------------

class EjecutorAcciones:
    """
    Ejecuta las acciones del ratón y teclado en el sistema operativo.
    Requiere: pip install pyautogui pillow
    """

    def __init__(self):
        try:
            import pyautogui
            self.pyautogui = pyautogui
            pyautogui.FAILSAFE = True  # mover ratón a esquina superior izquierda para abortar
            pyautogui.PAUSE = 0.5      # pausa entre acciones para estabilidad
            self.disponible = True
        except ImportError:
            print("AVISO: pyautogui no está instalado. Ejecutando en modo simulación.")
            self.disponible = False

    def capturar_pantalla(self) -> str:
        """Captura la pantalla y devuelve la imagen en base64."""
        if not self.disponible:
            return self._pantalla_simulada()

        import io
        screenshot = self.pyautogui.screenshot()
        screenshot = screenshot.resize((ANCHO, ALTO))
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def _pantalla_simulada(self) -> str:
        """Devuelve una imagen PNG mínima para modo simulación."""
        import struct
        import zlib
        def chunk(tipo, datos):
            c = tipo + datos
            return struct.pack(">I", len(datos)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(b"\x00\xcc\xcc\xcc"))
        png += chunk(b"IEND", b"")
        return base64.standard_b64encode(png).decode()

    def ejecutar_accion(self, accion: dict) -> Optional[str]:
        """
        Ejecuta una acción de Computer Use.
        Devuelve la imagen de pantalla resultante si es una acción de captura.
        """
        tipo = accion.get("action")
        print(f"  [Ejecutor] Acción: {tipo} | Params: {accion}")

        if not self.disponible:
            print(f"  [Simulación] Se ejecutaría: {tipo}")
            return None

        pg = self.pyautogui

        if tipo == "screenshot":
            return self.capturar_pantalla()

        elif tipo == "left_click":
            coord = accion["coordinate"]
            pg.click(coord[0], coord[1])

        elif tipo == "right_click":
            coord = accion["coordinate"]
            pg.rightClick(coord[0], coord[1])

        elif tipo == "double_click":
            coord = accion["coordinate"]
            pg.doubleClick(coord[0], coord[1])

        elif tipo == "mouse_move":
            coord = accion["coordinate"]
            pg.moveTo(coord[0], coord[1])

        elif tipo == "type":
            pg.typewrite(accion["text"], interval=0.05)

        elif tipo == "key":
            # Convertir formato de Claude a pyautogui
            tecla = accion["text"].replace("+", " ").lower()
            pg.hotkey(*tecla.split())

        elif tipo == "scroll":
            coord = accion.get("coordinate", [ANCHO // 2, ALTO // 2])
            direccion = accion.get("direction", "down")
            cantidad = accion.get("amount", 3)
            clics = cantidad if direccion == "down" else -cantidad
            pg.scroll(clics, x=coord[0], y=coord[1])

        elif tipo == "left_click_drag":
            inicio = accion["start_coordinate"]
            fin = accion["coordinate"]
            pg.dragTo(fin[0], fin[1], button="left", duration=0.5)

        time.sleep(0.3)  # esperar a que la UI se actualice
        return None


# ---------------------------------------------------------------------------
# Bucle de control principal
# ---------------------------------------------------------------------------

def agente_computer_use(objetivo: str, max_pasos: int = 20) -> str:
    """
    Agente de Computer Use que ejecuta una tarea en la UI.

    Args:
        objetivo: La tarea a realizar en lenguaje natural.
        max_pasos: Número máximo de acciones antes de detenerse.

    Returns:
        Descripción del resultado final.
    """
    ejecutor = EjecutorAcciones()
    historial = []
    paso = 0

    print(f"\n{'='*60}")
    print(f"AGENTE COMPUTER USE")
    print(f"Objetivo: {objetivo}")
    print(f"{'='*60}\n")

    # Captura inicial de la pantalla
    screenshot_actual = ejecutor.capturar_pantalla()

    # Mensaje inicial con la captura y el objetivo
    historial.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_actual,
                },
            },
            {"type": "text", "text": objetivo},
        ],
    })

    while paso < max_pasos:
        paso += 1
        print(f"\n--- Paso {paso}/{max_pasos} ---")

        respuesta = cliente.beta.messages.create(
            model=MODELO,
            max_tokens=4096,
            tools=HERRAMIENTAS_COMPUTER_USE,
            system=(
                "Eres un agente que controla un computador. Observa la pantalla "
                "y ejecuta acciones para completar el objetivo del usuario. "
                "Sé metódico: primero identifica dónde estás en la interfaz, "
                "luego toma la acción más lógica. Si algo no funciona, prueba "
                "una alternativa. Cuando hayas completado el objetivo, describe "
                "el resultado sin usar más herramientas."
            ),
            messages=historial,
            betas=["computer-use-2024-10-22"],
        )

        # Añadir respuesta del asistente al historial
        historial.append({"role": "assistant", "content": respuesta.content})

        # Si el modelo terminó (sin más acciones)
        if respuesta.stop_reason == "end_turn":
            texto_final = next(
                (b.text for b in respuesta.content if hasattr(b, "text")), ""
            )
            print(f"\nTarea completada en {paso} pasos.")
            print(f"Resultado: {texto_final}")
            return texto_final

        # Procesar acciones de Computer Use
        resultados_herramientas = []

        for bloque in respuesta.content:
            if bloque.type != "tool_use":
                continue

            accion = bloque.input
            resultado_screenshot = ejecutor.ejecutar_accion(accion)

            # Si la acción fue un screenshot, incluir la imagen; si no, capturar la pantalla actualizada
            if resultado_screenshot:
                imagen_resultado = resultado_screenshot
            else:
                time.sleep(0.5)
                imagen_resultado = ejecutor.capturar_pantalla()

            # Construir el resultado para el historial
            contenido_resultado = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": imagen_resultado,
                    },
                }
            ]

            resultados_herramientas.append({
                "type": "tool_result",
                "tool_use_id": bloque.id,
                "content": contenido_resultado,
            })

        historial.append({"role": "user", "content": resultados_herramientas})

    return f"Se alcanzó el límite de {max_pasos} pasos sin completar la tarea."


if __name__ == "__main__":
    resultado = agente_computer_use(
        objetivo="Abre el navegador web y navega a https://example.com",
        max_pasos=10,
    )
    print(f"\nResultado final: {resultado}")
```

---

## 4. Caso práctico — automatizar un formulario web

En este caso práctico, Claude rellena un formulario HTML paso a paso. Usaremos Playwright para controlar el navegador de forma programática, combinado con Computer Use para la visión.

```python
# pip install playwright anthropic
# playwright install chromium

import asyncio
import base64
import anthropic
from playwright.async_api import async_playwright, Page

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"

# HTML de formulario de prueba (en producción sería una URL real)
FORMULARIO_HTML = """
<!DOCTYPE html>
<html>
<head><title>Formulario de Registro</title></head>
<body style="font-family: sans-serif; max-width: 500px; margin: 40px auto; padding: 20px;">
  <h1>Registro de Usuario</h1>
  <form id="registro">
    <label>Nombre completo:<br>
      <input type="text" id="nombre" name="nombre" style="width:100%; padding:8px; margin:8px 0;">
    </label><br>
    <label>Correo electrónico:<br>
      <input type="email" id="email" name="email" style="width:100%; padding:8px; margin:8px 0;">
    </label><br>
    <label>Contraseña:<br>
      <input type="password" id="password" name="password" style="width:100%; padding:8px; margin:8px 0;">
    </label><br>
    <label>País:
      <select id="pais" name="pais" style="width:100%; padding:8px; margin:8px 0;">
        <option value="">Selecciona un país</option>
        <option value="es">España</option>
        <option value="mx">México</option>
        <option value="ar">Argentina</option>
        <option value="co">Colombia</option>
      </select>
    </label><br>
    <label>
      <input type="checkbox" id="terminos" name="terminos">
      Acepto los términos y condiciones
    </label><br><br>
    <button type="submit" style="background:#007bff; color:white; padding:10px 20px; border:none; cursor:pointer;">
      Registrarse
    </button>
  </form>
  <div id="resultado" style="margin-top:20px; color:green;"></div>
  <script>
    document.getElementById('registro').addEventListener('submit', function(e) {
      e.preventDefault();
      document.getElementById('resultado').textContent = '¡Registro completado con éxito!';
    });
  </script>
</body>
</html>
"""

DATOS_USUARIO = {
    "nombre": "María García López",
    "email": "maria.garcia@ejemplo.com",
    "password": "ContraseñaSegura123!",
    "pais": "es",
}


async def capturar_pantalla_playwright(pagina: Page) -> str:
    """Captura la pantalla del navegador y devuelve base64."""
    screenshot_bytes = await pagina.screenshot(full_page=False)
    return base64.standard_b64encode(screenshot_bytes).decode("utf-8")


async def llenar_formulario_con_ia(datos: dict) -> str:
    """
    Usa Claude con Computer Use para rellenar el formulario.
    El agente ve la pantalla y decide qué hacer en cada paso.
    """
    async with async_playwright() as p:
        navegador = await p.chromium.launch(headless=False)  # headless=True para producción
        pagina = await navegador.new_page(viewport={"width": 1280, "height": 800})

        # Cargar el formulario local
        await pagina.set_content(FORMULARIO_HTML)
        await pagina.wait_for_load_state("networkidle")

        print("Formulario cargado. Iniciando agente de Computer Use...\n")

        # Captura inicial
        screenshot = await capturar_pantalla_playwright(pagina)

        historial = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": screenshot},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Rellena el formulario de registro con estos datos:\n"
                            f"- Nombre: {datos['nombre']}\n"
                            f"- Email: {datos['email']}\n"
                            f"- Contraseña: {datos['password']}\n"
                            f"- País: España\n\n"
                            f"Una vez rellenado, marca los términos y condiciones y haz clic en 'Registrarse'."
                        ),
                    },
                ],
            }
        ]

        herramientas = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1280,
                "display_height_px": 800,
                "display_number": 1,
            }
        ]

        for paso in range(15):  # máximo 15 acciones para rellenar el formulario
            respuesta = cliente.beta.messages.create(
                model=MODELO,
                max_tokens=2048,
                tools=herramientas,
                system=(
                    "Eres un agente que controla un navegador web para rellenar formularios. "
                    "Observa la pantalla y haz clic en los campos para rellenarlos. "
                    "Usa 'type' para escribir texto después de hacer clic en un campo. "
                    "Para campos select, haz clic en el elemento y luego usa JavaScript si es necesario."
                ),
                messages=historial,
                betas=["computer-use-2024-10-22"],
            )

            historial.append({"role": "assistant", "content": respuesta.content})

            if respuesta.stop_reason == "end_turn":
                texto = next((b.text for b in respuesta.content if hasattr(b, "text")), "")
                print(f"Agente terminó: {texto}")
                break

            resultados = []
            for bloque in respuesta.content:
                if bloque.type != "tool_use":
                    continue

                accion = bloque.input
                accion_tipo = accion.get("action")
                print(f"Paso {paso+1}: {accion_tipo} → {accion}")

                # Traducir acciones de Claude a Playwright
                if accion_tipo == "screenshot":
                    pass  # simplemente capturamos de nuevo
                elif accion_tipo == "left_click":
                    x, y = accion["coordinate"]
                    await pagina.mouse.click(x, y)
                elif accion_tipo == "double_click":
                    x, y = accion["coordinate"]
                    await pagina.mouse.dblclick(x, y)
                elif accion_tipo == "type":
                    await pagina.keyboard.type(accion["text"], delay=50)
                elif accion_tipo == "key":
                    await pagina.keyboard.press(accion["text"])
                elif accion_tipo == "scroll":
                    x, y = accion.get("coordinate", [640, 400])
                    delta = -300 if accion.get("direction") == "up" else 300
                    await pagina.mouse.wheel(0, delta)

                await asyncio.sleep(0.5)
                nueva_captura = await capturar_pantalla_playwright(pagina)

                resultados.append({
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": nueva_captura},
                        }
                    ],
                })

            historial.append({"role": "user", "content": resultados})

        # Verificar resultado
        await asyncio.sleep(1)
        resultado_elemento = await pagina.query_selector("#resultado")
        resultado_texto = await resultado_elemento.inner_text() if resultado_elemento else "Sin resultado"

        await navegador.close()
        return resultado_texto


if __name__ == "__main__":
    resultado = asyncio.run(llenar_formulario_con_ia(DATOS_USUARIO))
    print(f"\nResultado del formulario: {resultado}")
```

---

## 5. Caso práctico — agente de scraping visual

A veces el scraping con CSS selectors o XPath es frágil ante cambios de diseño. Un agente visual entiende la página semánticamente.

```python
import asyncio
import base64
import json
import anthropic
from playwright.async_api import async_playwright

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"


async def extraer_informacion_visual(url: str, que_extraer: str) -> dict:
    """
    Navega a una URL y extrae información usando visión de IA.
    No usa selectores CSS: Claude interpreta la página visualmente.

    Args:
        url: URL a visitar.
        que_extraer: Descripción en lenguaje natural de qué extraer.

    Returns:
        Diccionario con la información extraída.
    """
    async with async_playwright() as p:
        navegador = await p.chromium.launch(headless=True)
        pagina = await navegador.new_page(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        )

        print(f"Navegando a: {url}")
        await pagina.goto(url, wait_until="networkidle", timeout=30000)

        # Captura de pantalla completa con scroll
        capturas = []

        # Captura inicial (parte visible)
        captura1 = await pagina.screenshot()
        capturas.append(base64.standard_b64encode(captura1).decode())

        # Hacer scroll y capturar más contenido
        for _ in range(2):
            await pagina.mouse.wheel(0, 600)
            await asyncio.sleep(0.8)
            captura_n = await pagina.screenshot()
            capturas.append(base64.standard_b64encode(captura_n).decode())

        await navegador.close()

    # Construir el mensaje con múltiples capturas
    contenido = []
    for i, captura in enumerate(capturas):
        contenido.append({"type": "text", "text": f"Captura {i+1} de {len(capturas)}:"})
        contenido.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": captura},
        })

    contenido.append({
        "type": "text",
        "text": (
            f"Analiza las capturas de pantalla de la página web y extrae la siguiente información:\n\n"
            f"{que_extraer}\n\n"
            f"Devuelve SOLO un JSON válido con las claves relevantes para la información solicitada. "
            f"Si no encuentras algún dato, usa null como valor."
        ),
    })

    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=2048,
        system=(
            "Eres un experto en extracción de información de páginas web. "
            "Analizas capturas de pantalla y extraes datos estructurados con precisión. "
            "Devuelves siempre JSON válido."
        ),
        messages=[{"role": "user", "content": contenido}],
    )

    texto = respuesta.content[0].text.strip()

    # Limpiar formato markdown si lo hay
    if "```" in texto:
        partes = texto.split("```")
        for parte in partes:
            parte = parte.strip()
            if parte.startswith("json"):
                parte = parte[4:].strip()
            try:
                return json.loads(parte)
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"resultado_texto": texto, "error": "No se pudo parsear como JSON"}


if __name__ == "__main__":
    # Ejemplo con Wikipedia (pública, sin restricciones de scraping)
    resultado = asyncio.run(
        extraer_informacion_visual(
            url="https://es.wikipedia.org/wiki/Inteligencia_artificial",
            que_extraer=(
                "Extrae: título de la página, primeras 3 secciones principales del artículo, "
                "y si aparece alguna fecha o año importante mencionado en el inicio del artículo."
            ),
        )
    )

    print("\nInformación extraída:")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
```

---

## 6. Seguridad y consideraciones éticas

Computer Use amplifica enormemente la capacidad de automatización. Con esa potencia vienen responsabilidades importantes.

**Riesgos técnicos:**

- **Acciones irreversibles:** un clic en "Eliminar cuenta" o "Confirmar pago" no se puede deshacer. Siempre implementa confirmación humana para acciones críticas.
- **Fuga de credenciales:** si el agente puede ver la pantalla, puede leer contraseñas si aparecen en texto claro. Nunca incluyas credenciales en el contexto del agente.
- **Bucles infinitos:** el agente puede quedar atrapado en un estado de la UI. Implementa siempre un límite de pasos.
- **Costos:** Computer Use es intensivo en tokens (las imágenes son costosas). Un agente sin límites puede generar facturas inesperadas.

**Medidas de seguridad recomendadas:**

```python
# 1. Lista de sitios permitidos
SITIOS_PERMITIDOS = ["app.interna.empresa.com", "dashboard.ejemplo.com"]

def verificar_url_permitida(url: str) -> bool:
    from urllib.parse import urlparse
    dominio = urlparse(url).netloc
    return any(permitido in dominio for permitido in SITIOS_PERMITIDOS)

# 2. Confirmación humana para acciones de alto riesgo
ACCIONES_CRITICAS = ["submit", "delete", "purchase", "transfer", "confirm"]

def requiere_confirmacion(texto_boton: str) -> bool:
    return any(critica in texto_boton.lower() for critica in ACCIONES_CRITICAS)

# 3. Límite de tiempo de ejecución
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("El agente ha excedido el tiempo máximo de ejecución.")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minutos máximo

# 4. Modo sandbox: ejecutar en un navegador aislado
# Usa playwright con un perfil de usuario temporal y sin acceso a cookies reales
```

**Consideraciones éticas:**

- **Solo en sitios donde tengas permiso:** usar Computer Use en sitios que prohíben el scraping o la automatización (en sus Términos de Servicio) puede ser ilegal en algunas jurisdicciones.
- **No para engañar:** no uses Computer Use para simular comportamiento humano en pruebas de Captcha, crear cuentas falsas o manipular métricas.
- **Transparencia:** si automatizas interacciones con otras personas (chats, formularios de contacto), identifica que es un sistema automatizado.
- **Principio de mínimo acceso:** el agente solo debe tener acceso a lo que necesita. No le des acceso al escritorio completo si solo necesita controlar una ventana.

---

## 7. Alternativas ligeras

Computer Use con Claude es potente pero costoso. Para casos más simples, existen alternativas:

**PyAutoGUI — control de teclado y ratón sin IA:**

```python
# pip install pyautogui pillow
import pyautogui
import time

# Mover el ratón a coordenadas y hacer clic
pyautogui.click(x=500, y=300)

# Escribir texto
pyautogui.typewrite("Hola mundo", interval=0.05)

# Capturar pantalla y buscar imagen
posicion = pyautogui.locateOnScreen("boton_aceptar.png", confidence=0.8)
if posicion:
    pyautogui.click(posicion)
```

**Playwright con selectores — scraping y automatización web robusta:**

```python
# pip install playwright && playwright install chromium
import asyncio
from playwright.async_api import async_playwright

async def automatizar_web():
    async with async_playwright() as p:
        navegador = await p.chromium.launch()
        pagina = await navegador.new_page()
        await pagina.goto("https://ejemplo.com")

        # Interactuar con elementos por selector CSS o texto
        await pagina.fill("#campo-nombre", "Juan García")
        await pagina.click("button:has-text('Enviar')")
        await pagina.wait_for_selector(".mensaje-exito")

        resultado = await pagina.inner_text(".mensaje-exito")
        print(resultado)
        await navegador.close()

asyncio.run(automatizar_web())
```

**Cuándo usar cada herramienta:**

| Herramienta | Caso de uso | Costo | Complejidad |
|---|---|---|---|
| **Computer Use + Claude** | Webs complejas, sin selectores estables, razonamiento necesario | Alto | Alta |
| **Playwright + selectores** | Webs con estructura HTML estable | Bajo | Media |
| **PyAutoGUI** | Aplicaciones de escritorio (no web) | Muy bajo | Baja |
| **Playwright + LLM** | Playwright para navegar, LLM para interpretar | Medio | Media |

---

## 8. Extensiones sugeridas

- **Grabación de sesiones:** guarda cada screenshot y acción en disco para depuración y auditoría.
- **Modo de confirmación:** antes de ejecutar cualquier acción, muéstrasela al usuario y espera aprobación (útil para procesos de alto riesgo).
- **Reintentos inteligentes:** si Claude detecta que una acción no tuvo el efecto esperado, que intente una alternativa automáticamente.
- **Integración con MCP:** crea un servidor MCP que exponga Computer Use como herramienta estándar, usable desde Claude Desktop.
- **Multi-monitor:** adapta el bucle para trabajar con múltiples pantallas, útil en flujos de trabajo profesionales.
- **Agente de QA automatizado:** usa Computer Use para hacer pruebas de regresión visual en tu propia aplicación web.

---

**Siguiente:** [04 — Memoria a Largo Plazo](./04-memoria-largo-plazo.md)
