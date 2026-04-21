# Seguridad y límites en Computer Use: principios y guardrails

## Principio de mínimo privilegio

La regla más importante en Computer Use: **el agente solo debe tener acceso a lo que necesita para la tarea específica**.

```
❌ Mal: el agente tiene acceso a toda la pantalla con sesión de producción abierta
✅ Bien: el agente corre en un entorno aislado con acceso solo a la aplicación necesaria
```

### Entorno aislado mínimo

```yaml
# docker-compose.yml — entorno de mínimo privilegio
services:
  computer-use-agent:
    image: anthropics/anthropic-quickstarts:computer-use-demo-latest
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ALLOWED_URLS=erp.empresa.com,portal.proveedor.com  # Solo estas URLs
      - MAX_ITERATIONS=30  # Límite de iteraciones
    network_mode: "bridge"
    # Sin acceso a red interna corporativa salvo las URLs permitidas
    dns: ["8.8.8.8"]
    read_only: false
    tmpfs:
      - /tmp  # Solo escritura en /tmp
    volumes:
      - ./output:/output:rw  # Solo puede escribir en output/
      # Sin acceso a directorios del host
```

## Qué NO dejar hacer a un agente Computer Use

```python
# Lista de acciones prohibidas
ACCIONES_PROHIBIDAS = [
    "transferencia bancaria",
    "pago",
    "eliminar base de datos",
    "borrar todos",
    "formatear disco",
    "enviar a todos los contactos",
    "publicar en redes sociales",
    "cambiar contraseña de administrador",
]

def validar_instruccion(instruccion: str) -> tuple[bool, str]:
    """Verifica que la instrucción no contiene acciones prohibidas."""
    instruccion_lower = instruccion.lower()
    for accion in ACCIONES_PROHIBIDAS:
        if accion in instruccion_lower:
            return False, f"Instrucción contiene acción prohibida: '{accion}'"
    return True, ""

# Uso obligatorio antes de ejecutar cualquier tarea
def ejecutar_tarea_segura(instruccion: str):
    permitida, motivo = validar_instruccion(instruccion)
    if not permitida:
        raise ValueError(f"Instrucción rechazada: {motivo}")
    # ... continuar con el agente
```

## Guardrails: lista blanca de URLs y aplicaciones

```python
from urllib.parse import urlparse

URLS_PERMITIDAS = {
    "erp.miempresa.com",
    "portal.proveedor.com",
    "docs.miempresa.com",
}

APLICACIONES_PERMITIDAS = {
    "firefox",
    "chromium",
    "gedit",  # Solo editor de texto básico
}

def guardrail_url(url: str) -> bool:
    """Verifica que una URL está en la lista blanca."""
    dominio = urlparse(url).netloc.lower().removeprefix("www.")
    if dominio not in URLS_PERMITIDAS:
        raise PermissionError(f"URL no permitida: {dominio}. Permitidas: {URLS_PERMITIDAS}")
    return True

def guardrail_aplicacion(nombre_app: str) -> bool:
    """Verifica que una aplicación está permitida."""
    if nombre_app.lower() not in APLICACIONES_PERMITIDAS:
        raise PermissionError(f"Aplicación no permitida: {nombre_app}")
    return True

# Interceptar comandos bash antes de ejecutarlos
import shlex

COMANDOS_PELIGROSOS = ["rm", "dd", "mkfs", "shred", "wget", "curl", "ssh", "scp"]

def guardrail_bash(comando: str) -> bool:
    try:
        tokens = shlex.split(comando)
    except ValueError:
        raise ValueError(f"Comando bash malformado: {comando}")

    primer_comando = tokens[0] if tokens else ""
    if primer_comando in COMANDOS_PELIGROSOS:
        raise PermissionError(f"Comando peligroso bloqueado: {primer_comando}")
    return True
```

## Human-in-the-loop para acciones críticas

```python
import anthropic

client = anthropic.Anthropic()

ACCIONES_QUE_REQUIEREN_CONFIRMACION = [
    "enviar email",
    "guardar y contabilizar",
    "confirmar pedido",
    "firmar documento",
    "publicar",
]

def requiere_confirmacion(descripcion_accion: str) -> bool:
    return any(k in descripcion_accion.lower() for k in ACCIONES_QUE_REQUIEREN_CONFIRMACION)

def pedir_confirmacion_usuario(descripcion: str) -> bool:
    """Pausa el agente y pide confirmación humana."""
    print(f"\n⚠️  ACCIÓN QUE REQUIERE CONFIRMACIÓN:")
    print(f"   {descripcion}")
    respuesta = input("   ¿Confirmas esta acción? (s/n): ").strip().lower()
    return respuesta in ("s", "si", "sí", "y", "yes")

def agente_con_hitl(instruccion: str, max_iteraciones: int = 20):
    """Agente Computer Use con paradas para aprobación humana."""
    mensajes = [{"role": "user", "content": instruccion}]
    tools = [
        {"type": "computer_20250124", "name": "computer", "display_width_px": 1280, "display_height_px": 800, "display_number": 1},
        {"type": "bash_20250124", "name": "bash"},
    ]

    for iteracion in range(max_iteraciones):
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
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
            descripcion = f"{tu.name}: {tu.input}"

            # Parar y pedir confirmación si la acción es crítica
            if requiere_confirmacion(str(tu.input)):
                if not pedir_confirmacion_usuario(descripcion):
                    return "Tarea cancelada por el usuario"

            # Ejecutar la acción (simplificado)
            resultados.append({"type": "tool_result", "tool_use_id": tu.id, "content": "OK"})

        mensajes.append({"role": "user", "content": resultados})
```

## Auditoría y logging completo

```python
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("computer-use-audit")

class AuditoriaComputerUse:
    def __init__(self, sesion_id: str):
        self.sesion_id = sesion_id
        self.log_path = Path(f"logs/sesion_{sesion_id}_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def registrar_accion(self, iteracion: int, herramienta: str, entrada: dict, resultado: str):
        entrada_redactada = self._redactar_datos_sensibles(entrada)
        registro = {
            "timestamp": datetime.now().isoformat(),
            "sesion": self.sesion_id,
            "iteracion": iteracion,
            "herramienta": herramienta,
            "entrada": entrada_redactada,
            "resultado_preview": resultado[:200] if resultado else "",
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        logger.info(f"[{self.sesion_id}:{iteracion}] {herramienta} → {herramienta}")

    def _redactar_datos_sensibles(self, datos: dict) -> dict:
        copia = dict(datos)
        campos_sensibles = ["password", "token", "secret", "key", "credential"]
        for campo in campos_sensibles:
            if campo in copia:
                copia[campo] = "***REDACTADO***"
        return copia
```

## Checklist de seguridad antes de desplegar

```markdown
## Pre-despliegue

- [ ] El agente corre en Docker con red restringida
- [ ] Lista blanca de URLs configurada y probada
- [ ] Lista blanca de aplicaciones configurada
- [ ] Guardrails de comandos bash implementados
- [ ] Validación de instrucciones entrantes activa
- [ ] Human-in-the-loop para acciones irreversibles
- [ ] Logging de auditoría completo activado
- [ ] Credenciales en variables de entorno (nunca hardcodeadas)
- [ ] Límite de iteraciones configurado (max 30-50)
- [ ] Timeout global de sesión configurado
- [ ] Tests de penetración básicos realizados
- [ ] Revisión de seguridad por un segundo par de ojos
```

## Resumen

- **Mínimo privilegio**: entornos Docker aislados con acceso mínimo
- **Guardrails**: lista blanca de URLs y apps, bloqueo de comandos peligrosos
- **Human-in-the-loop**: pausa obligatoria antes de acciones irreversibles
- **Auditoría**: log completo de cada acción con redacción de datos sensibles
- **Regla de oro**: nunca ejecutes Computer Use con acceso directo a producción sin sandbox
