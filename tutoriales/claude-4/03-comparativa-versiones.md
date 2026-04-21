# Claude 3 vs Claude 4: migración y diferencias clave

## ¿Qué cambia de Claude 3 a Claude 4?

La familia Claude 4.X supone una mejora significativa respecto a Claude 3 en varias dimensiones:

| Dimensión | Claude 3 | Claude 4.X |
|-----------|----------|------------|
| Razonamiento | Bueno | Mejorado (+30% en benchmarks) |
| Seguimiento de instrucciones | Sólido | Más preciso en instrucciones largas |
| Extended Thinking | No disponible | Disponible en Opus 4.7 |
| Computer Use | Beta limitada | Soporte mejorado en Sonnet 4.6 |
| Context window | 200K | 200K (sin cambio) |
| Velocidad | — | Haiku 4.5 más rápido que Haiku 3 |

## Modelos retirados vs activos

Anthropic retira gradualmente los modelos más antiguos. Estado en 2025:

| Modelo antiguo | Estado | Migrar a |
|----------------|--------|----------|
| `claude-3-opus-20240229` | Deprecado | `claude-opus-4-7` |
| `claude-3-sonnet-20240229` | Deprecado | `claude-sonnet-4-6` |
| `claude-3-haiku-20240307` | Disponible (sin soporte largo plazo) | `claude-haiku-4-5-20251001` |
| `claude-3-5-sonnet-20241022` | Disponible | `claude-sonnet-4-6` (recomendado) |
| `claude-3-5-haiku-20241022` | Disponible | `claude-haiku-4-5-20251001` |

Consulta siempre la [documentación oficial de Anthropic](https://docs.anthropic.com/en/docs/about-claude/models) para el estado actualizado.

## Guía de migración

La migración es sencilla: en la mayoría de casos, solo necesitas cambiar el `model` ID.

### Paso 1: actualizar dependencias

```bash
pip install --upgrade anthropic
# Asegúrate de tener >= 0.49
python -c "import anthropic; print(anthropic.__version__)"
```

### Paso 2: reemplazar IDs de modelo

```python
# ANTES (Claude 3)
MODELOS_LEGACY = {
    "opus": "claude-3-opus-20240229",
    "sonnet": "claude-3-5-sonnet-20241022",
    "haiku": "claude-3-haiku-20240307",
}

# DESPUÉS (Claude 4)
MODELOS_CLAUDE4 = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}
```

### Paso 3: verificar compatibilidad de parámetros

La API de mensajes es 100% compatible. No necesitas cambiar la estructura de tus llamadas:

```python
import anthropic

client = anthropic.Anthropic()

# Esta llamada funciona igual con Claude 3 y Claude 4
response = client.messages.create(
    model="claude-sonnet-4-6",  # Solo cambia esto
    max_tokens=1024,
    system="Eres un asistente útil.",
    messages=[
        {"role": "user", "content": "¿Cuál es la capital de Francia?"}
    ],
)
print(response.content[0].text)
```

### Paso 4: script de migración masiva

Si tienes varios archivos con IDs hardcodeados, usa este script:

```python
import re
from pathlib import Path

REEMPLAZOS = {
    "claude-3-opus-20240229": "claude-opus-4-7",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-sonnet-20240229": "claude-sonnet-4-6",
    "claude-3-haiku-20240307": "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-20241022": "claude-haiku-4-5-20251001",
}

def migrar_archivo(ruta: Path) -> int:
    """Reemplaza IDs de modelo en un archivo. Devuelve el número de cambios."""
    contenido = ruta.read_text(encoding="utf-8")
    cambios = 0
    for viejo, nuevo in REEMPLAZOS.items():
        if viejo in contenido:
            contenido = contenido.replace(viejo, nuevo)
            cambios += contenido.count(nuevo)
    if cambios:
        ruta.write_text(contenido, encoding="utf-8")
    return cambios

# Migrar todos los archivos Python del proyecto
proyecto = Path(".")
total = 0
for archivo in proyecto.rglob("*.py"):
    n = migrar_archivo(archivo)
    if n:
        print(f"  {archivo}: {n} reemplazo(s)")
        total += n

print(f"\nTotal de reemplazos: {total}")
```

## Diferencias de comportamiento

### Seguimiento de instrucciones más preciso

Claude 4.X sigue instrucciones largas y complejas con mayor fidelidad. Si tenías prompts que ocasionalmente fallaban en Claude 3, vale la pena re-testarlos:

```python
# Prueba de seguimiento de instrucciones
prompt_sistema = """
Responde SIEMPRE en formato JSON con exactamente estas claves:
- "resumen": string de máximo 50 palabras
- "puntuacion": número del 1 al 10
- "recomendacion": booleano
No incluyas texto fuera del JSON.
"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    system=prompt_sistema,
    messages=[{"role": "user", "content": "Evalúa este producto: 'Auriculares inalámbricos con 30h de batería y cancelación de ruido activa por 89€'"}],
)
# Claude 4 es más consistente devolviendo JSON puro
import json
datos = json.loads(response.content[0].text)
print(datos)
```

### Tool use más robusto

Claude 4 selecciona y llama herramientas con mayor precisión, especialmente en escenarios con múltiples herramientas disponibles:

```python
tools = [
    {
        "name": "buscar_producto",
        "description": "Busca un producto por nombre en el catálogo",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre del producto"},
                "categoria": {"type": "string", "enum": ["electronica", "ropa", "hogar"]},
            },
            "required": ["nombre"],
        },
    },
    {
        "name": "obtener_precio",
        "description": "Obtiene el precio actual de un producto por su ID",
        "input_schema": {
            "type": "object",
            "properties": {"producto_id": {"type": "string"}},
            "required": ["producto_id"],
        },
    },
]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "¿Cuánto cuesta el iPhone 16?"}],
)
# Claude 4 selecciona correctamente buscar_producto antes que obtener_precio
```

## Checklist de migración

- [ ] `pip install anthropic>=0.49`
- [ ] Reemplazar todos los IDs de modelo con el script de migración
- [ ] Re-ejecutar tests de calidad con los nuevos modelos
- [ ] Verificar que los prompts de sistema siguen funcionando
- [ ] Revisar los `max_tokens` — Claude 4 puede ser más verboso
- [ ] Actualizar la documentación interna con los nuevos IDs
- [ ] Si usas Extended Thinking, añadir el parámetro `betas`

## Resumen

- La migración de Claude 3 a Claude 4 es un cambio de una línea en la mayoría de casos
- Claude 4 mejora en razonamiento, seguimiento de instrucciones y tool use
- Comprueba la documentación oficial para los IDs actualizados
- Extended Thinking es una funcionalidad nueva exclusiva de Opus 4.7
