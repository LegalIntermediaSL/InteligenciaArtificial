# Casos empresariales de Computer Use: RPA inteligente

## RPA clásico vs RPA con IA

| Aspecto | RPA clásico (UiPath, Automation Anywhere) | RPA con Computer Use |
|---------|------------------------------------------|---------------------|
| Configuración | Semanas de programación | Horas con instrucciones |
| Adaptabilidad | Rígido, se rompe con cambios | Se adapta visualmente |
| Manejo de excepciones | Manual, requiere rediseño | Razona sobre excepciones |
| Coste | Licencias altas, mantenimiento | Pay-per-use API |
| Escalado | Infraestructura dedicada | Horizontal vía API |
| Auditoría | Logs de acciones | Logs + razonamiento |

**Cuándo preferir RPA clásico:** procesos completamente estables, alto volumen (>10K/día), donde el coste por llamada importa.  
**Cuándo preferir Computer Use:** procesos que cambian frecuentemente, excepciones complejas, prototipado rápido.

## Caso 1: Procesamiento de facturas en ERP legacy

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

def procesar_factura_en_erp(
    datos_factura: dict,
    erp_url: str,
    credenciales: dict,
) -> dict:
    """
    Introduce una factura en un ERP legacy usando Computer Use.
    
    datos_factura: {'proveedor': str, 'importe': float, 'fecha': str, 'concepto': str}
    credenciales: {'usuario': str, 'password': str}  — usar variables de entorno
    """
    instruccion = f"""
    1. Ve a {erp_url} e inicia sesión con usuario: {credenciales['usuario']}
    2. Navega a Compras → Facturas → Nueva factura
    3. Rellena los campos:
       - Proveedor: {datos_factura['proveedor']}
       - Fecha: {datos_factura['fecha']}
       - Importe: {datos_factura['importe']}
       - Concepto: {datos_factura['concepto']}
    4. Haz clic en "Guardar y contabilizar"
    5. Captura el número de factura asignado por el sistema
    6. Responde con JSON: {{"numero_factura": "...", "exito": true}}
    """

    # [bucle de control estándar - ver tutorial 01]
    # resultado = agente_computer_use(instruccion)
    # return json.loads(resultado)
    return {"numero_factura": "FAC-2025-001", "exito": True}  # placeholder
```

## Caso 2: Triaje y clasificación automática de emails

```python
def procesar_bandeja_entrada(max_emails: int = 20) -> list[dict]:
    """
    Abre el cliente de correo, lee los emails no leídos y los clasifica.
    Devuelve una lista con la clasificación de cada email.
    """
    instruccion = f"""
    Abre el cliente de correo. Para cada uno de los primeros {max_emails} emails no leídos:
    1. Haz clic en el email para leerlo
    2. Clasifícalo como: "urgente", "soporte", "ventas", "spam", "otro"
    3. Marca los de tipo "spam" para mover a carpeta Spam
    4. Para los "urgentes", aplica la etiqueta roja "URGENTE"
    
    Al terminar, devuelve un JSON array con cada email procesado:
    [{{"de": "...", "asunto": "...", "categoria": "...", "accion": "..."}}]
    """
    # [bucle de control]
    return []

# Programar para ejecutar cada hora
import schedule
import time

def job_email():
    clasificados = procesar_bandeja_entrada()
    print(f"Emails procesados: {len(clasificados)}")
    for email in [e for e in clasificados if e.get("categoria") == "urgente"]:
        print(f"  URGENTE: {email['de']} — {email['asunto']}")

schedule.every().hour.do(job_email)
```

## Caso 3: Actualización de CRM desde múltiples fuentes

```python
def sincronizar_contacto_crm(
    nombre: str,
    linkedin_url: str,
    crm_url: str,
) -> bool:
    """
    Busca un contacto en LinkedIn, extrae su información actualizada
    y la sincroniza en el CRM.
    """
    instruccion = f"""
    Tarea: actualizar el perfil de {nombre} en el CRM con datos de LinkedIn.
    
    Pasos:
    1. Ve a {linkedin_url} y extrae:
       - Cargo actual
       - Empresa actual
       - Ciudad
       - Resumen profesional (primeras 100 palabras)
    
    2. Ve a {crm_url} y busca el contacto "{nombre}"
    
    3. Actualiza los campos con los datos de LinkedIn
    
    4. Guarda los cambios
    
    Responde "ACTUALIZADO" si todo fue bien, o "ERROR: motivo" si falló.
    """
    # [bucle de control]
    return True
```

## Caso 4: Testing visual de aplicaciones

```python
def ejecutar_test_visual(
    url_app: str,
    pasos_test: list[str],
    nombre_test: str,
) -> dict:
    """
    Ejecuta un test de UI descrito en lenguaje natural.
    Útil para aplicaciones donde los tests de Selenium son frágiles.
    """
    pasos_formateados = "\n".join(f"{i+1}. {paso}" for i, paso in enumerate(pasos_test))

    instruccion = f"""
    Ejecuta el test '{nombre_test}' en {url_app}:
    
    {pasos_formateados}
    
    Para cada paso, toma un screenshot y verifica que el resultado es el esperado.
    Si un paso falla, para y describe el problema.
    
    Responde en JSON:
    {{
      "test": "{nombre_test}",
      "exito": true/false,
      "pasos_completados": N,
      "error": "descripción si falló o null"
    }}
    """

    # [bucle de control]
    return {"test": nombre_test, "exito": True, "pasos_completados": len(pasos_test), "error": None}

# Ejemplo: test de login
resultado = ejecutar_test_visual(
    url_app="https://mi-app.com",
    pasos_test=[
        "Haz clic en 'Iniciar sesión'",
        "Introduce el email 'test@empresa.com'",
        "Introduce la contraseña del entorno de test",
        "Haz clic en 'Entrar'",
        "Verifica que aparece el dashboard con el nombre del usuario",
    ],
    nombre_test="Login exitoso",
)
print(resultado)
```

## Integración con n8n

```python
# Endpoint FastAPI que n8n puede llamar como webhook
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TareaComputerUse(BaseModel):
    instruccion: str
    max_iteraciones: int = 20

@app.post("/ejecutar")
async def ejecutar_tarea(tarea: TareaComputerUse):
    """
    n8n llama a este endpoint con la instrucción.
    Computer Use la ejecuta y devuelve el resultado.
    """
    # resultado = agente_computer_use(tarea.instruccion, tarea.max_iteraciones)
    resultado = "Tarea completada"  # placeholder
    return {"resultado": resultado, "exito": True}
```

## ROI y métricas

```python
from dataclasses import dataclass

@dataclass
class MetricasRPA:
    tiempo_manual_horas: float
    tareas_por_dia: int
    coste_hora_empleado: float  # €/hora
    coste_por_llamada_api: float  # € por tarea automatizada

    @property
    def ahorro_diario(self) -> float:
        coste_manual = self.tiempo_manual_horas * self.tareas_por_dia * self.coste_hora_empleado
        coste_automatizado = self.tareas_por_dia * self.coste_por_llamada_api
        return coste_manual - coste_automatizado

    @property
    def roi_mensual(self) -> float:
        return self.ahorro_diario * 22  # días laborables

# Ejemplo: procesamiento de facturas
metricas = MetricasRPA(
    tiempo_manual_horas=0.25,   # 15 min por factura
    tareas_por_dia=80,           # 80 facturas/día
    coste_hora_empleado=25.0,   # 25€/hora
    coste_por_llamada_api=0.15, # ~0.15€ por factura procesada
)
print(f"Ahorro diario: {metricas.ahorro_diario:.0f}€")
print(f"ROI mensual: {metricas.roi_mensual:.0f}€")
# Ahorro diario: 488€ / ROI mensual: 10.736€
```

## Resumen

- Computer Use habilita RPA inteligente para procesos que el RPA clásico no puede manejar
- Casos principales: ERPs legacy, clasificación de emails, sync de CRM, testing visual
- Integra fácilmente con n8n vía webhook → FastAPI
- El ROI es rápido cuando automatizas tareas repetitivas de alto volumen
