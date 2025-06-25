"""
SESIÓN 9: VISUALIZACIÓN BÁSICA DE DATOS DE POZOS
Laboratorio 1: Visualización de Series Temporales de Producción

OBJETIVO:
Crear visualizaciones efectivas de series temporales para análisis
de tendencias de producción y comunicación de resultados.

CONTEXTO EMPRESARIAL:
Como analista en Meridian Consulting, necesitas generar visualizaciones
que comuniquen claramente las tendencias de producción a los directivos
de las empresas petroleras clientes.

DATOS DISPONIBLES:
- produccion_historica.csv: Datos diarios de producción de 5 pozos
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

print("=== LABORATORIO 1: SERIES TEMPORALES DE PRODUCCIÓN ===")
print()

# Configuración inicial
plt.style.use('seaborn-v0_8-whitegrid')
ruta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos', 'produccion_historica.csv')

# PARTE 1: CARGA Y PREPARACIÓN DE DATOS
print("PARTE 1: Carga y preparación de datos")
print("-" * 50)

# TODO: Cargar el archivo CSV y convertir la columna 'fecha' a datetime
df = None  # Reemplazar con tu código

# TODO: Mostrar información básica del dataset
# - Número de registros
# - Pozos únicos
# - Rango de fechas
# - Primeras 5 filas

print("\n" + "="*50 + "\n")

# PARTE 2: VISUALIZACIÓN BÁSICA DE SERIE TEMPORAL
print("PARTE 2: Serie temporal de un pozo individual")
print("-" * 50)

# TODO: Crear un gráfico de línea mostrando la producción del POZO-A-001
# Requisitos:
# - Tamaño de figura: 12x6
# - Título descriptivo
# - Etiquetas en los ejes
# - Grid activado
# - Formato de fecha legible en el eje X

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 3: COMPARACIÓN DE MÚLTIPLES POZOS
print("PARTE 3: Comparación de producción entre pozos")
print("-" * 50)

# TODO: Crear un gráfico comparando la producción de todos los pozos
# Requisitos:
# - Todos los pozos en el mismo gráfico
# - Diferentes colores para cada pozo
# - Leyenda clara
# - Destacar el pozo con mayor producción promedio

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 4: ANÁLISIS DE TENDENCIAS
print("PARTE 4: Análisis de tendencias con media móvil")
print("-" * 50)

# TODO: Para el POZO-B-001:
# 1. Graficar la producción diaria (línea delgada, con transparencia)
# 2. Calcular y graficar la media móvil de 5 días (línea gruesa)
# 3. Identificar y marcar el punto de producción máxima
# 4. Identificar y marcar el punto de producción mínima

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 5: DASHBOARD DE PRODUCCIÓN
print("PARTE 5: Dashboard integrado de producción")
print("-" * 50)

# TODO: Crear un dashboard con 4 subplots:
# 1. Superior izquierda: Producción total diaria (suma de todos los pozos)
# 2. Superior derecha: Producción promedio por pozo (gráfico de barras)
# 3. Inferior izquierda: Evolución de la presión promedio
# 4. Inferior derecha: Relación temperatura vs producción (scatter plot)

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 6: VISUALIZACIÓN PARA REPORTE EJECUTIVO
print("PARTE 6: Gráfico profesional para reporte")
print("-" * 50)

# TODO: Crear una visualización profesional que muestre:
# 1. Producción acumulada por campo (CAMPO-A, CAMPO-B, CAMPO-C)
# 2. Añadir anotaciones para eventos importantes
# 3. Incluir proyección para los próximos 5 días (línea punteada)
# 4. Formato profesional con colores corporativos
# 5. Guardar en alta resolución (300 DPI)

# Tu código aquí

print("\n" + "="*50 + "\n")

# DESAFÍO ADICIONAL
print("DESAFÍO: Análisis de eficiencia de producción")
print("-" * 50)

# TODO: Crear una visualización que muestre:
# 1. Calcular la eficiencia como: (producción actual / producción máxima del pozo) * 100
# 2. Crear un heatmap mostrando la eficiencia diaria de cada pozo
# 3. Identificar patrones de baja eficiencia
# 4. Proponer mejoras basadas en los patrones observados

# Tu código aquí

print("\n✅ Laboratorio completado")
print("\nCRITERIOS DE EVALUACIÓN:")
print("- [ ] Correcta carga y preparación de datos")
print("- [ ] Gráficos claros y bien etiquetados")
print("- [ ] Uso apropiado de colores y estilos")
print("- [ ] Análisis de tendencias implementado")
print("- [ ] Dashboard funcional y informativo")
print("- [ ] Visualización profesional para reportes")
print("- [ ] Código limpio y documentado")