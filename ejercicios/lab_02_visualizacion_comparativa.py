"""
SESIÓN 9: VISUALIZACIÓN BÁSICA DE DATOS DE POZOS
Laboratorio 2: Visualización Comparativa y Análisis Visual

OBJETIVO:
Desarrollar habilidades para crear visualizaciones comparativas
que faciliten la toma de decisiones basada en datos.

CONTEXTO EMPRESARIAL:
El cliente necesita identificar oportunidades de optimización
comparando el rendimiento entre diferentes pozos y campos.
Tu análisis visual debe revelar patrones y anomalías.

DATOS DISPONIBLES:
- produccion_historica.csv: Datos de producción diaria
- resumen_mensual.csv: Métricas agregadas por campo
- comparacion_pozos.json: Información detallada de cada pozo
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

print("=== LABORATORIO 2: VISUALIZACIÓN COMPARATIVA ===")
print()

# Rutas de datos
base_path = os.path.join(os.path.dirname(__file__), '..', 'datos')
ruta_produccion = os.path.join(base_path, 'produccion_historica.csv')
ruta_mensual = os.path.join(base_path, 'resumen_mensual.csv')
ruta_json = os.path.join(base_path, 'comparacion_pozos.json')

# PARTE 1: ANÁLISIS COMPARATIVO DE POZOS
print("PARTE 1: Comparación de características de pozos")
print("-" * 50)

# TODO: Cargar datos del archivo JSON
# Crear visualizaciones que comparen:
# 1. Profundidad vs Producción acumulada (scatter plot)
# 2. Años de operación vs Costo operativo (con tamaño = producción)
# 3. Distribución de tipos de pozos (pie chart)

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 2: ANÁLISIS DE CORRELACIONES
print("PARTE 2: Matriz de correlaciones entre variables")
print("-" * 50)

# TODO: Usando produccion_historica.csv:
# 1. Crear matriz de correlación entre: producción, presión, temperatura, API gravity
# 2. Visualizar como heatmap con valores
# 3. Identificar las correlaciones más fuertes
# 4. Crear scatter plots para las 2 correlaciones más significativas

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 3: ANÁLISIS DE RENDIMIENTO POR CAMPO
print("PARTE 3: Comparación de rendimiento entre campos")
print("-" * 50)

# TODO: Usando resumen_mensual.csv:
# 1. Crear gráfico de líneas mostrando evolución de eficiencia por campo
# 2. Añadir área sombreada para mostrar el rango de eficiencia objetivo (90-95%)
# 3. Calcular y mostrar tendencia lineal para cada campo
# 4. Identificar el campo con mejor mejora en eficiencia

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 4: ANÁLISIS DE VARIABILIDAD
print("PARTE 4: Análisis de variabilidad en la producción")
print("-" * 50)

# TODO: Analizar la variabilidad de producción:
# 1. Calcular coeficiente de variación para cada pozo
# 2. Crear boxplot mostrando distribución de producción por pozo
# 3. Añadir puntos para outliers
# 4. Crear gráfico de violín como alternativa
# 5. Identificar pozos más estables vs más volátiles

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 5: ANÁLISIS MULTIDIMENSIONAL
print("PARTE 5: Visualización multidimensional")
print("-" * 50)

# TODO: Crear visualización que muestre simultáneamente:
# 1. Eje X: Presión promedio
# 2. Eje Y: Temperatura promedio
# 3. Tamaño de punto: Producción promedio
# 4. Color: API Gravity
# 5. Forma: Tipo de pozo (si está disponible)
# 6. Añadir líneas de regresión por grupo

# Tu código aquí

print("\n" + "="*50 + "\n")

# PARTE 6: DASHBOARD EJECUTIVO
print("PARTE 6: Dashboard ejecutivo integral")
print("-" * 50)

# TODO: Crear un dashboard ejecutivo con 6 paneles:
# Panel 1: KPI principal - Producción total actual
# Panel 2: Tendencia de producción últimos 7 días
# Panel 3: Ranking de pozos por eficiencia
# Panel 4: Mapa de calor de producción por día y pozo
# Panel 5: Indicadores de alerta (pozos bajo umbral)
# Panel 6: Proyección a 30 días

# Tu código aquí

print("\n" + "="*50 + "\n")

# DESAFÍO: STORYTELLING CON DATOS
print("DESAFÍO: Narrativa visual de datos")
print("-" * 50)

# TODO: Crear una secuencia de 3-4 visualizaciones que cuenten una historia:
# 1. "El problema": Identificar caída en producción
# 2. "El análisis": Mostrar correlación con otras variables
# 3. "La causa raíz": Revelar el factor principal
# 4. "La solución": Proponer optimización con proyección de mejora

# Tu código aquí

print("\n✅ Laboratorio completado")
print("\nCRITERIOS DE EVALUACIÓN:")
print("- [ ] Análisis comparativo implementado correctamente")
print("- [ ] Correlaciones identificadas y visualizadas")
print("- [ ] Análisis de variabilidad completo")
print("- [ ] Visualizaciones multidimensionales efectivas")
print("- [ ] Dashboard ejecutivo funcional")
print("- [ ] Narrativa visual coherente y persuasiva")
print("- [ ] Insights accionables identificados")