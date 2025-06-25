"""
SESIÓN 9: VISUALIZACIÓN BÁSICA DE DATOS DE POZOS
Laboratorio 1: Visualización de Series Temporales de Producción - SOLUCIÓN

OBJETIVO:
Crear visualizaciones efectivas de series temporales para análisis
de tendencias de producción y comunicación de resultados.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import timedelta

print("=== LABORATORIO 1: SERIES TEMPORALES DE PRODUCCIÓN - SOLUCIÓN ===")
print()

# Configuración inicial
plt.style.use('seaborn-v0_8-whitegrid')
ruta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos', 'produccion_historica.csv')

# PARTE 1: CARGA Y PREPARACIÓN DE DATOS
print("PARTE 1: Carga y preparación de datos")
print("-" * 50)

# Cargar el archivo CSV y convertir la columna 'fecha' a datetime
df = pd.read_csv(ruta_datos)
df['fecha'] = pd.to_datetime(df['fecha'])

# Mostrar información básica del dataset
print(f"Número de registros: {len(df)}")
print(f"Pozos únicos: {df['pozo'].nunique()}")
print(f"Lista de pozos: {', '.join(df['pozo'].unique())}")
print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
print("\nPrimeras 5 filas:")
print(df.head())

print("\n" + "="*50 + "\n")

# PARTE 2: VISUALIZACIÓN BÁSICA DE SERIE TEMPORAL
print("PARTE 2: Serie temporal de un pozo individual")
print("-" * 50)

# Crear un gráfico de línea mostrando la producción del POZO-A-001
pozo_a001 = df[df['pozo'] == 'POZO-A-001']

plt.figure(figsize=(12, 6))
plt.plot(pozo_a001['fecha'], pozo_a001['barriles_diarios'], 
         'b-', linewidth=2, marker='o', markersize=6, label='POZO-A-001')
plt.title('Producción Diaria - POZO-A-001 (Enero 2024)', fontsize=16, fontweight='bold')
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Barriles por Día (bpd)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

# PARTE 3: COMPARACIÓN DE MÚLTIPLES POZOS
print("PARTE 3: Comparación de producción entre pozos")
print("-" * 50)

# Calcular producción promedio para identificar el mejor pozo
produccion_promedio = df.groupby('pozo')['barriles_diarios'].mean().sort_values(ascending=False)
mejor_pozo = produccion_promedio.index[0]

# Crear gráfico comparando todos los pozos
plt.figure(figsize=(14, 8))

colores = {'POZO-A-001': '#1f77b4', 'POZO-A-002': '#aec7e8', 
           'POZO-B-001': '#ff7f0e', 'POZO-B-002': '#ffbb78', 
           'POZO-C-001': '#2ca02c'}

for pozo in df['pozo'].unique():
    datos_pozo = df[df['pozo'] == pozo]
    linewidth = 3 if pozo == mejor_pozo else 2
    alpha = 1.0 if pozo == mejor_pozo else 0.7
    
    plt.plot(datos_pozo['fecha'], datos_pozo['barriles_diarios'], 
             color=colores[pozo], linewidth=linewidth, alpha=alpha,
             marker='o' if pozo == mejor_pozo else None, 
             markersize=4 if pozo == mejor_pozo else None,
             label=f'{pozo} ({"Mejor" if pozo == mejor_pozo else f"Prom: {produccion_promedio[pozo]:.0f}"})')

plt.title('Comparación de Producción Diaria - Todos los Pozos', fontsize=16, fontweight='bold')
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Barriles por Día (bpd)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"\n✓ Pozo con mayor producción promedio: {mejor_pozo} ({produccion_promedio[mejor_pozo]:.0f} bpd)")

print("\n" + "="*50 + "\n")

# PARTE 4: ANÁLISIS DE TENDENCIAS
print("PARTE 4: Análisis de tendencias con media móvil")
print("-" * 50)

# Análisis para POZO-B-001
pozo_b001 = df[df['pozo'] == 'POZO-B-001'].copy()
pozo_b001 = pozo_b001.sort_values('fecha')

# Calcular media móvil de 5 días
pozo_b001['media_movil_5d'] = pozo_b001['barriles_diarios'].rolling(window=5, center=True).mean()

# Identificar máximos y mínimos
idx_max = pozo_b001['barriles_diarios'].idxmax()
idx_min = pozo_b001['barriles_diarios'].idxmin()
punto_max = pozo_b001.loc[idx_max]
punto_min = pozo_b001.loc[idx_min]

plt.figure(figsize=(12, 7))

# Producción diaria
plt.plot(pozo_b001['fecha'], pozo_b001['barriles_diarios'], 
         'lightblue', linewidth=1.5, alpha=0.6, label='Producción diaria')

# Media móvil
plt.plot(pozo_b001['fecha'], pozo_b001['media_movil_5d'], 
         'darkblue', linewidth=3, label='Media móvil 5 días')

# Marcar máximo
plt.scatter(punto_max['fecha'], punto_max['barriles_diarios'], 
           color='green', s=200, zorder=5, edgecolor='darkgreen', linewidth=2)
plt.annotate(f'Máximo: {punto_max["barriles_diarios"]:.0f} bpd',
             xy=(punto_max['fecha'], punto_max['barriles_diarios']),
             xytext=(punto_max['fecha'] + timedelta(days=1), punto_max['barriles_diarios'] + 10),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, fontweight='bold')

# Marcar mínimo
plt.scatter(punto_min['fecha'], punto_min['barriles_diarios'], 
           color='red', s=200, zorder=5, edgecolor='darkred', linewidth=2)
plt.annotate(f'Mínimo: {punto_min["barriles_diarios"]:.0f} bpd',
             xy=(punto_min['fecha'], punto_min['barriles_diarios']),
             xytext=(punto_min['fecha'] + timedelta(days=1), punto_min['barriles_diarios'] - 10),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, fontweight='bold')

plt.title('Análisis de Tendencias - POZO-B-001', fontsize=16, fontweight='bold')
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Barriles por Día (bpd)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

# PARTE 5: DASHBOARD DE PRODUCCIÓN
print("PARTE 5: Dashboard integrado de producción")
print("-" * 50)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Dashboard de Producción - Enero 2024', fontsize=18, fontweight='bold')

# 1. Producción total diaria
produccion_total = df.groupby('fecha')['barriles_diarios'].sum()
ax1.plot(produccion_total.index, produccion_total.values, 
         'b-', linewidth=3, marker='o', markersize=6)
ax1.set_title('Producción Total Diaria', fontsize=14)
ax1.set_ylabel('Total bpd')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Producción promedio por pozo
prod_promedio_pozo = df.groupby('pozo')['barriles_diarios'].mean().sort_values(ascending=False)
colores_barras = [colores[pozo] for pozo in prod_promedio_pozo.index]
bars = ax2.bar(range(len(prod_promedio_pozo)), prod_promedio_pozo.values, color=colores_barras)
ax2.set_title('Producción Promedio por Pozo', fontsize=14)
ax2.set_ylabel('Promedio bpd')
ax2.set_xticks(range(len(prod_promedio_pozo)))
ax2.set_xticklabels(prod_promedio_pozo.index, rotation=45)

# Añadir valores en las barras
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

# 3. Evolución de presión promedio
presion_promedio = df.groupby('fecha')['presion_psi'].mean()
ax3.plot(presion_promedio.index, presion_promedio.values, 
         'g-', linewidth=2.5)
ax3.set_title('Presión Promedio Diaria', fontsize=14)
ax3.set_ylabel('Presión (psi)')
ax3.set_xlabel('Fecha')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Relación temperatura vs producción
scatter = ax4.scatter(df['temperatura_f'], df['barriles_diarios'], 
                     c=df['presion_psi'], cmap='viridis', alpha=0.6, s=50)
ax4.set_title('Temperatura vs Producción', fontsize=14)
ax4.set_xlabel('Temperatura (°F)')
ax4.set_ylabel('Producción (bpd)')
ax4.grid(True, alpha=0.3)

# Añadir colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Presión (psi)', rotation=270, labelpad=20)

plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

# PARTE 6: VISUALIZACIÓN PARA REPORTE EJECUTIVO
print("PARTE 6: Gráfico profesional para reporte")
print("-" * 50)

# Preparar datos por campo
df['campo'] = df['pozo'].str.extract(r'POZO-([A-C])')
df['campo'] = 'CAMPO-' + df['campo']

# Calcular producción acumulada por campo
produccion_campo = df.groupby(['fecha', 'campo'])['barriles_diarios'].sum().reset_index()
produccion_campo = produccion_campo.sort_values(['campo', 'fecha'])
produccion_campo['produccion_acumulada'] = produccion_campo.groupby('campo')['barriles_diarios'].cumsum()

# Proyección simple (últimos 5 días de tendencia)
ultima_fecha = df['fecha'].max()
fechas_proyeccion = pd.date_range(start=ultima_fecha + timedelta(days=1), periods=5)

plt.figure(figsize=(14, 8))

# Colores corporativos
colores_corporativos = {'CAMPO-A': '#003f5c', 'CAMPO-B': '#bc5090', 'CAMPO-C': '#ffa600'}

for campo in ['CAMPO-A', 'CAMPO-B', 'CAMPO-C']:
    datos_campo = produccion_campo[produccion_campo['campo'] == campo]
    
    # Datos históricos
    plt.plot(datos_campo['fecha'], datos_campo['produccion_acumulada'], 
             color=colores_corporativos[campo], linewidth=3, label=f'{campo} (Histórico)')
    
    # Calcular tendencia de los últimos 5 días
    ultimos_5_dias = datos_campo.tail(5)
    if len(ultimos_5_dias) > 1:
        tendencia_diaria = ultimos_5_dias['barriles_diarios'].mean()
        ultima_produccion = datos_campo['produccion_acumulada'].iloc[-1]
        
        # Proyección
        proyeccion = [ultima_produccion + tendencia_diaria * i for i in range(1, 6)]
        plt.plot(fechas_proyeccion, proyeccion, '--', 
                color=colores_corporativos[campo], linewidth=2, alpha=0.7,
                label=f'{campo} (Proyección)')

# Añadir anotaciones
plt.annotate('Inicio del período de análisis',
             xy=(df['fecha'].min(), 0), xytext=(df['fecha'].min() + timedelta(days=2), 15000),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

plt.annotate('Proyección a 5 días',
             xy=(fechas_proyeccion[0], 50000), xytext=(fechas_proyeccion[2], 55000),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))

# Formato profesional
plt.title('Producción Acumulada por Campo con Proyección a 5 Días', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Producción Acumulada (barriles)', fontsize=14)
plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.2, linestyle='--')

# Personalizar ejes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.xticks(rotation=45)
plt.tight_layout()

# Guardar en alta resolución
output_dir = os.path.join(os.path.dirname(__file__), '..', 'reportes')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'produccion_acumulada_ejecutivo.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✓ Gráfico guardado en: reportes/produccion_acumulada_ejecutivo.png")

print("\n" + "="*50 + "\n")

# DESAFÍO ADICIONAL
print("DESAFÍO: Análisis de eficiencia de producción")
print("-" * 50)

# Calcular eficiencia
df['produccion_maxima'] = df.groupby('pozo')['barriles_diarios'].transform('max')
df['eficiencia'] = (df['barriles_diarios'] / df['produccion_maxima']) * 100

# Preparar datos para heatmap
pivot_eficiencia = df.pivot_table(values='eficiencia', index='pozo', columns='fecha')

# Crear heatmap
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Heatmap de eficiencia
im = ax1.imshow(pivot_eficiencia.values, cmap='RdYlGn', aspect='auto', vmin=70, vmax=100)
ax1.set_yticks(range(len(pivot_eficiencia.index)))
ax1.set_yticklabels(pivot_eficiencia.index)
ax1.set_xticks(range(0, len(pivot_eficiencia.columns), 2))
ax1.set_xticklabels([d.strftime('%d/%m') for d in pivot_eficiencia.columns[::2]], rotation=45)
ax1.set_title('Heatmap de Eficiencia Diaria por Pozo (%)', fontsize=16, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Eficiencia (%)', rotation=270, labelpad=20)

# Identificar patrones
eficiencia_promedio = pivot_eficiencia.mean(axis=1)
dias_baja_eficiencia = (pivot_eficiencia < 85).sum(axis=1)

# Gráfico de barras con patrones identificados
x = range(len(eficiencia_promedio))
bars = ax2.bar(x, eficiencia_promedio.values, color=['red' if e < 90 else 'green' for e in eficiencia_promedio.values])
ax2.axhline(y=90, color='red', linestyle='--', label='Umbral objetivo (90%)')
ax2.set_xticks(x)
ax2.set_xticklabels(eficiencia_promedio.index, rotation=45)
ax2.set_ylabel('Eficiencia Promedio (%)')
ax2.set_title('Eficiencia Promedio por Pozo', fontsize=14)
ax2.legend()

# Añadir texto con días de baja eficiencia
for i, (bar, dias) in enumerate(zip(bars, dias_baja_eficiencia.values)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{dias}d<85%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n✓ Patrones identificados:")
print(f"- Pozos con eficiencia promedio < 90%: {list(eficiencia_promedio[eficiencia_promedio < 90].index)}")
print(f"- Pozo más eficiente: {eficiencia_promedio.idxmax()} ({eficiencia_promedio.max():.1f}%)")
print(f"- Pozo menos eficiente: {eficiencia_promedio.idxmin()} ({eficiencia_promedio.min():.1f}%)")

print("\n✓ Recomendaciones basadas en el análisis:")
print("1. Priorizar mantenimiento en pozos con eficiencia < 90%")
print("2. Investigar causas de baja eficiencia en días específicos")
print("3. Replicar prácticas del pozo más eficiente")
print("4. Establecer sistema de alertas para eficiencia < 85%")

print("\n✅ Laboratorio completado exitosamente")