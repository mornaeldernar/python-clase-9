"""
SESIÓN 9: VISUALIZACIÓN BÁSICA DE DATOS DE POZOS
Demo 2: Integración Pandas + Matplotlib

Este demo muestra cómo combinar el poder de Pandas para manipular datos
con Matplotlib para crear visualizaciones efectivas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

print("=== DEMO 2: INTEGRACIÓN PANDAS + MATPLOTLIB ===")
print()

# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# 1. CARGAR Y VISUALIZAR DATOS CON PANDAS
print("1. Cargando datos de producción histórica")
print("-" * 50)

# Cargar datos
ruta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos', 'produccion_historica.csv')
df = pd.read_csv(ruta_datos)
df['fecha'] = pd.to_datetime(df['fecha'])

print(f"Datos cargados: {len(df)} registros")
print(f"Pozos únicos: {df['pozo'].nunique()}")
print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
print("\nPrimeras filas:")
print(df.head())

input("\nPresiona Enter para continuar...")

# 2. VISUALIZACIÓN DIRECTA CON PANDAS
print("\n2. Métodos de visualización integrados en Pandas")
print("-" * 50)

# Gráfico de líneas simple
plt.figure(figsize=(12, 6))
for pozo in df['pozo'].unique():
    datos_pozo = df[df['pozo'] == pozo]
    plt.plot(datos_pozo['fecha'], datos_pozo['barriles_diarios'], 
             label=pozo, linewidth=2, marker='o', markersize=4)

plt.title('Producción Diaria por Pozo - Método plot() de Pandas', fontsize=14)
plt.xlabel('Fecha')
plt.ylabel('Barriles por Día')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar...")

# 3. AGRUPACIÓN Y VISUALIZACIÓN
print("\n3. Agrupación de datos y visualización de agregados")
print("-" * 50)

# Producción promedio por pozo
produccion_promedio = df.groupby('pozo')['barriles_diarios'].agg(['mean', 'std'])
produccion_promedio = produccion_promedio.sort_values('mean', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de barras con error bars
produccion_promedio['mean'].plot(kind='bar', ax=ax1, color='skyblue', 
                                yerr=produccion_promedio['std'], 
                                capsize=5, edgecolor='black', linewidth=1.5)
ax1.set_title('Producción Promedio por Pozo (±1 std)', fontsize=14)
ax1.set_xlabel('Pozo')
ax1.set_ylabel('Producción Promedio (bpd)')
ax1.tick_params(axis='x', rotation=45)

# Añadir valores en las barras
for i, (idx, row) in enumerate(produccion_promedio.iterrows()):
    ax1.text(i, row['mean'] + row['std'] + 20, f"{row['mean']:.0f}", 
             ha='center', va='bottom', fontweight='bold')

# Gráfico horizontal para mejor legibilidad
produccion_promedio['mean'].plot(kind='barh', ax=ax2, color='lightcoral')
ax2.set_title('Ranking de Producción', fontsize=14)
ax2.set_xlabel('Producción Promedio (bpd)')
ax2.invert_yaxis()

plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar...")

# 4. ANÁLISIS TEMPORAL CON PANDAS
print("\n4. Análisis temporal y tendencias")
print("-" * 50)

# Crear pivot table para análisis temporal
pivot_produccion = df.pivot_table(values='barriles_diarios', 
                                  index='fecha', 
                                  columns='pozo')

# Calcular media móvil
ventana = 3
media_movil = pivot_produccion.rolling(window=ventana).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Gráfico 1: Producción diaria
for col in pivot_produccion.columns:
    ax1.plot(pivot_produccion.index, pivot_produccion[col], 
             alpha=0.3, linewidth=1, label=f'{col} (diario)')
    ax1.plot(media_movil.index, media_movil[col], 
             linewidth=2.5, label=f'{col} (MM-{ventana}d)')

ax1.set_title(f'Producción Diaria y Media Móvil ({ventana} días)', fontsize=14)
ax1.set_ylabel('Producción (bpd)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Gráfico 2: Producción acumulada
produccion_acumulada = pivot_produccion.cumsum()
for col in produccion_acumulada.columns:
    ax2.plot(produccion_acumulada.index, produccion_acumulada[col], 
             linewidth=2.5, label=col)

ax2.set_title('Producción Acumulada', fontsize=14)
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Producción Acumulada (barriles)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Formatear fechas
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar...")

# 5. CORRELACIONES Y HEATMAPS
print("\n5. Análisis de correlaciones con visualización")
print("-" * 50)

# Preparar datos para correlación
correlacion_data = df.pivot_table(index='fecha', 
                                 columns='pozo', 
                                 values=['barriles_diarios', 'presion_psi', 'temperatura_f'])

# Calcular correlación entre pozos para producción
corr_produccion = correlacion_data['barriles_diarios'].corr()

plt.figure(figsize=(10, 8))
im = plt.imshow(corr_produccion, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Añadir valores de correlación
for i in range(len(corr_produccion)):
    for j in range(len(corr_produccion)):
        plt.text(j, i, f'{corr_produccion.iloc[i, j]:.2f}', 
                ha='center', va='center', 
                color='white' if abs(corr_produccion.iloc[i, j]) > 0.5 else 'black')

plt.colorbar(im, label='Correlación')
plt.xticks(range(len(corr_produccion.columns)), corr_produccion.columns, rotation=45)
plt.yticks(range(len(corr_produccion.index)), corr_produccion.index)
plt.title('Matriz de Correlación - Producción entre Pozos', fontsize=14)
plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar...")

# 6. ANÁLISIS MULTIVARIABLE
print("\n6. Visualización de múltiples variables")
print("-" * 50)

# Seleccionar un pozo para análisis detallado
pozo_analisis = 'POZO-B-001'
datos_pozo = df[df['pozo'] == pozo_analisis].copy()

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Variable 1: Producción
ax1 = axes[0]
ax1.plot(datos_pozo['fecha'], datos_pozo['barriles_diarios'], 
         'b-', linewidth=2, label='Producción')
ax1.set_ylabel('Producción (bpd)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title(f'Análisis Multivariable - {pozo_analisis}', fontsize=14)
ax1.grid(True, alpha=0.3)

# Variable 2: Presión
ax2 = axes[1]
ax2.plot(datos_pozo['fecha'], datos_pozo['presion_psi'], 
         'g-', linewidth=2, label='Presión')
ax2.set_ylabel('Presión (psi)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.grid(True, alpha=0.3)

# Variable 3: Temperatura
ax3 = axes[2]
ax3.plot(datos_pozo['fecha'], datos_pozo['temperatura_f'], 
         'r-', linewidth=2, label='Temperatura')
ax3.set_ylabel('Temperatura (°F)', color='r')
ax3.tick_params(axis='y', labelcolor='r')
ax3.set_xlabel('Fecha')
ax3.grid(True, alpha=0.3)

# Rotar fechas
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar...")

# 7. BOXPLOTS Y DISTRIBUCIONES
print("\n7. Análisis de distribuciones por categoría")
print("-" * 50)

# Cargar datos mensuales
ruta_mensual = os.path.join(os.path.dirname(__file__), '..', 'datos', 'resumen_mensual.csv')
df_mensual = pd.read_csv(ruta_mensual)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot de producción por campo
campos_data = []
campos_labels = []
for campo in df_mensual['campo'].unique():
    campos_data.append(df_mensual[df_mensual['campo'] == campo]['promedio_diario'])
    campos_labels.append(campo)

bp = ax1.boxplot(campos_data, labels=campos_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
    patch.set_facecolor(color)
    
ax1.set_title('Distribución de Producción Promedio por Campo', fontsize=14)
ax1.set_ylabel('Producción Promedio Diaria (bpd)')
ax1.grid(True, alpha=0.3, axis='y')

# Scatter plot: Eficiencia vs Producción
scatter = ax2.scatter(df_mensual['promedio_diario'], 
                     df_mensual['eficiencia_operativa'],
                     c=df_mensual['campo'].astype('category').cat.codes,
                     s=100, alpha=0.6, cmap='viridis')

# Añadir etiquetas de campo
for campo in df_mensual['campo'].unique():
    datos_campo = df_mensual[df_mensual['campo'] == campo]
    ax2.annotate(campo, 
                xy=(datos_campo['promedio_diario'].mean(), 
                    datos_campo['eficiencia_operativa'].mean()),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

ax2.set_title('Eficiencia Operativa vs Producción', fontsize=14)
ax2.set_xlabel('Producción Promedio Diaria (bpd)')
ax2.set_ylabel('Eficiencia Operativa (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. GUARDADO DE FIGURAS
print("\n8. Guardando visualizaciones en alta calidad")
print("-" * 50)

# Crear una visualización para guardar
fig, ax = plt.subplots(figsize=(10, 6))

# Gráfico de producción total por campo
produccion_campo = df.groupby(['fecha', 'pozo']).agg({
    'barriles_diarios': 'sum'
}).reset_index()

# Extraer campo del nombre del pozo
produccion_campo['campo'] = produccion_campo['pozo'].str.extract(r'(POZO-[A-C])')

# Agrupar por campo
produccion_por_campo = produccion_campo.groupby(['fecha', 'campo'])['barriles_diarios'].sum().reset_index()

# Crear gráfico
for campo in produccion_por_campo['campo'].unique():
    datos = produccion_por_campo[produccion_por_campo['campo'] == campo]
    ax.plot(datos['fecha'], datos['barriles_diarios'], 
            linewidth=3, label=campo.replace('POZO-', 'CAMPO-'), 
            marker='o', markersize=6)

ax.set_title('Producción Total por Campo - Enero 2024', fontsize=16, fontweight='bold')
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Producción Total (bpd)', fontsize=12)
ax.legend(title='Campo', fontsize=11)
ax.grid(True, alpha=0.3)

# Formatear ejes
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()

# Guardar en diferentes formatos
output_dir = os.path.join(os.path.dirname(__file__), '..', 'graficos_generados')
os.makedirs(output_dir, exist_ok=True)

# Alta resolución para reportes
fig.savefig(os.path.join(output_dir, 'produccion_campos_alta_res.png'), 
            dpi=300, bbox_inches='tight')

# Formato vectorial para presentaciones
fig.savefig(os.path.join(output_dir, 'produccion_campos.pdf'), 
            format='pdf', bbox_inches='tight')

# Formato web
fig.savefig(os.path.join(output_dir, 'produccion_campos_web.png'), 
            dpi=72, bbox_inches='tight')

print("✅ Visualizaciones guardadas en:")
print(f"   - {output_dir}/produccion_campos_alta_res.png (300 DPI)")
print(f"   - {output_dir}/produccion_campos.pdf (Vectorial)")
print(f"   - {output_dir}/produccion_campos_web.png (72 DPI)")

plt.show()

print("\n✅ Demo completado: Integración Pandas + Matplotlib")
print("\nConceptos cubiertos:")
print("- Visualización directa desde DataFrames")
print("- Agrupación y agregación con visualización")
print("- Análisis temporal con medias móviles")
print("- Matrices de correlación y heatmaps")
print("- Análisis multivariable")
print("- Boxplots y distribuciones")
print("- Guardado de figuras en múltiples formatos")