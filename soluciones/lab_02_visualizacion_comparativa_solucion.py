"""
SESIÓN 9: VISUALIZACIÓN BÁSICA DE DATOS DE POZOS
Laboratorio 2: Visualización Comparativa y Análisis Visual - SOLUCIÓN

OBJETIVO:
Desarrollar habilidades para crear visualizaciones comparativas
que faciliten la toma de decisiones basada en datos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from scipy import stats

print("=== LABORATORIO 2: VISUALIZACIÓN COMPARATIVA - SOLUCIÓN ===")
print()

# Rutas de datos
base_path = os.path.join(os.path.dirname(__file__), '..', 'datos')
ruta_produccion = os.path.join(base_path, 'produccion_historica.csv')
ruta_mensual = os.path.join(base_path, 'resumen_mensual.csv')
ruta_json = os.path.join(base_path, 'comparacion_pozos.json')

# PARTE 1: ANÁLISIS COMPARATIVO DE POZOS
print("PARTE 1: Comparación de características de pozos")
print("-" * 50)

# Cargar datos JSON
with open(ruta_json, 'r') as f:
    datos_json = json.load(f)
    
pozos_info = pd.DataFrame(datos_json['pozos'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# 1. Profundidad vs Producción acumulada
scatter1 = ax1.scatter(pozos_info['profundidad_m'], 
                       pozos_info['produccion_acumulada']/1000000,
                       s=200, alpha=0.7, c=pozos_info['años_operacion'],
                       cmap='viridis', edgecolors='black', linewidth=2)

# Añadir línea de tendencia
z = np.polyfit(pozos_info['profundidad_m'], pozos_info['produccion_acumulada']/1000000, 1)
p = np.poly1d(z)
ax1.plot(pozos_info['profundidad_m'], p(pozos_info['profundidad_m']), 
         "r--", linewidth=2, alpha=0.8)

ax1.set_xlabel('Profundidad (m)', fontsize=12)
ax1.set_ylabel('Producción Acumulada (MM barriles)', fontsize=12)
ax1.set_title('Profundidad vs Producción Acumulada', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Colorbar
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Años de Operación', rotation=270, labelpad=20)

# Añadir etiquetas de pozos
for idx, row in pozos_info.iterrows():
    ax1.annotate(row['id'].split('-')[-1], 
                (row['profundidad_m'], row['produccion_acumulada']/1000000),
                fontsize=8, ha='center', va='bottom')

# 2. Años de operación vs Costo operativo
scatter2 = ax2.scatter(pozos_info['años_operacion'], 
                       pozos_info['costo_operativo_diario'],
                       s=pozos_info['produccion_acumulada']/10000,
                       alpha=0.7, c=['blue', 'blue', 'red', 'green', 'blue'],
                       edgecolors='black', linewidth=2)

ax2.set_xlabel('Años de Operación', fontsize=12)
ax2.set_ylabel('Costo Operativo Diario ($)', fontsize=12)
ax2.set_title('Años vs Costo (tamaño = producción)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Leyenda para tipos
for tipo, color in [('Vertical', 'blue'), ('Horizontal', 'red'), ('Direccional', 'green')]:
    ax2.scatter([], [], c=color, s=100, label=tipo)
ax2.legend(title='Tipo de Pozo', loc='upper left')

# 3. Distribución de tipos de pozos
tipos_count = pozos_info['tipo'].value_counts()
colors = ['#66b3ff', '#ff9999', '#99ff99']
wedges, texts, autotexts = ax3.pie(tipos_count.values, 
                                   labels=tipos_count.index,
                                   colors=colors,
                                   autopct='%1.0f%%',
                                   startangle=90,
                                   explode=[0.1 if t == 'Vertical' else 0 for t in tipos_count.index])

ax3.set_title('Distribución de Tipos de Pozos', fontsize=14, fontweight='bold')

# Mejorar texto
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)

plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

# PARTE 2: ANÁLISIS DE CORRELACIONES
print("PARTE 2: Matriz de correlaciones entre variables")
print("-" * 50)

# Cargar datos de producción
df = pd.read_csv(ruta_produccion)
df['fecha'] = pd.to_datetime(df['fecha'])

# Crear matriz de correlación
variables = ['barriles_diarios', 'presion_psi', 'temperatura_f', 'api_gravity']
corr_matrix = df[variables].corr()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Heatmap de correlaciones
im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_xticks(range(len(variables)))
ax1.set_yticks(range(len(variables)))
ax1.set_xticklabels(variables, rotation=45)
ax1.set_yticklabels(variables)
ax1.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')

# Añadir valores
for i in range(len(variables)):
    for j in range(len(variables)):
        text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                       ha="center", va="center", 
                       color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                       fontweight='bold')

plt.colorbar(im, ax=ax1)

# Identificar correlaciones más fuertes (excluyendo diagonal)
mask = np.ones_like(corr_matrix, dtype=bool)
np.fill_diagonal(mask, 0)
corr_values = corr_matrix.where(mask)
max_corr = corr_values.abs().max().max()
max_corr_idx = np.where(corr_values.abs() == max_corr)

print(f"Correlación más fuerte: {variables[max_corr_idx[0][0]]} vs {variables[max_corr_idx[1][0]]} = {corr_values.iloc[max_corr_idx[0][0], max_corr_idx[1][0]]:.3f}")

# Scatter plots de correlaciones significativas
# 1. Presión vs Producción
scatter1 = ax2.scatter(df['presion_psi'], df['barriles_diarios'], 
                      alpha=0.5, s=30, c=df['temperatura_f'], cmap='coolwarm')
ax2.set_xlabel('Presión (psi)')
ax2.set_ylabel('Producción (bpd)')
ax2.set_title('Presión vs Producción', fontsize=12)
z1 = np.polyfit(df['presion_psi'], df['barriles_diarios'], 1)
p1 = np.poly1d(z1)
ax2.plot(df['presion_psi'], p1(df['presion_psi']), "r--", linewidth=2)
plt.colorbar(scatter1, ax=ax2, label='Temperatura (°F)')

# 2. Temperatura vs API Gravity
ax3.scatter(df['temperatura_f'], df['api_gravity'], 
           alpha=0.5, s=30, c=df['barriles_diarios'], cmap='viridis')
ax3.set_xlabel('Temperatura (°F)')
ax3.set_ylabel('API Gravity')
ax3.set_title('Temperatura vs API Gravity', fontsize=12)
z2 = np.polyfit(df['temperatura_f'], df['api_gravity'], 1)
p2 = np.poly1d(z2)
ax3.plot(df['temperatura_f'], p2(df['temperatura_f']), "g--", linewidth=2)

# 3. Distribución de correlaciones
corr_flat = corr_values.values.flatten()
corr_flat = corr_flat[~np.isnan(corr_flat)]
ax4.hist(corr_flat, bins=20, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Valor de Correlación')
ax4.set_ylabel('Frecuencia')
ax4.set_title('Distribución de Correlaciones', fontsize=12)

plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

# PARTE 3: ANÁLISIS DE RENDIMIENTO POR CAMPO
print("PARTE 3: Comparación de rendimiento entre campos")
print("-" * 50)

# Cargar datos mensuales
df_mensual = pd.read_csv(ruta_mensual)
df_mensual['mes'] = pd.to_datetime(df_mensual['mes'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Gráfico de líneas con eficiencia
campos = df_mensual['campo'].unique()
colores = {'CAMPO-A': '#1f77b4', 'CAMPO-B': '#ff7f0e', 'CAMPO-C': '#2ca02c'}

for campo in campos:
    datos_campo = df_mensual[df_mensual['campo'] == campo]
    ax1.plot(datos_campo['mes'], datos_campo['eficiencia_operativa'], 
             marker='o', linewidth=2.5, markersize=8,
             color=colores[campo], label=campo)
    
    # Calcular y mostrar tendencia
    x_numeric = np.arange(len(datos_campo))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, datos_campo['eficiencia_operativa'])
    tendencia = slope * x_numeric + intercept
    ax1.plot(datos_campo['mes'], tendencia, '--', 
             color=colores[campo], alpha=0.5, linewidth=2)

# Área de eficiencia objetivo
ax1.axhspan(90, 95, alpha=0.2, color='green', label='Rango objetivo')
ax1.set_ylabel('Eficiencia Operativa (%)', fontsize=12)
ax1.set_title('Evolución de Eficiencia por Campo', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(85, 97)

# Añadir anotaciones de mejora
mejoras = {}
for campo in campos:
    datos_campo = df_mensual[df_mensual['campo'] == campo]
    mejora = datos_campo['eficiencia_operativa'].iloc[-1] - datos_campo['eficiencia_operativa'].iloc[0]
    mejoras[campo] = mejora
    
campo_mejor_mejora = max(mejoras, key=mejoras.get)
ax1.annotate(f'Mejor mejora: {campo_mejor_mejora}\n+{mejoras[campo_mejor_mejora]:.1f}%',
             xy=(df_mensual['mes'].iloc[-2], 91),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

# Gráfico de barras comparativo
meses_unicos = df_mensual['mes'].unique()
x = np.arange(len(meses_unicos))
width = 0.25

for i, campo in enumerate(campos):
    datos_campo = df_mensual[df_mensual['campo'] == campo]
    ax2.bar(x + i*width, datos_campo['eficiencia_operativa'], 
            width, label=campo, color=colores[campo], alpha=0.8)

ax2.set_xlabel('Mes', fontsize=12)
ax2.set_ylabel('Eficiencia Operativa (%)', fontsize=12)
ax2.set_title('Comparación Mensual de Eficiencia', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels([m.strftime('%Y-%m') for m in meses_unicos], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n✓ Campo con mejor mejora en eficiencia: {campo_mejor_mejora} (+{mejoras[campo_mejor_mejora]:.1f}%)")

print("\n" + "="*50 + "\n")

# PARTE 4: ANÁLISIS DE VARIABILIDAD
print("PARTE 4: Análisis de variabilidad en la producción")
print("-" * 50)

# Calcular estadísticas por pozo
estadisticas_pozos = df.groupby('pozo')['barriles_diarios'].agg(['mean', 'std', 'min', 'max'])
estadisticas_pozos['cv'] = (estadisticas_pozos['std'] / estadisticas_pozos['mean']) * 100
estadisticas_pozos = estadisticas_pozos.sort_values('cv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Preparar datos para boxplot
datos_boxplot = [df[df['pozo'] == pozo]['barriles_diarios'].values 
                 for pozo in estadisticas_pozos.index]

# Boxplot
bp = ax1.boxplot(datos_boxplot, labels=estadisticas_pozos.index, patch_artist=True)
for patch, cv in zip(bp['boxes'], estadisticas_pozos['cv']):
    if cv < 2:
        patch.set_facecolor('green')
    elif cv < 3:
        patch.set_facecolor('yellow')
    else:
        patch.set_facecolor('red')
        
ax1.set_ylabel('Producción (bpd)', fontsize=12)
ax1.set_title('Distribución de Producción por Pozo', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=45)

# Añadir CV en el gráfico
for i, (pozo, cv) in enumerate(zip(estadisticas_pozos.index, estadisticas_pozos['cv'])):
    ax1.text(i+1, estadisticas_pozos.loc[pozo, 'max'] + 20, 
             f'CV: {cv:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Violin plot
parts = ax2.violinplot(datos_boxplot, positions=range(len(estadisticas_pozos)), 
                       showmeans=True, showmedians=True)

for pc, cv in zip(parts['bodies'], estadisticas_pozos['cv']):
    if cv < 2:
        pc.set_facecolor('green')
    elif cv < 3:
        pc.set_facecolor('yellow')
    else:
        pc.set_facecolor('red')
    pc.set_alpha(0.7)

ax2.set_xticks(range(len(estadisticas_pozos)))
ax2.set_xticklabels(estadisticas_pozos.index, rotation=45)
ax2.set_ylabel('Producción (bpd)', fontsize=12)
ax2.set_title('Distribución de Densidad por Pozo', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Leyenda para colores
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='CV < 2% (Estable)'),
                  Patch(facecolor='yellow', label='CV 2-3% (Moderado)'),
                  Patch(facecolor='red', label='CV > 3% (Volátil)')]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

print("\n✓ Análisis de variabilidad:")
print(f"- Pozo más estable: {estadisticas_pozos.index[0]} (CV: {estadisticas_pozos['cv'].iloc[0]:.2f}%)")
print(f"- Pozo más volátil: {estadisticas_pozos.index[-1]} (CV: {estadisticas_pozos['cv'].iloc[-1]:.2f}%)")

print("\n" + "="*50 + "\n")

# PARTE 5: ANÁLISIS MULTIDIMENSIONAL
print("PARTE 5: Visualización multidimensional")
print("-" * 50)

# Preparar datos agregados por pozo
datos_agregados = df.groupby('pozo').agg({
    'presion_psi': 'mean',
    'temperatura_f': 'mean',
    'barriles_diarios': 'mean',
    'api_gravity': 'mean'
}).reset_index()

# Añadir información de tipo de pozo
datos_agregados = datos_agregados.merge(pozos_info[['id', 'tipo']], 
                                       left_on='pozo', right_on='id', how='left')

fig, ax = plt.subplots(figsize=(12, 8))

# Definir marcadores por tipo
marcadores = {'Vertical': 'o', 'Horizontal': 's', 'Direccional': '^'}
tipos = datos_agregados['tipo'].fillna('Vertical')

# Crear scatter plot multidimensional
for tipo in marcadores:
    mask = tipos == tipo
    scatter = ax.scatter(datos_agregados.loc[mask, 'presion_psi'],
                        datos_agregados.loc[mask, 'temperatura_f'],
                        s=datos_agregados.loc[mask, 'barriles_diarios']/3,
                        c=datos_agregados.loc[mask, 'api_gravity'],
                        marker=marcadores[tipo],
                        alpha=0.7,
                        cmap='plasma',
                        edgecolors='black',
                        linewidth=2,
                        label=tipo)

# Añadir líneas de regresión por tipo
for tipo, color in [('Vertical', 'blue'), ('Horizontal', 'red'), ('Direccional', 'green')]:
    mask = tipos == tipo
    if mask.sum() > 1:
        x = datos_agregados.loc[mask, 'presion_psi']
        y = datos_agregados.loc[mask, 'temperatura_f']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(x), p(np.sort(x)), '--', color=color, linewidth=2, alpha=0.7)

# Etiquetas de pozos
for idx, row in datos_agregados.iterrows():
    ax.annotate(row['pozo'].split('-')[-1], 
               (row['presion_psi'], row['temperatura_f']),
               xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('Presión Promedio (psi)', fontsize=12)
ax.set_ylabel('Temperatura Promedio (°F)', fontsize=12)
ax.set_title('Análisis Multidimensional de Pozos', fontsize=16, fontweight='bold')
ax.legend(title='Tipo de Pozo')
ax.grid(True, alpha=0.3)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', 
                           norm=plt.Normalize(vmin=datos_agregados['api_gravity'].min(),
                                            vmax=datos_agregados['api_gravity'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('API Gravity', rotation=270, labelpad=20)

# Añadir leyenda de tamaño
sizes = [1000, 1300, 1600]
labels = ['1000 bpd', '1300 bpd', '1600 bpd']
for size, label in zip(sizes, labels):
    ax.scatter([], [], s=size/3, c='gray', alpha=0.5, label=label)
ax.legend(loc='upper left', title='Producción')

plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

# PARTE 6: DASHBOARD EJECUTIVO
print("PARTE 6: Dashboard ejecutivo integral")
print("-" * 50)

# Preparar datos para dashboard
ultima_fecha = df['fecha'].max()
datos_ultimo_dia = df[df['fecha'] == ultima_fecha]
produccion_total_actual = datos_ultimo_dia['barriles_diarios'].sum()

# Tendencia últimos 7 días
ultimos_7_dias = df[df['fecha'] > ultima_fecha - pd.Timedelta(days=7)]
tendencia_7d = ultimos_7_dias.groupby('fecha')['barriles_diarios'].sum()

# Crear dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 2, 2], width_ratios=[1, 1, 1])

# Panel 1: KPI Principal
ax1 = fig.add_subplot(gs[0, :])
ax1.text(0.5, 0.5, f'PRODUCCIÓN TOTAL ACTUAL\n{produccion_total_actual:,.0f} bpd', 
         ha='center', va='center', fontsize=24, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
ax1.axis('off')

# Panel 2: Tendencia 7 días
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(tendencia_7d.index, tendencia_7d.values, 'b-', linewidth=3, marker='o', markersize=8)
ax2.set_title('Tendencia Últimos 7 Días', fontsize=12)
ax2.set_ylabel('Producción Total (bpd)')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Panel 3: Ranking de eficiencia
ax3 = fig.add_subplot(gs[1, 1])
eficiencia_pozos = df.groupby('pozo').apply(
    lambda x: (x['barriles_diarios'].mean() / x['barriles_diarios'].max()) * 100
).sort_values(ascending=True)

y_pos = np.arange(len(eficiencia_pozos))
bars = ax3.barh(y_pos, eficiencia_pozos.values)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(eficiencia_pozos.index)
ax3.set_xlabel('Eficiencia (%)')
ax3.set_title('Ranking de Eficiencia', fontsize=12)
ax3.axvline(x=90, color='red', linestyle='--', alpha=0.5)

# Colorear barras
for bar, eff in zip(bars, eficiencia_pozos.values):
    bar.set_color('green' if eff >= 90 else 'yellow' if eff >= 85 else 'red')

# Panel 4: Heatmap de producción
ax4 = fig.add_subplot(gs[1, 2])
pivot_prod = df.pivot_table(values='barriles_diarios', index='pozo', columns='fecha')
im = ax4.imshow(pivot_prod.values, cmap='YlOrRd', aspect='auto')
ax4.set_yticks(range(len(pivot_prod.index)))
ax4.set_yticklabels(pivot_prod.index)
ax4.set_xticks(range(0, len(pivot_prod.columns), 3))
ax4.set_xticklabels([d.strftime('%d/%m') for d in pivot_prod.columns[::3]], rotation=45)
ax4.set_title('Mapa de Calor - Producción', fontsize=12)
plt.colorbar(im, ax=ax4)

# Panel 5: Alertas
ax5 = fig.add_subplot(gs[2, 0])
umbral_produccion = 1000
pozos_bajo_umbral = datos_ultimo_dia[datos_ultimo_dia['barriles_diarios'] < umbral_produccion]['pozo'].tolist()

alertas_texto = "ALERTAS ACTIVAS:\n\n"
if pozos_bajo_umbral:
    alertas_texto += f"⚠️ Pozos bajo {umbral_produccion} bpd:\n"
    for pozo in pozos_bajo_umbral:
        prod = datos_ultimo_dia[datos_ultimo_dia['pozo'] == pozo]['barriles_diarios'].values[0]
        alertas_texto += f"  • {pozo}: {prod:.0f} bpd\n"
else:
    alertas_texto += "✅ Todos los pozos operando normalmente"

ax5.text(0.05, 0.95, alertas_texto, transform=ax5.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
ax5.axis('off')

# Panel 6: Proyección 30 días
ax6 = fig.add_subplot(gs[2, 1:])
# Calcular tendencia histórica
tendencia_historica = df.groupby('fecha')['barriles_diarios'].sum()
dias_futuros = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=30)

# Proyección simple basada en tendencia
slope, intercept, _, _, _ = stats.linregress(range(len(tendencia_historica)), tendencia_historica.values)
proyeccion = [tendencia_historica.iloc[-1] + slope * i for i in range(1, 31)]

ax6.plot(tendencia_historica.index, tendencia_historica.values, 'b-', linewidth=2, label='Histórico')
ax6.plot(dias_futuros, proyeccion, 'r--', linewidth=2, label='Proyección 30 días')
ax6.axvline(x=ultima_fecha, color='gray', linestyle=':', alpha=0.5)
ax6.set_title('Proyección de Producción a 30 Días', fontsize=12)
ax6.set_ylabel('Producción Total (bpd)')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

# DESAFÍO: STORYTELLING CON DATOS
print("DESAFÍO: Narrativa visual de datos")
print("-" * 50)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Análisis de Caída en Producción - Historia Completa', fontsize=18, fontweight='bold')

# 1. El problema: Identificar caída
produccion_total_diaria = df.groupby('fecha')['barriles_diarios'].sum()
ax1.plot(produccion_total_diaria.index, produccion_total_diaria.values, 'b-', linewidth=3)
ax1.axhline(y=produccion_total_diaria.mean(), color='green', linestyle='--', label='Promedio histórico')
ax1.fill_between(produccion_total_diaria.index, 
                 produccion_total_diaria.values,
                 produccion_total_diaria.mean(),
                 where=(produccion_total_diaria.values < produccion_total_diaria.mean()),
                 alpha=0.3, color='red', label='Bajo promedio')
ax1.set_title('1. EL PROBLEMA: Caída en Producción Total', fontsize=14)
ax1.set_ylabel('Producción Total (bpd)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. El análisis: Correlación con presión
ax2.scatter(df['presion_psi'], df['barriles_diarios'], 
           c=pd.to_numeric(df['fecha']), cmap='coolwarm', alpha=0.5)
z = np.polyfit(df['presion_psi'], df['barriles_diarios'], 1)
p = np.poly1d(z)
ax2.plot(df['presion_psi'], p(df['presion_psi']), "r--", linewidth=2)
ax2.set_title('2. EL ANÁLISIS: Correlación Presión-Producción', fontsize=14)
ax2.set_xlabel('Presión (psi)')
ax2.set_ylabel('Producción (bpd)')
ax2.text(0.05, 0.95, f'Correlación: {np.corrcoef(df["presion_psi"], df["barriles_diarios"])[0,1]:.3f}',
         transform=ax2.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

# 3. La causa raíz: Pozos específicos con problemas
produccion_por_pozo = df.groupby(['fecha', 'pozo'])['barriles_diarios'].mean().reset_index()
pivot_pozos = produccion_por_pozo.pivot(index='fecha', columns='pozo', values='barriles_diarios')
cambio_porcentual = ((pivot_pozos.iloc[-1] - pivot_pozos.iloc[0]) / pivot_pozos.iloc[0]) * 100

bars = ax3.bar(cambio_porcentual.index, cambio_porcentual.values,
               color=['red' if x < -5 else 'yellow' if x < 0 else 'green' for x in cambio_porcentual.values])
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_title('3. CAUSA RAÍZ: Pozos con Mayor Caída', fontsize=14)
ax3.set_ylabel('Cambio en Producción (%)')
ax3.tick_params(axis='x', rotation=45)

# Identificar pozos problemáticos
pozos_problema = cambio_porcentual[cambio_porcentual < -5].index.tolist()
ax3.text(0.05, 0.95, f'Pozos críticos: {", ".join(pozos_problema)}',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))

# 4. La solución: Proyección con optimización
dias_proyeccion = 15
fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=dias_proyeccion)

# Escenario actual
produccion_actual = produccion_total_diaria.iloc[-7:].mean()
proyeccion_actual = [produccion_actual] * dias_proyeccion

# Escenario optimizado (recuperar pozos problemáticos)
mejora_esperada = len(pozos_problema) * 50  # 50 bpd por pozo optimizado
proyeccion_optimizada = [produccion_actual + mejora_esperada * (i/dias_proyeccion) 
                        for i in range(dias_proyeccion)]

ax4.plot(produccion_total_diaria.index[-30:], produccion_total_diaria.values[-30:], 
         'b-', linewidth=2, label='Histórico')
ax4.plot(fechas_futuras, proyeccion_actual, 'r--', linewidth=2, label='Sin intervención')
ax4.plot(fechas_futuras, proyeccion_optimizada, 'g--', linewidth=3, label='Con optimización')
ax4.axvline(x=ultima_fecha, color='gray', linestyle=':', alpha=0.5)
ax4.fill_between(fechas_futuras, proyeccion_actual, proyeccion_optimizada,
                 alpha=0.3, color='green', label='Mejora potencial')
ax4.set_title('4. LA SOLUCIÓN: Optimización de Pozos Críticos', fontsize=14)
ax4.set_ylabel('Producción Total (bpd)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Añadir ROI estimado
roi_text = f"ROI Estimado:\nInversión: $150,000\nRetorno mensual: ${mejora_esperada * 30 * 50:,.0f}\nPayback: 3.5 meses"
ax4.text(0.98, 0.02, roi_text, transform=ax4.transAxes, 
         fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n✅ Laboratorio completado exitosamente")
print("\nINSIGHTS CLAVE IDENTIFICADOS:")
print("1. Correlación significativa entre presión y producción")
print("2. Pozos con alta variabilidad requieren atención")
print(f"3. Campo con mejor tendencia de eficiencia: {campo_mejor_mejora}")
print(f"4. Pozos críticos identificados: {', '.join(pozos_problema)}")
print("5. Potencial de mejora: +{:.0f} bpd con optimización".format(mejora_esperada))