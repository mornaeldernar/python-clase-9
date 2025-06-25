"""
SESIÓN 9: VISUALIZACIÓN BÁSICA DE DATOS DE POZOS
Demo 1: Introducción a Matplotlib - Gráficos Básicos

Este demo cubre los conceptos fundamentales de Matplotlib aplicados
a la visualización de datos de producción petrolera.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

print("=== DEMO 1: MATPLOTLIB BÁSICO ===")
print()

# 1. GRÁFICO DE LÍNEAS - Producción temporal
print("1. Gráfico de Líneas - Serie Temporal de Producción")
print("-" * 50)

# Datos simulados de producción diaria
dias = 30
fechas = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(dias)]
produccion = 1200 + np.random.normal(0, 50, dias).cumsum()

# Crear figura y eje
plt.figure(figsize=(10, 6))
plt.plot(fechas, produccion, linewidth=2, color='darkblue')
plt.title('Producción Diaria - POZO-A-001', fontsize=16, fontweight='bold')
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Barriles por Día (bpd)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar con el siguiente gráfico...")

# 2. GRÁFICO DE BARRAS - Comparación entre pozos
print("\n2. Gráfico de Barras - Comparación de Producción por Pozo")
print("-" * 50)

pozos = ['POZO-A-001', 'POZO-A-002', 'POZO-B-001', 'POZO-B-002', 'POZO-C-001']
produccion_promedio = [1250, 980, 1580, 850, 1100]
colores = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(10, 6))
barras = plt.bar(pozos, produccion_promedio, color=colores, edgecolor='black', linewidth=1.5)

# Añadir valores encima de las barras
for barra, valor in zip(barras, produccion_promedio):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 20,
             f'{valor:,}', ha='center', va='bottom', fontweight='bold')

plt.title('Producción Promedio por Pozo - Enero 2024', fontsize=16, fontweight='bold')
plt.xlabel('Pozo', fontsize=12)
plt.ylabel('Producción Promedio (bpd)', fontsize=12)
plt.ylim(0, max(produccion_promedio) * 1.15)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar con el siguiente gráfico...")

# 3. GRÁFICO DE DISPERSIÓN - Relación Presión vs Producción
print("\n3. Gráfico de Dispersión - Presión vs Producción")
print("-" * 50)

# Generar datos correlacionados
np.random.seed(42)
presion = np.random.uniform(2500, 3200, 50)
produccion_scatter = 0.4 * presion + np.random.normal(0, 100, 50)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(presion, produccion_scatter, 
                     c=produccion_scatter, 
                     cmap='viridis', 
                     s=100, 
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=1)

# Añadir línea de tendencia
z = np.polyfit(presion, produccion_scatter, 1)
p = np.poly1d(z)
plt.plot(presion, p(presion), "r--", linewidth=2, label=f'Tendencia: y = {z[0]:.2f}x + {z[1]:.2f}')

plt.colorbar(scatter, label='Producción (bpd)')
plt.title('Relación entre Presión y Producción', fontsize=16, fontweight='bold')
plt.xlabel('Presión (psi)', fontsize=12)
plt.ylabel('Producción (bpd)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

input("\nPresiona Enter para continuar con el siguiente gráfico...")

# 4. SUBPLOTS - Múltiples visualizaciones
print("\n4. Subplots - Dashboard de Métricas del Pozo")
print("-" * 50)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dashboard de Monitoreo - POZO-B-001', fontsize=16, fontweight='bold')

# Subplot 1: Producción temporal
dias_corto = 7
fechas_corto = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(dias_corto)]
produccion_corto = [1580, 1575, 1570, 1565, 1560, 1555, 1550]
ax1.plot(fechas_corto, produccion_corto, 'o-', linewidth=2, markersize=8)
ax1.set_title('Producción Última Semana')
ax1.set_ylabel('bpd')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Subplot 2: Temperatura
temperatura = [190, 189, 189, 188, 188, 187, 187]
ax2.plot(fechas_corto, temperatura, 's-', color='red', linewidth=2, markersize=8)
ax2.set_title('Temperatura del Pozo')
ax2.set_ylabel('°F')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Subplot 3: Distribución de API Gravity
api_gravity = np.random.normal(33.2, 0.3, 100)
ax3.hist(api_gravity, bins=20, color='green', alpha=0.7, edgecolor='black')
ax3.set_title('Distribución API Gravity')
ax3.set_xlabel('API Gravity')
ax3.set_ylabel('Frecuencia')
ax3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Comparación de costos
categorias = ['Operación', 'Mantenimiento', 'Químicos', 'Personal']
costos = [3200, 1800, 900, 1500]
ax4.pie(costos, labels=categorias, autopct='%1.1f%%', startangle=90)
ax4.set_title('Distribución de Costos Operativos')

plt.tight_layout()
plt.show()

# 5. PERSONALIZACIÓN AVANZADA
print("\n5. Personalización Avanzada - Gráfico Profesional")
print("-" * 50)

# Crear datos para múltiples pozos
pozos_multiples = ['POZO-A-001', 'POZO-B-001', 'POZO-C-001']
colores_pozos = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(12, 7))

for i, (pozo, color) in enumerate(zip(pozos_multiples, colores_pozos)):
    produccion_pozo = 1200 - i*100 + np.random.normal(0, 30, dias).cumsum()
    plt.plot(fechas, produccion_pozo, 
             linewidth=2.5, 
             color=color, 
             label=pozo,
             marker='o' if i == 0 else 's' if i == 1 else '^',
             markersize=6,
             markevery=5)

# Personalización completa
plt.title('Comparación de Producción - Enero 2024', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Fecha', fontsize=14, fontweight='bold')
plt.ylabel('Producción (bpd)', fontsize=14, fontweight='bold')

# Añadir anotación
plt.annotate('Mantenimiento programado', 
             xy=(fechas[15], 1100), 
             xytext=(fechas[10], 950),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

# Personalizar leyenda
plt.legend(loc='upper right', 
          frameon=True, 
          fancybox=True, 
          shadow=True,
          fontsize=12)

# Personalizar grid
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Personalizar ejes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Añadir línea horizontal de referencia
plt.axhline(y=1200, color='gray', linestyle=':', linewidth=2, alpha=0.7)
plt.text(fechas[-1], 1210, 'Objetivo: 1,200 bpd', ha='right', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n✅ Demo completado: Fundamentos de Matplotlib")
print("\nConceptos cubiertos:")
print("- Gráficos de líneas para series temporales")
print("- Gráficos de barras para comparaciones")
print("- Gráficos de dispersión para correlaciones")
print("- Subplots para dashboards")
print("- Personalización avanzada de gráficos")