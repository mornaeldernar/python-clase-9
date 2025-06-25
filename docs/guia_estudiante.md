# Guía del Estudiante - Sesión 9: Visualización de Datos

## 🎯 Objetivos de la Sesión

En esta sesión aprenderás a crear visualizaciones profesionales de datos petroleros que te permitirán comunicar hallazgos de manera efectiva a diferentes audiencias, desde ingenieros hasta directivos.

## 📚 Preparación Previa

### Conocimientos Necesarios
- Python básico (variables, funciones, bucles)
- Pandas para manipulación de datos
- Conceptos básicos de estadística

### Configuración del Entorno
```python
# Verificar instalación
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("✅ Matplotlib:", plt.matplotlib.__version__)
print("✅ Pandas:", pd.__version__)
print("✅ NumPy:", np.__version__)
```

## 🗺️ Ruta de Aprendizaje

### 1. Fundamentos de Matplotlib (30 min)

#### Conceptos Clave
- **Figure y Axes**: Contenedores principales de los gráficos
- **Plot**: Función para crear líneas
- **Scatter**: Función para gráficos de dispersión
- **Bar**: Función para gráficos de barras

#### Ejemplo Práctico
```python
# Gráfico simple
import matplotlib.pyplot as plt

dias = [1, 2, 3, 4, 5]
produccion = [1200, 1250, 1180, 1300, 1280]

plt.figure(figsize=(8, 5))
plt.plot(dias, produccion, 'b-o', linewidth=2, markersize=8)
plt.title('Producción Semanal')
plt.xlabel('Día')
plt.ylabel('Barriles')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Integración con Pandas (30 min)

#### Ventajas de Pandas + Matplotlib
- Visualización directa desde DataFrames
- Manejo automático de índices y fechas
- Agrupación y agregación integrada

#### Ejemplo con Datos Reales
```python
# Cargar y visualizar datos
df = pd.read_csv('datos/produccion_historica.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# Gráfico por pozo
df.groupby('pozo')['barriles_diarios'].mean().plot(kind='bar')
plt.title('Producción Promedio por Pozo')
plt.ylabel('Barriles por Día')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3. Personalización Avanzada (30 min)

#### Elementos a Personalizar
- Colores y estilos
- Anotaciones y etiquetas
- Leyendas y títulos
- Escalas y formatos

#### Técnicas Profesionales
```python
# Gráfico profesional
fig, ax = plt.subplots(figsize=(10, 6))

# Personalizar estilo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Añadir anotaciones
ax.annotate('Máximo histórico', 
            xy=(fecha_max, valor_max),
            xytext=(fecha_max + 2, valor_max + 50),
            arrowprops=dict(arrowstyle='->', color='red'))
```

## 💡 Tips y Mejores Prácticas

### 1. Elección del Tipo de Gráfico

| Tipo de Dato | Gráfico Recomendado | Cuándo Usar |
|--------------|-------------------|-------------|
| Serie temporal | Líneas | Tendencias en el tiempo |
| Comparaciones | Barras | Valores entre categorías |
| Correlaciones | Dispersión | Relación entre variables |
| Proporciones | Pastel/Donut | Partes de un todo |
| Distribuciones | Histograma/Boxplot | Análisis estadístico |

### 2. Principios de Diseño

1. **Claridad sobre complejidad**
   - Menos es más
   - Elimina elementos innecesarios
   - Usa espacio en blanco

2. **Consistencia visual**
   - Misma paleta de colores
   - Fuentes uniformes
   - Escalas comparables

3. **Accesibilidad**
   - Colores distinguibles
   - Texto legible
   - Contraste adecuado

### 3. Errores Comunes a Evitar

❌ **NO HACER:**
- Gráficos 3D innecesarios
- Demasiados colores
- Ejes sin etiquetas
- Escalas engañosas

✅ **HACER:**
- Gráficos 2D claros
- Paleta limitada y significativa
- Etiquetas descriptivas
- Escalas honestas

## 🛠️ Herramientas y Atajos

### Comandos Útiles de Matplotlib
```python
# Configuración global
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Estilos predefinidos
plt.style.use('seaborn-v0_8-darkgrid')

# Guardar figuras
plt.savefig('grafico.png', dpi=300, bbox_inches='tight')

# Subplots múltiples
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
```

### Paletas de Colores Recomendadas
```python
# Colores corporativos
colores_meridian = {
    'azul': '#003f5c',
    'naranja': '#ff6361',
    'verde': '#58b368',
    'gris': '#8c8c8c'
}

# Paletas para mapas de calor
# 'RdYlGn' - Rojo a Verde (bueno para métricas)
# 'viridis' - Accesible y profesional
# 'coolwarm' - Azul a Rojo (temperaturas)
```

## 📊 Plantillas de Código

### 1. Dashboard Básico
```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dashboard de Producción', fontsize=16)

# Panel 1: Serie temporal
ax1.plot(fechas, valores)
ax1.set_title('Producción Diaria')

# Panel 2: Comparación
ax2.bar(pozos, promedios)
ax2.set_title('Promedio por Pozo')

# Panel 3: Correlación
ax3.scatter(x, y)
ax3.set_title('Presión vs Producción')

# Panel 4: Distribución
ax4.hist(datos, bins=20)
ax4.set_title('Distribución de API')

plt.tight_layout()
```

### 2. Gráfico para Reporte Ejecutivo
```python
# Configuración profesional
plt.figure(figsize=(12, 7))
plt.style.use('seaborn-v0_8-white')

# Gráfico principal
plt.plot(fechas, produccion, linewidth=3, color='#003f5c')

# Formato ejecutivo
plt.title('Tendencia de Producción - Q1 2024', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Producción (miles de barriles)', fontsize=14)

# Grid sutil
plt.grid(True, alpha=0.2, linestyle='--')

# Eliminar bordes superiores
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

## 🎯 Ejercicios de Práctica

### Nivel Básico
1. Crear un gráfico de líneas simple
2. Añadir título y etiquetas
3. Cambiar colores y estilos
4. Guardar en diferentes formatos

### Nivel Intermedio
1. Crear subplots múltiples
2. Añadir anotaciones y leyendas
3. Personalizar ejes y escalas
4. Integrar con datos de Pandas

### Nivel Avanzado
1. Desarrollar dashboard completo
2. Implementar interactividad básica
3. Crear funciones de visualización reutilizables
4. Generar reportes automatizados

## 🔧 Solución de Problemas Comunes

### Problema 1: Fechas ilegibles en eje X
```python
# Solución
plt.xticks(rotation=45)
# o
fig.autofmt_xdate()
```

### Problema 2: Leyenda fuera del gráfico
```python
# Solución
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
```

### Problema 3: Texto superpuesto
```python
# Solución
plt.tight_layout()
# o ajustar manualmente
plt.subplots_adjust(hspace=0.3, wspace=0.3)
```

## 📚 Recursos de Consulta

### Durante los Ejercicios
1. **Documentación rápida**: `help(plt.plot)`
2. **Galería de ejemplos**: matplotlib.org/gallery
3. **Cheat sheet**: Buscar "matplotlib cheat sheet pdf"

### Para Profundizar
- [Matplotlib Tutorial Oficial](https://matplotlib.org/stable/tutorials/index.html)
- [Effective Matplotlib](https://pbpython.com/effective-matplotlib.html)
- [Data Visualization Best Practices](https://www.tableau.com/learn/articles/best-practices-for-effective-dashboards)

## ✅ Autoevaluación

### Checkpoint 1: Fundamentos
- [ ] Puedo crear gráficos básicos (líneas, barras, dispersión)
- [ ] Sé añadir títulos, etiquetas y leyendas
- [ ] Puedo cambiar colores y estilos
- [ ] Entiendo la estructura Figure/Axes

### Checkpoint 2: Integración
- [ ] Puedo visualizar datos desde Pandas
- [ ] Sé crear gráficos agrupados
- [ ] Puedo manejar fechas en los ejes
- [ ] Entiendo cómo exportar gráficos

### Checkpoint 3: Profesional
- [ ] Puedo crear dashboards con subplots
- [ ] Sé personalizar completamente un gráfico
- [ ] Puedo crear visualizaciones para reportes
- [ ] Aplico mejores prácticas de diseño

## 🎉 Proyecto Final

### Objetivo
Crear un dashboard ejecutivo completo que incluya:
- Vista general de producción
- Análisis de tendencias
- Comparaciones entre activos
- Sistema de alertas visuales

### Entregables
1. Código Python documentado
2. Dashboard en formato PNG/PDF
3. Breve análisis de insights encontrados

### Criterios de Éxito
- Claridad y profesionalismo
- Correcta elección de visualizaciones
- Insights accionables identificados
- Código reutilizable

---

**¡Éxito en tu aprendizaje! Recuerda que la práctica hace al maestro en visualización de datos.**