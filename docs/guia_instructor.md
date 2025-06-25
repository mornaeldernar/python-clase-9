# Guía del Instructor - Sesión 9: Visualización de Datos

## 🎯 Objetivos de Enseñanza

### Objetivo Principal
Capacitar a los participantes en la creación de visualizaciones profesionales de datos petroleros utilizando Matplotlib y Pandas, enfocándose en la comunicación efectiva de insights para la toma de decisiones.

### Objetivos Específicos
1. Dominar los fundamentos de Matplotlib
2. Integrar visualizaciones con análisis de Pandas
3. Aplicar principios de diseño visual
4. Crear dashboards informativos para diferentes audiencias

## ⏱️ Estructura de la Sesión (2 horas)

### Distribución del Tiempo
```
00:00 - 00:10 | Introducción y objetivos
00:10 - 00:40 | Demo 1: Fundamentos de Matplotlib
00:40 - 01:10 | Demo 2: Integración Pandas + Matplotlib
01:10 - 01:20 | Break
01:20 - 01:50 | Laboratorios prácticos
01:50 - 02:00 | Cierre y Q&A
```

## 📋 Preparación Pre-Sesión

### Checklist del Instructor
- [ ] Verificar funcionamiento de demos
- [ ] Preparar datasets de ejemplo
- [ ] Revisar casos de uso relevantes
- [ ] Configurar ambiente de desarrollo
- [ ] Preparar ejemplos de malas prácticas

### Configuración Técnica
```python
# Script de verificación
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

print("Verificando ambiente...")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Archivos de datos: {os.listdir('../datos')}")
```

## 🎓 Estrategias de Enseñanza

### 1. Introducción (10 min)

#### Gancho Inicial
"¿Cuántas veces han tenido datos excelentes pero no han logrado comunicar su importancia? Hoy aprenderemos a contar historias con datos."

#### Actividad Rompehielo
Mostrar dos visualizaciones del mismo dataset:
- Una mal diseñada (3D, muchos colores, sin etiquetas)
- Una bien diseñada (clara, profesional, informativa)

Preguntar: "¿Cuál presentarían a un director?"

### 2. Demo 1: Fundamentos (30 min)

#### Estructura Didáctica
1. **Concepto** (5 min)
   - Explicar anatomía de un gráfico
   - Mostrar diagrama Figure/Axes

2. **Demostración** (15 min)
   - Construir gráfico paso a paso
   - Narrar cada decisión de diseño

3. **Práctica Guiada** (10 min)
   - Estudiantes replican con variaciones
   - Troubleshooting en vivo

#### Puntos Clave a Enfatizar
- "Cada elemento visual debe tener un propósito"
- "La claridad es más importante que la estética"
- "Piensen en su audiencia al diseñar"

### 3. Demo 2: Integración con Pandas (30 min)

#### Progresión Conceptual
1. Visualización directa desde DataFrames
2. Agrupación y visualización
3. Series temporales
4. Dashboards complejos

#### Ejemplos Contextualizados
```python
# Caso real: Análisis de caída de producción
df = pd.read_csv('produccion_historica.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# Mostrar problema
print("Gerente: ¿Por qué cayó la producción?")

# Visualizar para investigar
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Producción total
produccion_total = df.groupby('fecha')['barriles_diarios'].sum()
ax1.plot(produccion_total)
ax1.set_title('Producción Total - ¿Dónde está el problema?')

# Desglose por pozo
for pozo in df['pozo'].unique():
    datos = df[df['pozo'] == pozo]
    ax2.plot(datos['fecha'], datos['barriles_diarios'], label=pozo)
ax2.set_title('Por Pozo - ¡Encontramos el culpable!')
ax2.legend()
```

### 4. Laboratorios Prácticos (30 min)

#### Metodología de Facilitación
1. **Introducir el problema** (2 min)
   - Contexto empresarial real
   - Objetivos claros

2. **Trabajo individual** (20 min)
   - Circular y asistir
   - Identificar errores comunes

3. **Revisión grupal** (8 min)
   - Mostrar 2-3 soluciones
   - Discutir enfoques alternativos

#### Errores Comunes y Soluciones

| Error Común | Causa | Solución |
|------------|-------|----------|
| Gráficos ilegibles | Tamaño pequeño | `figsize=(12, 8)` |
| Fechas superpuestas | Rotación | `plt.xticks(rotation=45)` |
| Leyenda cortada | Posición | `bbox_to_anchor` |
| Colores confusos | Mal contraste | Paletas predefinidas |

## 💡 Casos de Uso Reales

### Caso 1: Reporte Mensual de Producción
**Contexto**: El gerente necesita ver tendencias y anomalías rápidamente

**Solución Visual**:
```python
# Dashboard de una página
fig = plt.figure(figsize=(11, 8.5))  # Tamaño carta
gs = fig.add_gridspec(3, 2)

# KPI principal arriba
ax_kpi = fig.add_subplot(gs[0, :])
# Tendencia en el medio
ax_trend = fig.add_subplot(gs[1, :])
# Comparaciones abajo
ax_comp1 = fig.add_subplot(gs[2, 0])
ax_comp2 = fig.add_subplot(gs[2, 1])
```

### Caso 2: Presentación a Inversionistas
**Contexto**: Mostrar potencial de optimización

**Elementos Clave**:
- Proyecciones con intervalos de confianza
- Escenarios comparativos
- ROI visualizado
- Estilo corporativo impecable

### Caso 3: Análisis Técnico Detallado
**Contexto**: Ingenieros investigando correlaciones

**Herramientas Visuales**:
- Matrices de correlación
- Scatter plots con regresiones
- Análisis multivariable
- Distribuciones estadísticas

## 🛠️ Recursos Durante la Sesión

### Scripts de Soporte
```python
# Función auxiliar para estilo corporativo
def aplicar_estilo_meridian(ax):
    """Aplica estilo visual corporativo"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    
# Paleta de colores corporativa
COLORES_MERIDIAN = {
    'principal': '#003f5c',
    'secundario': '#ff6361',
    'exito': '#58b368',
    'alerta': '#ffa600',
    'neutro': '#8c8c8c'
}
```

### Snippets para Compartir
```python
# 1. Guardar figuras profesionalmente
plt.savefig('reporte.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# 2. Formato de números en ejes
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# 3. Anotaciones profesionales
ax.annotate('Punto crítico', xy=(x, y), 
            xytext=(x+10, y+10),
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
```

## 📊 Evaluación y Feedback

### Rúbrica de Evaluación

| Criterio | Excelente (4) | Bueno (3) | Regular (2) | Necesita Mejora (1) |
|----------|---------------|-----------|-------------|-------------------|
| **Técnica** | Domina matplotlib y pandas | Usa funciones correctamente | Funcionalidad básica | Errores frecuentes |
| **Diseño** | Profesional y claro | Buena presentación | Legible | Confuso |
| **Insights** | Revela patrones ocultos | Identifica tendencias | Muestra datos | Solo gráficos |
| **Código** | Limpio y reutilizable | Organizado | Funcional | Desorganizado |

### Preguntas de Reflexión
1. ¿Qué tipo de gráfico elegirías para mostrar correlaciones?
2. ¿Cómo mejorarías este gráfico para una audiencia ejecutiva?
3. ¿Qué historia cuentan estos datos?

## 🎯 Actividades de Refuerzo

### Mini-Desafíos (5 min c/u)
1. **Desafío de Velocidad**: Crear gráfico de barras en < 2 min
2. **Desafío de Diseño**: Mejorar gráfico feo
3. **Desafío de Insights**: Encontrar anomalía en datos

### Ejercicio de Cierre
"Storytelling con Datos":
1. Dividir en grupos de 3
2. Cada grupo recibe mismo dataset
3. 10 min para crear visualización
4. 2 min para presentar su "historia"
5. Votar por la más convincente

## 🔧 Troubleshooting Common Issues

### Problemas Técnicos Frecuentes

```python
# Problema: "Figure size too small"
plt.figure(figsize=(12, 8))  # Aumentar tamaño

# Problema: "Latex error"
plt.rc('text', usetex=False)  # Desactivar LaTeX

# Problema: "Memory error con datasets grandes"
# Solución: Muestrear datos
df_sample = df.sample(n=1000) if len(df) > 10000 else df

# Problema: "Backend error"
import matplotlib
matplotlib.use('Agg')  # Cambiar backend
```

### Preguntas Frecuentes

**P: ¿Cuándo usar Matplotlib vs otras librerías?**
R: Matplotlib para control total, Seaborn para estadísticas, Plotly para interactividad.

**P: ¿Cómo elijo colores apropiados?**
R: Usa paletas probadas, considera daltonismo, mantén consistencia.

**P: ¿Cuál es el tamaño ideal de figura?**
R: Depende del medio: Pantalla (10x6), Impresión (8x5), Presentación (16x9).

## 📚 Material Complementario

### Para Compartir Post-Sesión
1. Cheat sheet de Matplotlib
2. Plantillas de dashboards
3. Guía de mejores prácticas
4. Enlaces a galerías de ejemplos

### Lecturas Recomendadas
- "Storytelling with Data" - Cole Nussbaumer Knaflic
- "The Visual Display of Quantitative Information" - Edward Tufte
- "Fundamentals of Data Visualization" - Claus Wilke

## ✅ Checklist Post-Sesión

- [ ] Compartir código de demos
- [ ] Enviar soluciones de laboratorios
- [ ] Recopilar feedback
- [ ] Actualizar materiales según comentarios
- [ ] Preparar casos adicionales para siguientes grupos

## 💬 Mensajes Clave para Reforzar

1. **"Los datos sin visualización son solo números"**
2. **"El mejor gráfico es el que comunica claramente"**
3. **"Diseñen para su audiencia, no para ustedes"**
4. **"La práctica perfecciona la visualización"**
5. **"Cada pixel debe ganarse su lugar"**

---

**Recuerda: Tu entusiasmo por la visualización de datos es contagioso. ¡Haz que los participantes se emocionen por contar historias con sus datos!**