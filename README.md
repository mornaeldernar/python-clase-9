# Sesión 9: Visualización Básica de Datos de Pozos

## 📋 Información General

- **Duración**: 2 horas
- **Nivel**: Intermedio
- **Prerequisitos**: Sesiones 1-8 (Python básico, Pandas)
- **Objetivo**: Dominar las técnicas de visualización de datos para comunicar efectivamente hallazgos analíticos en el contexto petrolero

## 🎯 Objetivos de Aprendizaje

Al finalizar esta sesión, los participantes serán capaces de:

1. **Crear visualizaciones básicas con Matplotlib**
   - Gráficos de líneas para series temporales
   - Gráficos de barras para comparaciones
   - Gráficos de dispersión para correlaciones
   - Subplots para dashboards integrados

2. **Integrar Pandas con Matplotlib**
   - Usar métodos de visualización integrados en DataFrames
   - Crear visualizaciones directamente desde datos procesados
   - Aplicar agrupaciones y agregaciones con visualización

3. **Personalizar visualizaciones profesionales**
   - Aplicar estilos y temas corporativos
   - Añadir anotaciones y elementos explicativos
   - Exportar gráficos en múltiples formatos

4. **Comunicar insights efectivamente**
   - Diseñar visualizaciones para audiencias ejecutivas
   - Crear narrativas visuales con datos
   - Generar dashboards informativos

## 📚 Contenido de la Sesión

### 1. Introducción a Matplotlib (30 min)
- Conceptos fundamentales de visualización
- Anatomía de un gráfico en Matplotlib
- Tipos básicos de gráficos y sus casos de uso
- Personalización básica de elementos

### 2. Integración con Pandas (30 min)
- Métodos `.plot()` en DataFrames y Series
- Visualización de datos agrupados
- Análisis temporal con visualizaciones
- Exportación y guardado de gráficos

### 3. Aplicaciones Prácticas (45 min)
- Visualización de series temporales de producción
- Comparación visual entre pozos y campos
- Análisis de correlaciones y tendencias
- Creación de dashboards operativos

### 4. Proyecto Final (15 min)
- Desarrollo de un dashboard ejecutivo completo
- Implementación de sistema de alertas visuales
- Generación automatizada de reportes

## 🛠️ Recursos y Materiales

### Estructura de Archivos
```
sesion-09/
├── README.md                    # Este archivo
├── datos/                       # Datasets de ejemplo
│   ├── produccion_historica.csv # Datos de producción diaria
│   ├── resumen_mensual.csv      # Métricas agregadas
│   └── comparacion_pozos.json   # Información de pozos
├── demos/                       # Demostraciones en vivo
│   ├── demo_01_matplotlib_basico.py
│   └── demo_02_pandas_matplotlib.py
├── ejercicios/                  # Laboratorios prácticos
│   ├── lab_01_series_temporales.py
│   ├── lab_02_visualizacion_comparativa.py
│   └── proyecto_dashboard_interactivo.py
├── soluciones/                  # Soluciones completas
│   ├── lab_01_series_temporales_solucion.py
│   └── lab_02_visualizacion_comparativa_solucion.py
└── docs/                        # Documentación adicional
    ├── guia_estudiante.md
    └── guia_instructor.md
```

### Datasets Disponibles

1. **produccion_historica.csv**
   - Datos diarios de 5 pozos durante enero 2024
   - Variables: fecha, pozo, barriles_diarios, presion_psi, temperatura_f, api_gravity
   - 75 registros totales

2. **resumen_mensual.csv**
   - Resumen mensual por campo (julio 2023 - enero 2024)
   - Variables: mes, campo, produccion_total, promedio_diario, eficiencia_operativa
   - 21 registros totales

3. **comparacion_pozos.json**
   - Información detallada de cada pozo
   - Incluye: profundidad, años de operación, tipo, producción acumulada, costos

## 💻 Configuración del Entorno

### Instalación de Dependencias
```bash
# Crear ambiente virtual (si no existe)
conda create -n petroleum-viz python=3.9
conda activate petroleum-viz

# Instalar paquetes necesarios
pip install pandas matplotlib numpy scipy
```

### Verificación del Entorno
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print(f"Matplotlib versión: {plt.matplotlib.__version__}")
print(f"Pandas versión: {pd.__version__}")
print(f"NumPy versión: {np.__version__}")
```

## 🚀 Inicio Rápido

### Ejemplo Básico
```python
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('datos/produccion_historica.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# Crear visualización simple
plt.figure(figsize=(10, 6))
for pozo in df['pozo'].unique():
    datos_pozo = df[df['pozo'] == pozo]
    plt.plot(datos_pozo['fecha'], datos_pozo['barriles_diarios'], label=pozo)

plt.title('Producción Diaria por Pozo')
plt.xlabel('Fecha')
plt.ylabel('Barriles por Día')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 📝 Ejercicios y Laboratorios

### Laboratorio 1: Series Temporales
- **Objetivo**: Crear visualizaciones efectivas de series temporales
- **Duración**: 30 minutos
- **Entregables**: 
  - Gráficos de producción temporal
  - Análisis de tendencias con media móvil
  - Dashboard básico de monitoreo

### Laboratorio 2: Visualización Comparativa
- **Objetivo**: Desarrollar visualizaciones para análisis comparativo
- **Duración**: 30 minutos
- **Entregables**:
  - Matriz de correlaciones visualizada
  - Análisis multidimensional
  - Dashboard ejecutivo

### Proyecto Final: Dashboard Interactivo
- **Objetivo**: Crear un sistema completo de visualización
- **Duración**: 45 minutos
- **Entregables**:
  - Dashboard con múltiples vistas
  - Sistema de alertas visuales
  - Reporte PDF automatizado

## 🎓 Valor para Meridian Consulting

Esta sesión proporciona habilidades críticas para:

1. **Comunicación Visual Efectiva**
   - Presentar hallazgos complejos de manera clara
   - Facilitar la toma de decisiones basada en datos
   - Mejorar la calidad de reportes y presentaciones

2. **Estandarización de Reportes**
   - Crear plantillas reutilizables de visualización
   - Automatizar generación de gráficos
   - Mantener consistencia visual corporativa

3. **Análisis Visual de Tendencias**
   - Identificar patrones y anomalías rápidamente
   - Comunicar tendencias operacionales
   - Proyectar escenarios futuros

4. **Dashboards para Directivos**
   - Diseñar interfaces informativas
   - Implementar KPIs visuales
   - Facilitar monitoreo en tiempo real

## 📊 Casos de Uso en la Industria

1. **Reportes Mensuales de Producción**
   - Visualización de tendencias de producción
   - Comparación entre pozos y campos
   - Identificación de pozos bajo rendimiento

2. **Análisis de Eficiencia Operativa**
   - Correlación entre variables operativas
   - Benchmarking visual entre activos
   - Identificación de oportunidades de mejora

3. **Presentaciones Ejecutivas**
   - Dashboards de alto nivel
   - Storytelling con datos
   - Proyecciones y escenarios

## 🔍 Recursos Adicionales

### Documentación Oficial
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Pandas Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

### Mejores Prácticas
- [Effective Data Visualization](https://www.tableau.com/learn/articles/best-practices-for-effective-dashboards)
- [Color Theory for Data Viz](https://blog.datawrapper.de/colorguide/)
- [Chart Selection Guide](https://extremepresentation.typepad.com/files/choosing-a-good-chart-09.pdf)

### Herramientas Complementarias
- **Plotly**: Para gráficos interactivos
- **Seaborn**: Para visualizaciones estadísticas
- **Dash**: Para aplicaciones web de visualización

## 👥 Soporte y Ayuda

### Durante la Sesión
- Levanta la mano virtual para preguntas
- Usa el chat para consultas rápidas
- Comparte tu pantalla si necesitas ayuda específica

### Después de la Sesión
- Revisa las soluciones proporcionadas
- Practica con los datasets adicionales
- Consulta la documentación de referencia

## ✅ Checklist de Aprendizaje

Al finalizar esta sesión, deberías ser capaz de:

- [ ] Crear gráficos básicos con Matplotlib
- [ ] Personalizar elementos visuales (títulos, ejes, colores)
- [ ] Integrar visualizaciones con análisis de Pandas
- [ ] Crear subplots y dashboards complejos
- [ ] Exportar gráficos en alta calidad
- [ ] Diseñar visualizaciones para audiencias ejecutivas
- [ ] Implementar mejores prácticas de visualización
- [ ] Crear narrativas visuales con datos

## 🎯 Siguiente Sesión

**Sesión 10: Introducción al Machine Learning para Predicción de Producción**
- Conceptos básicos de ML
- Preparación de datos para modelos
- Implementación de modelos predictivos
- Evaluación y optimización

---

*Material desarrollado para el programa de capacitación en Python para la industria petrolera - Meridian Consulting*