

Dashboard Streamlit con una pestaña por IAFAS (SALUDPOL, ONCOSALUD, FOSPEME, FOSMAR (Marina de Guerra), FOSFAP) más un comparativo,
tabla dinámica y metodología. Dentro de cada IAFAS: resumen e ingreso, ámbito de atención, diagnósticos, medicamentos, insumos, procedimientos,
gasto y evolución, tendencias y proyecciones. Misma arquitectura que el Observatorio del Financiamiento Oncológico, sin módulo normativo.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Datos: `data/fact_iafas.parquet` (anonimizado, sin DNI). Para regenerar desde un nuevo Excel:
`python build_data.py RUTA/DATA_IAFAS.xlsx`. Ingreso de la IAFAS (peso del INEN): barra lateral o `data/primas.csv` (iafas, anio, prima_anual).

Archivos: `app.py` (navegación y filtros), `panels.py` (paneles), `ui.py` (estilos), `data_processing.py`, `stats_analysis.py`, `build_data.py`.

### Asistente «Pregunta a los datos» (barra lateral)

Debajo de «Panel de control» hay un cuadro para consultar los datos en lenguaje natural (ej.: «Top 10 medicamentos de SALUDPOL en 2025»,
«Evolución anual del gasto de ONCOSALUD»). Responde con un texto breve, un gráfico y una tabla descargable; por defecto respeta los filtros del panel.

- **Modo IA (Claude):** se activa con `ANTHROPIC_API_KEY` en `.streamlit/secrets.toml` (ver `secrets.toml.example`), en Secrets de Streamlit Cloud
  o ingresando la clave en la propia barra lateral (solo para la sesión). El modelo no ejecuta código: llama a la herramienta `consultar_datos`
  (agregaciones predefinidas en `asistente.py`) y solo recibe tablas resumidas, sin identificadores. Modelo configurable con `ANTHROPIC_MODEL`.
- **Modo básico (sin clave):** intérprete por reglas que reconoce IAFAS, años, tipo de consumo, métrica, agrupación y ranking.
