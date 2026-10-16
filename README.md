

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
