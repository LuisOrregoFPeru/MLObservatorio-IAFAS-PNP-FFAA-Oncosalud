

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

### Novedades v21 (extracción V02)

- **Tipo de atención real.** La extracción V02 trae `TIPO_ATENCION` (ambulatorio u hospitalizado) en cada línea. Reemplaza la inferencia por paciente-mes que se usaba
  para medicamentos e insumos (coincidía con el dato real en 86.7% de su gasto). La pestaña pasa a llamarse «Tipo de atención»; para procedimientos se conserva el
  servicio del tarifario y se cruza con el tipo de atención. El filtro lateral «Ámbito de atención» se reemplazó por «Tipo de atención».
- **Estadío clínico** (`DX_ESTADIO`), agrupado en 0 a IV, con filtro lateral, dimensión en la tabla dinámica, el asistente y la tendencia por componente.
- **Desenlace vital** (`FEC_DECESO`, guardado solo como mes en `data/pacientes.parquet`). Nueva pestaña «Estadío y desenlace»: fallecidos, gasto en los últimos meses
  de vida, supervivencia de Kaplan-Meier por estadío o diagnóstico, quimioterapia en el mes del deceso y consumo facturado después del deceso. Filtro lateral «Condición vital».
- Para regenerar: `python build_data.py RUTA/DATA_PRESTACIONES_ASEGURADOS_2019_AGO2026_V02.xlsx`.

### Asistente «Pregunta a los datos» (barra lateral)

Debajo de «Panel de control» hay un cuadro para consultar los datos en lenguaje natural (ej.: «Top 10 medicamentos de SALUDPOL en 2025»,
«Evolución anual del gasto de ONCOSALUD»). Responde con un texto breve, un gráfico y una tabla descargable; por defecto respeta los filtros del panel.

- **Modo IA (Claude):** se activa con `ANTHROPIC_API_KEY` en `.streamlit/secrets.toml` (ver `secrets.toml.example`), en Secrets de Streamlit Cloud
  o ingresando la clave en la propia barra lateral (solo para la sesión). El modelo no ejecuta código: llama a la herramienta `consultar_datos`
  (agregaciones predefinidas en `asistente.py`) y solo recibe tablas resumidas, sin identificadores. Modelo configurable con `ANTHROPIC_MODEL`.
- **Modo básico (sin clave):** intérprete por reglas que reconoce IAFAS, años, tipo de consumo, métrica, agrupación y ranking.
- **Revisión de la pregunta:** al escribir (Ctrl+Enter o al salir del cuadro) un semáforo indica qué tan precisa es la pregunta según cinco elementos (métrica, IAFAS, periodo, foco y desglose). Los elementos que faltan aparecen como chips que añaden un fragmento editable; también se avisa de años fuera de la serie, años incompletos y preguntas causales o de proyección. Con modo IA, «Mejorar la pregunta con IA» propone una reformulación que se puede aceptar o ignorar. Los ejemplos se muestran como texto clicable.
