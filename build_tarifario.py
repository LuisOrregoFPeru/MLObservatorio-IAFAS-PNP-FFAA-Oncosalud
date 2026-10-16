"""
build_tarifario.py
------------------
Extrae del Tarifario INEN (Excel) el grupo de cada procedimiento. Los grupos son las filas de encabezado
(las que repiten «N° / CODIGO INEN / CPT SIS ...» con el nombre del servicio en la columna de descripción).

Uso:
    python build_tarifario.py RUTA/Tarifario_INEN.xlsx
Genera data/tarifario_inen.csv (cod, grupo_tarifario, desc_tarifa, tr, tpu). El código INEN es el COD_ITEM
de la extracción (a 6 dígitos).
"""
from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

OUT = Path(__file__).parent / "data" / "tarifario_inen.csv"


def main(path: str) -> None:
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    grp, rows = None, []
    for r in ws.iter_rows(min_row=6, values_only=True):
        b, c, desc = r[1], r[2], r[5]
        if b == "N°" and c and "CODIGO" in str(c).upper():
            if str(desc).upper().startswith("DESCRIPCI"):
                continue
            grp = " ".join(str(desc).split())
            continue
        if grp and c is not None and str(c).strip().isdigit():
            rows.append({"cod": str(c).strip().zfill(6), "grupo_tarifario": grp, "desc_tarifa": desc, "tr": r[6], "tpu": r[7]})
    t = pd.DataFrame(rows).drop_duplicates("cod")
    OUT.parent.mkdir(exist_ok=True)
    t.to_csv(OUT, index=False)
    print(len(t), "códigos en", t["grupo_tarifario"].nunique(), "grupos")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python build_tarifario.py RUTA/Tarifario_INEN.xlsx")
    main(sys.argv[1])
