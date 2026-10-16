"""Consolida las liquidaciones (cancelados) remitidas por la Oficina de Contabilidad y Finanzas (Informe 000271-2026-OCF-OGA/INEN).
Uso: python build_liquidaciones.py <carpeta con los xlsx>  ->  data/liquidaciones.csv"""
import sys, glob, re
import pandas as pd, openpyxl

MAP = {"SALUDPOL": "SALUDPOL", "EJERCITO": "FOSPEME (Ejército)", "MARINA": "FOSMAR (Marina de Guerra)", "FUERZA_AEREA": "FOSFAP (Fuerza Aérea)"}
def iafas_de(fn):
    for k, v in MAP.items():
        if k in fn.upper(): return v

def main(folder):
    rows = []
    for f in sorted(glob.glob(f"{folder}/*.xlsx")):
        ia = iafas_de(f.split("/")[-1])
        if not ia or "TARIFARIO" in f.upper() or "ALTO_COSTO" in f.upper(): continue
        wb = openpyxl.load_workbook(f, data_only=True, read_only=True)
        for ws in wb:
            if ws.title.lower() == "consolidado": continue
            hdr = None
            for r in ws.iter_rows(values_only=True):
                if r and r[0] == "num_oficio": hdr = list(r); continue
                if hdr is None or not r or r[0] is None: continue
                d = dict(zip(hdr, r))
                if d.get("num_monto_total") is None: continue
                rows.append(dict(iafas=ia, hoja=ws.title, num_oficio=d["num_oficio"], serie=d["num_serie"], documento=d["des_num_documento"],
                                 cliente=d.get("des_cliente"), monto=d["num_monto_total"], fecha=d["fec_cancelacion"]))
    df = pd.DataFrame(rows)
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
    df["fecha"] = pd.to_datetime(df["fecha"].astype(str), dayfirst=True, errors="coerce", format="mixed")
    return df

if __name__ == "__main__":
    df = main(sys.argv[1]); df.to_csv("data/liquidaciones.csv", index=False); print(len(df), df.monto.sum())
