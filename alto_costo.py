"""alto_costo.py: marca los medicamentos de la lista nominal de alto costo (no PNUME) en el catálogo de ítems."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

LISTA = Path(__file__).parent / "data" / "alto_costo.csv"
COD = "En lista (mismo código)"
MOL = "Misma molécula, otra presentación"
_STOP = {"de", "del", "la", "cepa", "polvo", "para"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


def molecula(desc: str) -> str:
    """Texto previo a la primera cifra de la descripción, sin tildes y en minúsculas."""
    toks = []
    for t in _norm(desc).split(" "):
        if re.match(r"^[\d(\[]", t):
            break
        toks.append(t)
    return " ".join(toks).strip(" +,-")


def lista() -> pd.DataFrame:
    if not LISTA.exists():
        return pd.DataFrame(columns=["cod_siga", "cod_sismed", "descripcion", "precio_ref", "obs"])
    d = pd.read_csv(LISTA, dtype={"cod_siga": str, "cod_sismed": str}).fillna({"obs": ""})
    d["molecula"] = d["descripcion"].map(molecula)
    return d


def marcar(items: pd.DataFrame) -> pd.DataFrame:
    """Agrega alto_costo (''/COD/MOL) y molecula_norm a un catálogo con cod_item, tipo, descripcion."""
    L = lista()
    items = items.copy()
    items["alto_costo"] = ""
    if L.empty:
        return items
    med = items["tipo"] == "MEDICAMENTO"
    mol = items["descripcion"].map(molecula)
    items.loc[med & mol.isin(set(L["molecula"])), "alto_costo"] = MOL
    items.loc[med & items["cod_item"].isin(set(L["cod_siga"])), "alto_costo"] = COD
    items["molecula_norm"] = mol
    return items
