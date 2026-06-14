"""
downsample_dataset.py
─────────────────────
Reduz cada Location{N}.csv (~43k linhas) para ~1000 linhas estratificadas
sazonalmente: ~500 amostras de JJA (verão) + ~500 de DJF (inverno),
preservando a ordem cronológica dentro de cada janela. O resultado é
escrito em Location{N}_filtered.csv no mesmo diretório.

A subamostragem é determinística (passo `len // 500`), o que garante
representação distribuída ao longo dos 5 anos do dataset (2017–2021)
em ambas as estações. Mantém o split 80/20 cronológico de data.py
válido — basta usar os arquivos _filtered.csv como entrada.

Uso:
    python src/utils/downsample_dataset.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import pandas as pd

# Permite importar config.py do diretório src/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import DATA_DIR, NUM_CLIENTS, SUMMER_MONTHS, WINTER_MONTHS  # noqa: E402

TARGET_PER_SEASON = 500


def _stratified_sample(df: pd.DataFrame, months: list[int], target: int) -> pd.DataFrame:
    """Subamostragem temporal uniforme dentro da janela sazonal indicada."""
    window = df[df["Time"].dt.month.isin(months)]
    if len(window) == 0:
        return window
    step = max(1, len(window) // target)
    return window.iloc[::step].head(target)


def downsample_one(input_path: str, output_path: str) -> tuple[int, dict]:
    df = pd.read_csv(input_path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    summer = _stratified_sample(df, SUMMER_MONTHS, TARGET_PER_SEASON)
    winter = _stratified_sample(df, WINTER_MONTHS, TARGET_PER_SEASON)

    out = pd.concat([summer, winter]).sort_values("Time").reset_index(drop=True)

    if not (800 <= len(out) <= 1000):
        raise RuntimeError(f"Tamanho fora do esperado para {os.path.basename(input_path)}: " f"{len(out)} linhas (esperado 800-1000)")

    out.to_csv(output_path, index=False)
    months_count = dict(Counter(out["Time"].dt.month.tolist()))
    return len(out), months_count


def main() -> None:
    print(f"[INFO] Lendo de: {DATA_DIR}")
    print(f"[INFO] Alvo: {TARGET_PER_SEASON} JJA + {TARGET_PER_SEASON} DJF por arquivo\n")

    for loc_id in range(1, NUM_CLIENTS + 1):
        src = os.path.join(DATA_DIR, f"Location{loc_id}.csv")
        dst = os.path.join(DATA_DIR, f"Location{loc_id}_filtered.csv")

        if not os.path.exists(src):
            print(f"[WARN] {src} não encontrado, pulando.")
            continue

        n, months = downsample_one(src, dst)
        summer_n = sum(months.get(m, 0) for m in SUMMER_MONTHS)
        winter_n = sum(months.get(m, 0) for m in WINTER_MONTHS)
        print(f"[OK]  Location{loc_id}: {n} linhas (JJA={summer_n}, DJF={winter_n}) → {os.path.basename(dst)}")

    print("\n[DONE] Subamostragem concluída.")


if __name__ == "__main__":
    main()
