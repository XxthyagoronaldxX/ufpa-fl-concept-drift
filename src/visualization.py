"""
visualization.py
────────────────
Geração do gráfico e tabela de resumo final (regressão de potência eólica).
A métrica visual exibida é a Acurácia = 100 − MAE (em p.p. de Power).
"""

from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt

from config import NUM_ROUNDS, DRIFT_ROUND, NUM_CLIENTS, LOCAL_EPOCHS, CYCLE_LEN, FEATURE_DIM

OUTPUT_FILE = "fl_wind_drift_results.png"

_COLORS = {
    "FL Padrão": "#2196F3",
    "FL Drift Recorrente": "#9C27B0",
    "FL Drift Recorrente (Com correção)": "#4CAF50",
}
_MARKERS = {
    "FL Padrão": "o",
    "FL Drift Recorrente": "D",
    "FL Drift Recorrente (Com correção)": "s",
}


def _style_for_label(label: str) -> tuple[str, str]:
    return _COLORS.get(label, "#607D8B"), _MARKERS.get(label, "o")


def _mae_of(entry) -> list:
    return entry["mae"] if isinstance(entry, dict) else entry[0]


def _acc_from_mae(mae_history: list) -> list:
    return [100.0 - x for x in mae_history]


def plot_results(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Gera e salva o painel de Acurácia comparativo (Acurácia = 100 − MAE).

    Args:
        histories:   dict {label: dict com 'mae'} — valores em p.p.
        drift_round: rodada em que o drift começa (linha vertical).
    """
    rounds = list(range(1, NUM_ROUNDS + 1))

    _, ax = plt.subplots(figsize=(16, 6))

    for label, entry in histories.items():
        acc_h = _acc_from_mae(_mae_of(entry))
        color, marker = _style_for_label(label)
        ax.plot(
            rounds,
            acc_h,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=5,
            label=label,
        )

    ax.axvline(x=drift_round, color="black", linestyle="--", linewidth=1.5, label=f"Início do Drift (rodada {drift_round})")
    ax.axvspan(drift_round, NUM_ROUNDS, alpha=0.06, color="red", label="Período com Drift")
    ax.set_xlabel("Rodada de Comunicação", fontsize=12)
    ax.set_ylabel("Acurácia no Teste (%) — maior é melhor", fontsize=12)
    ax.set_title("Federated Learning (Wind Power) — Acurácia por Rodada", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.xaxis.set_major_locator(MultipleLocator(5))

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Gráfico salvo: {OUTPUT_FILE}")


def _recovery_rounds(acc_history: list, drift_round: int, tolerance_pp: float = 1.0) -> str:
    """Rodadas até a acurácia recente voltar a ficar dentro de tolerance_pp do pico pré-drift."""
    if drift_round <= 1 or drift_round > len(acc_history):
        return "—"
    pre_max = max(acc_history[: drift_round - 1])
    target = pre_max - tolerance_pp
    for i in range(drift_round - 1, len(acc_history)):
        if acc_history[i] >= target:
            return str(i - (drift_round - 1) + 1)
    return f">{len(acc_history) - drift_round + 1}"


def print_summary(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Imprime tabela de resumo com Acurácia final, queda pós-drift e tempo de recuperação."""
    W = 88
    print(f"\n{'═' * W}")
    print("  RESUMO FINAL — FL com Concept Drift: Geração de Energia Eólica")
    print(f"{'═' * W}")
    print(f"  {'Cenário':<36} │ {'Acur. Final':>11} │ {'Queda Acur.':>11} │ {'Recuperação':>12}")
    print(f"  {'-' * 86}")

    for label, entry in histories.items():
        acc_h = _acc_from_mae(_mae_of(entry))
        pre = np.mean(acc_h[: drift_round - 1]) if drift_round > 1 else acc_h[0]
        post = np.mean(acc_h[drift_round - 1 :])
        rec = _recovery_rounds(acc_h, drift_round)
        drop = pre - post
        print(f"  {label:<36} │ {acc_h[-1]:>10.2f}% │ {drop:>9.2f} p.p. │ {rec:>9} rod.")

    print(f"{'═' * W}")
    print("\n  Configuração:")
    print(f"    • Clientes FL:      {NUM_CLIENTS} (1 por local)")
    print(f"    • Rodadas:          {NUM_ROUNDS}")
    print(f"    • Épocas locais:    {LOCAL_EPOCHS}")
    print(f"    • Início do drift:  rodada {DRIFT_ROUND}")
    print(f"    • Dataset:          wind power — {FEATURE_DIM} features (regressão)")
    print(f"    • Ciclo recorrente: {CYCLE_LEN} rodadas por fase")
    print(f"{'═' * W}\n")
