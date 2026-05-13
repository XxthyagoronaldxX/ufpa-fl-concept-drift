"""
visualization.py
────────────────
Geração dos gráficos e tabela de resumo final.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import NUM_ROUNDS, DRIFT_ROUND, NUM_CLIENTS, LOCAL_EPOCHS, CYCLE_LEN, FEATURE_DIM

OUTPUT_FILE = "fl_spam_drift_results.png"

_COLORS = {
    "FL Padrão": "#2196F3",
    "Drift Súbito": "#F44336",
    "Drift Gradual": "#FF9800",
    "Drift Recorrente": "#9C27B0",
    "Non-IID Parcial": "#795548",
    "FL Adaptativo": "#4CAF50",
}
_MARKERS = {
    "FL Padrão": "o",
    "Drift Súbito": "s",
    "Drift Gradual": "^",
    "Drift Recorrente": "D",
    "Non-IID Parcial": "P",
    "FL Adaptativo": "*",
}
_LINESTYLES = {
    "FL Padrão": "-",
    "Drift Súbito": "--",
    "Drift Gradual": "-.",
    "Drift Recorrente": ":",
    "Non-IID Parcial": (0, (3, 1, 1, 1)),
    "FL Adaptativo": "-",
}


def plot_results(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Gera e salva o painel com quatro gráficos comparativos.

    Args:
        histories:   dict {label: (acc_history, f1_history)}
        drift_round: rodada em que o drift começa (linha vertical).
    """
    rounds = list(range(1, NUM_ROUNDS + 1))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    _plot_accuracy_over_rounds(fig.add_subplot(gs[0, :]), histories, rounds, drift_round)
    _plot_f1_over_rounds(fig.add_subplot(gs[1, 0]), histories, rounds, drift_round)
    _plot_pre_post_accuracy(fig.add_subplot(gs[1, 1]), histories, drift_round)
    _plot_drift_impact(fig.add_subplot(gs[1, 2]), histories, drift_round)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Gráfico salvo: {OUTPUT_FILE}")


def _plot_accuracy_over_rounds(ax, histories, rounds, drift_round):
    for label, (acc_h, _) in histories.items():
        ax.plot(
            rounds,
            acc_h,
            color=_COLORS[label],
            marker=_MARKERS[label],
            linestyle=_LINESTYLES[label],
            linewidth=2,
            markersize=5,
            label=label,
        )
    ax.axvline(x=drift_round, color="black", linestyle="--", linewidth=1.5, label=f"Início do Drift (rodada {drift_round})")
    ax.axvspan(drift_round, NUM_ROUNDS, alpha=0.06, color="red", label="Período com Drift")
    ax.set_xlabel("Rodada de Comunicação", fontsize=12)
    ax.set_ylabel("Acurácia no Teste (%)", fontsize=12)
    ax.set_title("Federated Learning — Detecção de Spam: Padrão vs Concept Drift", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, NUM_ROUNDS)


def _plot_f1_over_rounds(ax, histories, rounds, drift_round):
    for label, (_, f1_h) in histories.items():
        ax.plot(rounds, f1_h, color=_COLORS[label], linestyle=_LINESTYLES[label], linewidth=2, label=label)
    ax.axvline(x=drift_round, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Rodada", fontsize=11)
    ax.set_ylabel("F1-Score (%)", fontsize=11)
    ax.set_title("F1-Score por Rodada", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_pre_post_accuracy(ax, histories, drift_round):
    labels_list = list(histories.keys())
    pre_list = [np.mean(histories[l][0][: drift_round - 1]) if drift_round > 1 else histories[l][0][0] for l in labels_list]
    post_list = [np.mean(histories[l][0][drift_round - 1 :]) for l in labels_list]

    x, w = np.arange(len(labels_list)), 0.35
    ax.bar(x - w / 2, pre_list, w, label="Pré-drift", color="#42A5F5", alpha=0.85)
    ax.bar(x + w / 2, post_list, w, label="Pós-drift", color="#EF5350", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=7, rotation=20)
    ax.set_ylabel("Acurácia Média (%)", fontsize=11)
    ax.set_title("Acurácia Média: Pré vs Pós Drift", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)


def _plot_drift_impact(ax, histories, drift_round):
    labels_list = list(histories.keys())
    pre_list = [np.mean(histories[l][0][: drift_round - 1]) if drift_round > 1 else histories[l][0][0] for l in labels_list]
    post_list = [np.mean(histories[l][0][drift_round - 1 :]) for l in labels_list]
    drops = [pre - post for pre, post in zip(pre_list, post_list)]

    bar_colors = [_COLORS[l] for l in labels_list]
    bars = ax.bar(labels_list, drops, color=bar_colors, alpha=0.85)

    for bar, val in zip(bars, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f} p.p.",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Queda de Acurácia (p.p.)", fontsize=11)
    ax.set_title("Impacto do Concept Drift", fontsize=11, fontweight="bold")
    ax.set_xticklabels(labels_list, fontsize=7, rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(drops) * 1.35, 1))


def _recovery_rounds(acc_history: list, drift_round: int, tolerance_pp: float = 2.0) -> str:
    """Rodadas necessárias para recuperar até tolerance_pp abaixo do pico pré-drift."""
    if drift_round <= 1 or drift_round > len(acc_history):
        return "—"
    pre_peak = max(acc_history[: drift_round - 1])
    target = pre_peak - tolerance_pp
    for i in range(drift_round - 1, len(acc_history)):
        if acc_history[i] >= target:
            return str(i - (drift_round - 1) + 1)  # rodadas após o drift
    return f">{len(acc_history) - drift_round + 1}"


def print_summary(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Imprime tabela de resumo com acurácia final, F1, queda pós-drift e tempo de recuperação."""
    W = 76
    print(f"\n{'═' * W}")
    print(f"  RESUMO FINAL — FL com Concept Drift: Spam de E-mail")
    print(f"{'═' * W}")
    print(f"  {'Cenário':<22} │ {'Acc Final':>9} │ {'F1 Final':>8} │ {'Queda Acc':>10} │ {'Recuperação':>12}")
    print(f"  {'-' * 70}")

    for label, (acc_h, f1_h) in histories.items():
        pre = np.mean(acc_h[: drift_round - 1]) if drift_round > 1 else acc_h[0]
        post = np.mean(acc_h[drift_round - 1 :])
        rec = _recovery_rounds(acc_h, drift_round)
        print(f"  {label:<22} │ {acc_h[-1]:>8.1f}% │ {f1_h[-1]:>7.1f}% │ {pre - post:>8.1f} p.p. │ {rec:>9} rod.")

    print(f"{'═' * W}")
    print(f"\n  Configuração:")
    print(f"    • Clientes FL:      {NUM_CLIENTS}")
    print(f"    • Rodadas:          {NUM_ROUNDS}")
    print(f"    • Épocas locais:    {LOCAL_EPOCHS}")
    print(f"    • Início do drift:  rodada {DRIFT_ROUND}")
    print(f"    • Dataset:          sintético — {FEATURE_DIM} features (spam/ham)")
    print(f"    • Ciclo recorrente: {CYCLE_LEN} rodadas por fase")
    print(f"{'═' * W}\n")
