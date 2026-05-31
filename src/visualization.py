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
WITHOUT_CORRECTION_FILE = "fl_spam_drift_sem_correcao.png"
WITH_CORRECTION_FILE = "fl_spam_drift_com_correcao.png"

_COLORS = {
    "FL Padrão": "#2196F3",
    "Drift Súbito": "#F44336",
    "Súbito": "#F44336",
    "Drift Gradual": "#FF9800",
    "Gradual": "#FF9800",
    "Drift Recorrente": "#9C27B0",
    "Recorrente": "#9C27B0",
    "FL Adaptativo": "#4CAF50",
}
_MARKERS = {
    "FL Padrão": "o",
    "Drift Súbito": "s",
    "Súbito": "s",
    "Drift Gradual": "^",
    "Gradual": "^",
    "Drift Recorrente": "D",
    "Recorrente": "D",
    "FL Adaptativo": "*",
}
_LINESTYLES = {
    "FL Padrão": "-",
    "Drift Súbito": "--",
    "Súbito": "-",
    "Drift Gradual": "-.",
    "Gradual": "-",
    "Drift Recorrente": ":",
    "Recorrente": "-",
    "FL Adaptativo": "-",
}


def _base_label(label: str) -> str:
    for key in _COLORS:
        if key in label:
            return key
    return label


def _style_for_label(label: str) -> tuple[str, str, str]:
    base = _base_label(label)
    linestyle = "--" if "sem correção" in label else _LINESTYLES.get(base, "-")
    return (
        _COLORS.get(base, "#607D8B"),
        _MARKERS.get(base, "o"),
        linestyle,
    )


def plot_results(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Gera e salva o painel com quatro gráficos comparativos.

    Args:
        histories:   dict {label: (acc_history, f1_history)}
        drift_round: rodada em que o drift começa (linha vertical).
    """
    rounds = list(range(1, NUM_ROUNDS + 1))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    _plot_accuracy_over_rounds(ax1, histories, rounds, drift_round)
    _plot_f1_over_rounds(ax2, histories, rounds, drift_round)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Gráfico salvo: {OUTPUT_FILE}")


def plot_separated_results(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Gera gráficos separados para os cenários com e sem correção."""
    without_correction = _filter_histories(histories, "sem correção")
    with_correction = _filter_histories(histories, "com correção")

    _plot_results_to_file(
        without_correction,
        drift_round,
        "Federated Learning — Sem Correção de Concept Drift",
        WITHOUT_CORRECTION_FILE,
    )
    _plot_results_to_file(
        with_correction,
        drift_round,
        "Federated Learning — Com Correção de Concept Drift",
        WITH_CORRECTION_FILE,
    )


def _filter_histories(histories: dict, correction_label: str) -> dict:
    filtered = {"FL Padrão": histories["FL Padrão"]} if "FL Padrão" in histories else {}
    filtered.update({label: values for label, values in histories.items() if correction_label in label})
    return filtered


def _plot_results_to_file(histories: dict, drift_round: int, title: str, output_file: str) -> None:
    rounds = list(range(1, NUM_ROUNDS + 1))

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    _plot_accuracy_over_rounds(ax1, histories, rounds, drift_round, title)
    _plot_f1_over_rounds(ax2, histories, rounds, drift_round)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Gráfico salvo: {output_file}")


def _plot_f1_over_rounds(ax, histories, rounds, drift_round):
    for label, (_, f1_h) in histories.items():
        color, marker, linestyle = _style_for_label(label)
        ax.plot(
            rounds,
            f1_h,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=label,
        )
    ax.axvline(x=drift_round, color="black", linestyle="--", linewidth=1.5, label=f"Início do Drift (rodada {drift_round})")
    ax.axvspan(drift_round, NUM_ROUNDS, alpha=0.06, color="red", label="Período com Drift")
    ax.set_xlabel("Rodada de Comunicação", fontsize=12)
    ax.set_ylabel("F1-score no Teste (%)", fontsize=12)
    ax.set_title("F1-score — Robustez do Modelo frente ao Concept Drift", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, NUM_ROUNDS)


def _plot_accuracy_over_rounds(ax, histories, rounds, drift_round, title="Federated Learning — Com e Sem Correção de Concept Drift"):
    for label, (acc_h, _) in histories.items():
        color, marker, linestyle = _style_for_label(label)
        ax.plot(
            rounds,
            acc_h,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=label,
        )
    ax.axvline(x=drift_round, color="black", linestyle="--", linewidth=1.5, label=f"Início do Drift (rodada {drift_round})")
    ax.axvspan(drift_round, NUM_ROUNDS, alpha=0.06, color="red", label="Período com Drift")
    ax.set_xlabel("Rodada de Comunicação", fontsize=12)
    ax.set_ylabel("Acurácia no Teste (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, NUM_ROUNDS)


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
    W = 92
    print(f"\n{'═' * W}")
    print("  RESUMO FINAL — FL com Concept Drift: Spam de E-mail")
    print(f"{'═' * W}")
    print(f"  {'Cenário':<28} │ {'Acc Final':>9} │ {'F1 Final':>8} │ {'Queda Acc':>10} │ {'Recuperação':>12}")
    print(f"  {'-' * 82}")

    for label, (acc_h, f1_h) in histories.items():
        pre = np.mean(acc_h[: drift_round - 1]) if drift_round > 1 else acc_h[0]
        post = np.mean(acc_h[drift_round - 1 :])
        rec = _recovery_rounds(acc_h, drift_round)
        print(f"  {label:<28} │ {acc_h[-1]:>8.1f}% │ {f1_h[-1]:>7.1f}% │ {pre - post:>8.1f} p.p. │ {rec:>9} rod.")

    print(f"{'═' * W}")
    print("\n  Configuração:")
    print(f"    • Clientes FL:      {NUM_CLIENTS}")
    print(f"    • Rodadas:          {NUM_ROUNDS}")
    print(f"    • Épocas locais:    {LOCAL_EPOCHS}")
    print(f"    • Início do drift:  rodada {DRIFT_ROUND}")
    print(f"    • Dataset:          sintético — {FEATURE_DIM} features (spam/ham)")
    print(f"    • Ciclo recorrente: {CYCLE_LEN} rodadas por fase")
    print(f"{'═' * W}\n")
