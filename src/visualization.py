"""
visualization.py
────────────────
Geração dos gráficos e tabela de resumo final (regressão de potência eólica).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import NUM_ROUNDS, DRIFT_ROUND, NUM_CLIENTS, LOCAL_EPOCHS, CYCLE_LEN, FEATURE_DIM, TOLERANCE_ACC

OUTPUT_FILE = "fl_wind_drift_results.png"
WITHOUT_CORRECTION_FILE = "fl_wind_drift_sem_correcao.png"
WITH_CORRECTION_FILE = "fl_wind_drift_com_correcao.png"
TREATMENT_FILE = "fl_wind_drift_tratamento.png"
ACCURACY_FILE = "fl_wind_drift_accuracy.png"

_SEVERITY_COLORS = {
    "baixa": "#FFC107",
    "média": "#FF9800",
    "alta": "#E53935",
}

_COLORS = {
    "FL Padrão": "#2196F3",
    "Drift Recorrente": "#9C27B0",
    "Recorrente": "#9C27B0",
    "FL Adaptativo": "#4CAF50",
}
_MARKERS = {
    "FL Padrão": "o",
    "Drift Recorrente": "D",
    "Recorrente": "D",
    "FL Adaptativo": "*",
}
_LINESTYLES = {
    "FL Padrão": "-",
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


def _mae_of(entry) -> list:
    return entry["mae"] if isinstance(entry, dict) else entry[0]


def _rmse_of(entry) -> list:
    return entry["rmse"] if isinstance(entry, dict) else entry[1]


def _events_of(entry) -> list:
    return entry["events"] if isinstance(entry, dict) else []


def _acc_of(entry) -> list:
    """Extrai a série de acurácia tolerante por rodada (∈ [0, 1])."""
    if isinstance(entry, dict) and "acc" in entry:
        return entry["acc"]
    return []


def plot_results(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Gera e salva o painel com MAE e RMSE comparativos.

    Args:
        histories:   dict {label: dict com 'mae'/'rmse'/'events'} — valores em p.p.
        drift_round: rodada em que o drift começa (linha vertical).
    """
    rounds = list(range(1, NUM_ROUNDS + 1))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    _plot_mae_over_rounds(ax1, histories, rounds, drift_round)
    _plot_rmse_over_rounds(ax2, histories, rounds, drift_round)

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
        "Federated Learning (Wind Power) — Sem Correção de Concept Drift",
        WITHOUT_CORRECTION_FILE,
    )
    _plot_results_to_file(
        with_correction,
        drift_round,
        "Federated Learning (Wind Power) — Com Correção de Concept Drift",
        WITH_CORRECTION_FILE,
    )


def _filter_histories(histories: dict, correction_label: str) -> dict:
    filtered = {"FL Padrão": histories["FL Padrão"]} if "FL Padrão" in histories else {}
    filtered.update({label: values for label, values in histories.items() if correction_label in label})
    return filtered


def plot_accuracy(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Painel comparativo da acurácia tolerante (ε-accuracy) por rodada.

    Métrica: para cada rodada, fração de previsões com |ŷ − y| < TOLERANCE_ACC.
    Mostra o quão "confiáveis" são as previsões além do erro médio — útil para
    visualizar o impacto sazonal: o inverno derruba bruscamente a acurácia
    porque a cauda do erro fica mais pesada, mesmo quando o MAE varia pouco.
    """
    rounds = list(range(1, NUM_ROUNDS + 1))
    tolerance_pp = TOLERANCE_ACC * 100.0

    fig, ax = plt.subplots(figsize=(14, 6))

    # Faixas verticais alternadas marcando as fases sazonais pós-drift.
    for i, start in enumerate(range(drift_round, NUM_ROUNDS + 1, CYCLE_LEN)):
        end = min(start + CYCLE_LEN, NUM_ROUNDS + 1)
        is_winter = i % 2 == 0
        color = "#90CAF9" if is_winter else "#FFE082"  # azul = inverno, amarelo = verão
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.18, color=color, zorder=0)

    for label, entry in histories.items():
        acc_h = _acc_of(entry)
        if not acc_h:
            continue
        color, marker, linestyle = _style_for_label(label)
        ax.plot(
            rounds,
            [a * 100.0 for a in acc_h],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=6,
            label=label,
            zorder=3,
        )

    ax.axvline(x=drift_round, color="black", linestyle="--", linewidth=1.5, zorder=2, label=f"Início do Drift (rodada {drift_round})")

    # Legendas auxiliares para as faixas sazonais (uma vez só).
    season_handles = [
        mpatches.Patch(color="#90CAF9", alpha=0.45, label="Fase inverno"),
        mpatches.Patch(color="#FFE082", alpha=0.45, label="Fase verão"),
    ]
    primary = ax.legend(loc="lower left", fontsize=10)
    ax.add_artist(primary)
    ax.legend(handles=season_handles, loc="lower right", fontsize=9, framealpha=0.85)

    ax.set_xlabel("Rodada de Comunicação", fontsize=12)
    ax.set_ylabel(f"Acurácia (%) — |erro| < {tolerance_pp:.0f} p.p.", fontsize=12)
    ax.set_title(
        f"Acurácia tolerante por rodada (ε = {tolerance_pp:.0f} p.p. de Power) — maior é melhor",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ACCURACY_FILE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Gráfico salvo: {ACCURACY_FILE}")


def _plot_results_to_file(histories: dict, drift_round: int, title: str, output_file: str) -> None:
    rounds = list(range(1, NUM_ROUNDS + 1))

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    _plot_mae_over_rounds(ax1, histories, rounds, drift_round, title)
    _plot_rmse_over_rounds(ax2, histories, rounds, drift_round)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Gráfico salvo: {output_file}")


def _plot_rmse_over_rounds(ax, histories, rounds, drift_round):
    for label, entry in histories.items():
        rmse_h = _rmse_of(entry)
        color, marker, linestyle = _style_for_label(label)
        ax.plot(
            rounds,
            rmse_h,
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
    ax.set_ylabel("RMSE no Teste (p.p. de Power) — menor é melhor", fontsize=12)
    ax.set_title("RMSE — Robustez do Modelo frente ao Concept Drift", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, NUM_ROUNDS)


def _plot_mae_over_rounds(ax, histories, rounds, drift_round, title="Federated Learning (Wind Power) — Com e Sem Correção"):
    for label, entry in histories.items():
        mae_h = _mae_of(entry)
        color, marker, linestyle = _style_for_label(label)
        ax.plot(
            rounds,
            mae_h,
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
    ax.set_ylabel("MAE no Teste (p.p. de Power) — menor é melhor", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, NUM_ROUNDS)


def _recovery_rounds(mae_history: list, drift_round: int, tolerance_pp: float = 1.0) -> str:
    """Rodadas até o MAE recente voltar a ficar dentro de tolerance_pp do mínimo pré-drift."""
    if drift_round <= 1 or drift_round > len(mae_history):
        return "—"
    pre_min = min(mae_history[: drift_round - 1])
    target = pre_min + tolerance_pp
    for i in range(drift_round - 1, len(mae_history)):
        if mae_history[i] <= target:
            return str(i - (drift_round - 1) + 1)
    return f">{len(mae_history) - drift_round + 1}"


def print_summary(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Imprime tabela de resumo com MAE/RMSE finais, alta pós-drift e tempo de recuperação."""
    W = 92
    print(f"\n{'═' * W}")
    print("  RESUMO FINAL — FL com Concept Drift: Geração de Energia Eólica")
    print(f"{'═' * W}")
    print(f"  {'Cenário':<28} │ {'MAE Final':>9} │ {'RMSE Final':>10} │ {'Alta MAE':>9} │ {'Recuperação':>12}")
    print(f"  {'-' * 90}")

    for label, entry in histories.items():
        mae_h = _mae_of(entry)
        rmse_h = _rmse_of(entry)
        pre = np.mean(mae_h[: drift_round - 1]) if drift_round > 1 else mae_h[0]
        post = np.mean(mae_h[drift_round - 1 :])
        rec = _recovery_rounds(mae_h, drift_round)
        rise = post - pre
        print(f"  {label:<28} │ {mae_h[-1]:>8.2f}% │ {rmse_h[-1]:>9.2f}% │ {rise:>7.2f} p.p. │ {rec:>9} rod.")

    print(f"{'═' * W}")
    print("\n  Configuração:")
    print(f"    • Clientes FL:      {NUM_CLIENTS} (1 por local)")
    print(f"    • Rodadas:          {NUM_ROUNDS}")
    print(f"    • Épocas locais:    {LOCAL_EPOCHS}")
    print(f"    • Início do drift:  rodada {DRIFT_ROUND}")
    print(f"    • Dataset:          wind power — {FEATURE_DIM} features (regressão)")
    print(f"    • Ciclo recorrente: {CYCLE_LEN} rodadas por fase")
    print(f"{'═' * W}\n")


# ── Gráfico didático: tratamento da correção de drift ───────────────────────


def _contiguous_runs(flags: list) -> list:
    """Retorna (start_round, end_round) inclusivos para cada bloco contíguo de True."""
    runs = []
    start = None
    for i, flag in enumerate(flags, start=1):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _phase_runs(events: list) -> list:
    """Agrupa rodadas em blocos contíguos por fase sazonal (verão/inverno)."""
    if not events:
        return []
    runs = []
    cur_phase = events[0].get("phase") or ""
    start = events[0]["round"]
    for ev in events[1:]:
        phase = ev.get("phase") or ""
        if phase != cur_phase:
            runs.append((start, ev["round"] - 1, cur_phase))
            cur_phase = phase
            start = ev["round"]
    runs.append((start, events[-1]["round"], cur_phase))
    return runs


def plot_correction_treatment(histories: dict, drift_round: int = DRIFT_ROUND) -> None:
    """Gráfico didático: anota a curva de erro com o tratamento aplicado ao drift.

    Mostra três painéis empilhados (eixo X compartilhado):
      1. MAE com correção (verde) vs. sem correção (cinza), marcadores de detecção
         por severidade e faixas indicando rodadas com correção ativa.
      2. Alavancas adaptativas: learning rate efetivo e épocas locais (linha-degrau).
      3. Razão de replay efetiva (área preenchida).
    """
    target_label = next((lbl for lbl in histories if "com correção" in lbl), None)
    reference_label = next((lbl for lbl in histories if "sem correção" in lbl), None)
    if target_label is None:
        print("[INFO] plot_correction_treatment: cenário 'com correção' ausente; gráfico pulado.")
        return

    target = histories[target_label]
    target_mae = _mae_of(target)
    events = _events_of(target)
    if not events:
        print("[INFO] plot_correction_treatment: telemetria de eventos ausente; gráfico pulado.")
        return

    rounds = [ev["round"] for ev in events]
    detected_flags = [ev["detected"] for ev in events]
    severities = [ev["severity"] for ev in events]
    correction_flags = [ev["correction_active"] for ev in events]
    lr_series = [ev["lr"] for ev in events]
    epochs_series = [ev["epochs"] for ev in events]
    replay_series = [ev["replay_ratio"] for ev in events]

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[2.4, 0.8, 1.0, 0.9], hspace=0.30)
    ax_mae = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[1, 0], sharex=ax_mae)
    ax_lev = fig.add_subplot(gs[2, 0], sharex=ax_mae)
    ax_rep = fig.add_subplot(gs[3, 0], sharex=ax_mae)

    # Faixas sazonais sutis ao fundo do painel principal
    for start, end, phase in _phase_runs(events):
        if not phase:
            continue
        color = "#1976D2" if "inverno" in phase else "#FB8C00"
        ax_mae.axvspan(start - 0.5, end + 0.5, alpha=0.05, color=color, zorder=0)

    # Faixas de correção ativa coloridas pela severidade do drift que as disparou
    correction_runs = _contiguous_runs(correction_flags)
    for start, end in correction_runs:
        sev_in_run = "baixa"
        # detecção que disparou geralmente está uma rodada antes do bloco
        if start - 2 >= 0 and detected_flags[start - 2] and severities[start - 2] != "none":
            sev_in_run = severities[start - 2]
        else:
            for i in range(start, end + 1):
                if detected_flags[i - 1] and severities[i - 1] != "none":
                    sev_in_run = severities[i - 1]
                    break
        color = _SEVERITY_COLORS.get(sev_in_run, "#4CAF50")
        ax_mae.axvspan(start - 0.5, end + 0.5, alpha=0.20, color=color, zorder=1)
        ax_delta.axvspan(start - 0.5, end + 0.5, alpha=0.12, color=color, zorder=0)
        ax_lev.axvspan(start - 0.5, end + 0.5, alpha=0.12, color=color, zorder=0)
        ax_rep.axvspan(start - 0.5, end + 0.5, alpha=0.12, color=color, zorder=0)

    # Linha vertical do início do drift
    for ax in (ax_mae, ax_delta, ax_lev, ax_rep):
        ax.axvline(x=drift_round, color="black", linestyle="--", linewidth=1.4, alpha=0.8, zorder=2)

    # Curva de referência "sem correção" (cinza pontilhada)
    if reference_label is not None:
        ref_mae = _mae_of(histories[reference_label])
        ax_mae.plot(rounds, ref_mae, color="#9E9E9E", linestyle=":", linewidth=2.0, marker="D", markersize=4, label="MAE sem correção (referência)", zorder=3)

    # Curva principal "com correção"
    ax_mae.plot(rounds, target_mae, color="#2E7D32", linestyle="-", linewidth=2.4, marker="*", markersize=8, label="MAE com correção", zorder=4)

    # Marcadores de detecção (triângulo invertido por severidade)
    first_detection_annotated = False
    mae_range = max(target_mae) - min(target_mae) if target_mae else 1.0
    offset = max(1.5, 0.06 * mae_range)
    for i, ev in enumerate(events):
        if not ev["detected"]:
            continue
        sev = ev["severity"] if ev["severity"] != "none" else "baixa"
        color = _SEVERITY_COLORS.get(sev, "#E53935")
        y = target_mae[i]
        ax_mae.scatter([ev["round"]], [y + offset], marker="v", s=120, color=color, edgecolors="black", linewidths=0.8, zorder=5)
        if not first_detection_annotated:
            ax_mae.annotate(
                f"DETECTADO ({sev})",
                xy=(ev["round"], y + offset),
                xytext=(ev["round"] + 1.2, y + offset + 1.8 * offset),
                fontsize=9,
                fontweight="bold",
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                zorder=6,
            )
            first_detection_annotated = True

    # Anotação automática do maior "ganho" pós-drift
    if reference_label is not None:
        ref_mae = _mae_of(histories[reference_label])
        post_indices = [i for i, r in enumerate(rounds) if r >= drift_round]
        if post_indices:
            best_idx = max(post_indices, key=lambda i: ref_mae[i] - target_mae[i])
            gain = ref_mae[best_idx] - target_mae[best_idx]
            if gain > 0.5:
                r = rounds[best_idx]
                y_low = target_mae[best_idx]
                y_high = ref_mae[best_idx]
                ax_mae.annotate(
                    "",
                    xy=(r, y_low),
                    xytext=(r, y_high),
                    arrowprops=dict(arrowstyle="<->", color="#1B5E20", lw=1.6),
                    zorder=6,
                )
                ax_mae.text(
                    r + 0.3,
                    (y_low + y_high) / 2,
                    f"ganho ≈ {gain:.1f} p.p.",
                    fontsize=10,
                    fontweight="bold",
                    color="#1B5E20",
                    va="center",
                    zorder=6,
                )

    # Legenda do painel principal incluindo proxies de severidade
    severity_handles = [
        mpatches.Patch(color=_SEVERITY_COLORS["baixa"], alpha=0.5, label="Correção ativa — drift baixo"),
        mpatches.Patch(color=_SEVERITY_COLORS["média"], alpha=0.5, label="Correção ativa — drift médio"),
        mpatches.Patch(color=_SEVERITY_COLORS["alta"], alpha=0.5, label="Correção ativa — drift alto"),
    ]
    drift_line = plt.Line2D([0], [0], color="black", linestyle="--", label=f"Início do drift (rod. {drift_round})")
    handles, labels = ax_mae.get_legend_handles_labels()
    extra = severity_handles + [drift_line]
    ax_mae.legend(handles + extra, labels + [h.get_label() for h in extra], fontsize=9, loc="lower left", ncol=2)
    ax_mae.set_ylabel("MAE (p.p. de Power)", fontsize=11)
    ax_mae.set_title("Tratamento do Concept Drift — Detecção, Correção Ativa e Impacto no MAE", fontsize=13, fontweight="bold")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.set_xlim(0.5, NUM_ROUNDS + 0.5)

    # ── Inset de zoom no período pós-drift (se houver referência) ──
    if reference_label is not None:
        ref_mae = _mae_of(histories[reference_label])
        winter_indices = [i for i, r in enumerate(rounds) if r >= drift_round and target_mae[i] > 16.0]
        if len(winter_indices) >= 2:
            x_lo = min(rounds[i] for i in winter_indices) - 0.5
            x_hi = max(rounds[i] for i in winter_indices) + 0.5
            ys = [target_mae[i] for i in winter_indices] + [ref_mae[i] for i in winter_indices]
            y_lo = min(ys) - 0.25
            y_hi = max(ys) + 0.25
            axins = inset_axes(ax_mae, width="34%", height="38%", loc="upper right", borderpad=1.2)
            for start, end in correction_runs:
                axins.axvspan(start - 0.5, end + 0.5, alpha=0.18, color="#4CAF50", zorder=0)
            axins.plot(rounds, ref_mae, color="#9E9E9E", linestyle=":", linewidth=1.8, marker="D", markersize=4, zorder=3)
            axins.plot(rounds, target_mae, color="#2E7D32", linestyle="-", linewidth=2.0, marker="*", markersize=7, zorder=4)
            axins.set_xlim(x_lo, x_hi)
            axins.set_ylim(y_lo, y_hi)
            axins.tick_params(axis="both", labelsize=8)
            axins.set_title("Zoom: invernos pós-drift", fontsize=9, fontweight="bold")
            axins.grid(True, alpha=0.3)
            ax_mae.indicate_inset_zoom(axins, edgecolor="#1B5E20", alpha=0.6)

    # ── Painel de delta: ganho ponto-a-ponto da correção ──
    if reference_label is not None:
        ref_mae = _mae_of(histories[reference_label])
        deltas = [ref_mae[i] - target_mae[i] for i in range(len(rounds))]
        bar_colors = []
        for d in deltas:
            if d > 0.05:
                bar_colors.append("#2E7D32")
            elif d < -0.05:
                bar_colors.append("#C62828")
            else:
                bar_colors.append("#BDBDBD")
        ax_delta.bar(rounds, deltas, color=bar_colors, edgecolor="black", linewidth=0.4, zorder=3)
        ax_delta.axhline(y=0, color="black", linewidth=0.8, zorder=2)

        # Anotar as 3 maiores barras (positivas e negativas) com o valor numérico
        sorted_idx = sorted(range(len(deltas)), key=lambda i: abs(deltas[i]), reverse=True)
        for i in sorted_idx[:3]:
            if abs(deltas[i]) < 0.1:
                continue
            va = "bottom" if deltas[i] >= 0 else "top"
            ax_delta.text(rounds[i], deltas[i], f"{deltas[i]:+.2f}", ha="center", va=va, fontsize=8, fontweight="bold", zorder=4)

        ax_delta.set_ylabel("Δ MAE (p.p.)", fontsize=10)
        ax_delta.grid(True, axis="y", alpha=0.3)
        gain_proxy = mpatches.Patch(color="#2E7D32", label="correção ajudou (>0)")
        loss_proxy = mpatches.Patch(color="#C62828", label="correção atrapalhou (<0)")
        ax_delta.legend(handles=[gain_proxy, loss_proxy], fontsize=8, loc="lower left", ncol=2)
        ax_delta.set_title("Ganho ponto-a-ponto: MAE(sem correção) − MAE(com correção)", fontsize=11)
    else:
        ax_delta.set_visible(False)

    # ── Painel 2: alavancas adaptativas (LR e épocas) ──
    ax_lev.step(rounds, lr_series, where="mid", color="#1565C0", linewidth=2.0, label="Learning rate efetivo")
    ax_lev.set_ylabel("Learning rate", fontsize=10, color="#1565C0")
    ax_lev.tick_params(axis="y", labelcolor="#1565C0")
    ax_lev.grid(True, alpha=0.3)
    ax_lev_e = ax_lev.twinx()
    ax_lev_e.step(rounds, epochs_series, where="mid", color="#6A1B9A", linewidth=2.0, linestyle="--", label="Épocas locais")
    ax_lev_e.set_ylabel("Épocas locais", fontsize=10, color="#6A1B9A")
    ax_lev_e.tick_params(axis="y", labelcolor="#6A1B9A")
    ax_lev_e.set_ylim(bottom=0, top=max(epochs_series) + 1)
    h1, l1 = ax_lev.get_legend_handles_labels()
    h2, l2 = ax_lev_e.get_legend_handles_labels()
    ax_lev.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax_lev.set_title("Alavancas adaptativas aplicadas durante a correção", fontsize=11)

    # ── Painel 3: replay ratio ──
    ax_rep.fill_between(rounds, replay_series, step="mid", color="#00897B", alpha=0.55, label="Replay ratio efetivo")
    ax_rep.step(rounds, replay_series, where="mid", color="#004D40", linewidth=1.5)
    ax_rep.set_ylabel("Replay ratio", fontsize=10)
    ax_rep.set_xlabel("Rodada de Comunicação", fontsize=11)
    ax_rep.set_ylim(0, max(replay_series + [0.6]) + 0.05)
    ax_rep.grid(True, alpha=0.3)
    ax_rep.legend(fontsize=9, loc="upper left")
    ax_rep.set_title("Memória recente reinjetada (anti-esquecimento)", fontsize=11)

    plt.savefig(TREATMENT_FILE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Gráfico salvo: {TREATMENT_FILE}")
