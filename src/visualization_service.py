import matplotlib.pyplot as plt


class VisualizationService:
    @staticmethod
    def plot_mae_history(mae_history: list[float]):
        rodadas = list(range(1, len(mae_history) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(rodadas, mae_history, marker="o", linestyle="-", color="b", linewidth=2, markersize=6)
        plt.title("Evolução do Erro do Modelo Global (Federated Learning)", fontsize=14, fontweight="bold")
        plt.xlabel("Rodadas Globais (Rounds)", fontsize=12)
        plt.ylabel("Erro Absoluto Médio (MAE)", fontsize=12)
        plt.xticks(rodadas)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_comparison(mae_history: list[float], mae_history_with_correction: list[float]):
        rodadas = list(range(1, len(mae_history) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(rodadas, mae_history, marker="o", linestyle="-", color="b", linewidth=2, markersize=6, label="Sem Correção")
        plt.plot(rodadas, mae_history_with_correction, marker="o", linestyle="-", color="r", linewidth=2, markersize=6, label="Com Correção")
        plt.title("Comparação do Erro do Modelo Global (Federated Learning)", fontsize=14, fontweight="bold")
        plt.xlabel("Rodadas Globais (Rounds)", fontsize=12)
        plt.ylabel("Erro Absoluto Médio (MAE)", fontsize=12)
        plt.xticks(rodadas)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
