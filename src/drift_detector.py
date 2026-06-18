class PageHinkleyDetector:
    """
    Detector de Concept Drift baseado no Teste Estatístico de Page-Hinkley.
    Adaptado para monitorizar aumentos no erro contínuo (como o MAE).
    """

    def __init__(self, threshold: float = 0.10, delta: float = 0.005, burn_in: int = 3):
        """
        Parâmetros:
        - threshold (lambda): Limite cumulativo. Se a soma dos aumentos de erro passar disto, é drift.
        - delta: Tolerância mínima. Variações de erro muito pequenas não são somadas.
        - burn_in: Número de rodadas iniciais (meses) ignoradas para a média estabilizar.
        """
        self.threshold = threshold
        self.delta = delta
        self.burn_in = burn_in

        # Variáveis de estado da memória do Page-Hinkley
        self.n_samples = 0
        self.cumulative_mean = 0.0
        self.cumulative_sum = 0.0

    def update_and_check(self, loss_atual: float) -> bool:
        self.n_samples += 1

        # 1. Atualiza a média histórica de forma incremental
        self.cumulative_mean += (loss_atual - self.cumulative_mean) / self.n_samples

        # Ignora as primeiras rodadas enquanto o modelo ainda está a aprender a média base
        if self.n_samples < self.burn_in:
            return False

        # 2. Equação do Page-Hinkley: m_t = max(0, m_{t-1} + (x_t - média - delta))
        # Queremos detetar se o erro está a AUMENTAR, logo medimos a diferença positiva.
        higher = (loss_atual - self.cumulative_mean) - self.delta

        # Acumula o valor. Se o erro descer, a soma estabiliza em zero (max 0).
        self.cumulative_sum = max(0.0, self.cumulative_sum + higher)

        # 3. Verifica se a soma cumulativa ultrapassou o limiar de alarme
        if self.cumulative_sum > self.threshold:
            # Reinicia os contadores para recomeçar a análise após o drift
            self._reset()
            return True

        return False

    def _reset(self):
        """Limpa a memória do detector para se adaptar à nova realidade climática."""
        self.n_samples = 0
        self.cumulative_mean = 0.0
        self.cumulative_sum = 0.0
