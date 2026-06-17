class PageHinkleyDetector:
    """
    Detector de Concept Drift baseado no Teste Estatístico de Page-Hinkley.
    Excelente para monitoramento sequencial de erros contínuos (como MAE em regressão).
    """

    def __init__(self, threshold: float = 0.10, delta: float = 0.005, burn_in: int = 3):
        """
        Parâmetros:
        - threshold (lambda): O limite de tolerância cumulativa. Se a soma dos erros passar disso, é drift!
        - delta: A magnitude mínima de variação permitida (filtra pequenos ruídos normais).
        - burn_in: Número de rodadas iniciais ignoradas para permitir que a média estabilize.
        """
        self.threshold = threshold
        self.delta = delta
        self.burn_in = burn_in

        # Variáveis de estado
        self.n_samples = 0
        self.cumulative_mean = 0.0
        self.cumulative_sum = 0.0  # O famoso 'm_t' na matemática do PH

    def update_and_check(self, current_loss: float) -> bool:
        """Atualiza as estatísticas e retorna True se detectar Concept Drift."""
        self.n_samples += 1

        # 1. Atualiza a média histórica de forma incremental (economiza memória)
        difference_to_mean = current_loss - self.cumulative_mean
        self.cumulative_mean += difference_to_mean / self.n_samples

        # Ignora as primeiras rodadas enquanto o modelo ainda está aprendendo o básico
        if self.n_samples < self.burn_in:
            return False

        # 2. Calcula a estatística de Page-Hinkley
        # Queremos detectar se o erro está AUMENTANDO, então pegamos a diferença positiva.
        # Subtraímos o 'delta' para não penalizar flutuações microscópicas naturais.
        increase_detected = (current_loss - self.cumulative_mean) - self.delta

        # Acumula o valor. Se a soma ficar negativa (o erro melhorou), cravamos em zero.
        self.cumulative_sum = max(0.0, self.cumulative_sum + increase_detected)

        # 3. Checa o limite de Drift
        if self.cumulative_sum > self.threshold:
            self._reset()  # Reinicia a memória do detector para a nova realidade climática
            return True

        return False

    def _reset(self):
        """Zera os contadores após um drift para recomeçar a análise do zero."""
        self.n_samples = 0
        self.cumulative_mean = 0.0
        self.cumulative_sum = 0.0
