"""
drift_detector.py
─────────────────
Detector de concept drift por janela móvel.

Estratégia: mantém um histórico FIFO das últimas N losses (ou MAEs). Quando a
janela está cheia, compara o valor mais recente com a média dos demais. Se o
valor atual ultrapassar `média × limiar`, considera-se que houve drift.

Detector puramente observacional: apenas sinaliza, não age sobre o modelo.
"""

import numpy as np


class DetectorDeDrift:
    def __init__(self, tamanho_janela: int = 5, limiar_alerta: float = 1.3):
        self.tamanho_janela = tamanho_janela
        self.limiar_alerta = limiar_alerta
        self.historico_loss: list[float] = []

    def atualizar_e_checar(self, loss_atual: float) -> bool:
        self.historico_loss.append(float(loss_atual))

        if len(self.historico_loss) > self.tamanho_janela:
            self.historico_loss.pop(0)

        if len(self.historico_loss) == self.tamanho_janela:
            media_passada = float(np.mean(self.historico_loss[:-1]))
            if loss_atual > media_passada * self.limiar_alerta:
                return True
        return False
