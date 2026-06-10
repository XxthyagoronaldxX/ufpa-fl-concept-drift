# Resumo dos documentos em `docs/`

## O que são os documentos

Os dois arquivos em [docs/](docs/) são, na verdade, **um único trabalho** dividido
em duas partes:

- [docs/artigo.pdf](docs/artigo.pdf) — Gama, Žliobaitė, Bifet, Pechenizkiy e
  Bouchachia (2014). *"A Survey on Concept Drift Adaptation"*, ACM Computing
  Surveys, vol. 46, nº 4, art. 44 (37 páginas).
- [docs/a44-gama-apndx.pdf](docs/a44-gama-apndx.pdf) — *"Online Appendix"* do
  mesmo survey (9 páginas), contendo pseudocódigo dos algoritmos clássicos
  (Page-Hinkley, ADWIN, SPC, CVFDT, DWM, AddExp) e um catálogo de datasets
  sintéticos e reais com concept drift.

Não se trata, portanto, de um artigo descrevendo um experimento específico, e sim
de uma **revisão sistemática** sobre adaptação a concept drift em aprendizado
online supervisionado. É a referência teórica usada como base do projeto deste
workspace (FL para detecção de spam com cenários de drift).

## Conteúdo do survey

### 1. Definição e tipos de drift

O survey formaliza concept drift entre dois instantes $t_0$ e $t_1$ como:

$$\exists X : p_{t_0}(X, y) \ne p_{t_1}(X, y)$$

E distingue dois grandes tipos:

- **Real concept drift** — muda $p(y \mid X)$, ou seja, a fronteira de decisão.
  É o foco central do survey.
- **Virtual drift** — muda apenas $p(X)$ sem afetar $p(y \mid X)$.

Quanto à *forma temporal* da mudança (Figura 2 do artigo), o survey trata:

- **Súbito / abrupto** — substituição instantânea de um conceito por outro.
- **Gradual** — alternância probabilística entre conceitos durante uma
  transição.
- **Incremental** — passagem por muitos conceitos intermediários.
- **Recorrente** — conceitos antigos voltam após algum tempo (sazonalidade).
- *Outliers / ruído* — não são drift, e o sistema deve ser robusto a eles.

### 2. Aplicações motivadoras

O artigo organiza aplicações em quatro categorias e usa exemplos como controle
de caldeira industrial, previsão de energia eólica em smart grids, classificação
de notícias, **filtragem de e-mail e detecção de spam**, análise de sentimento,
sistemas de recomendação e visão computacional em veículos autônomos (DARPA
Stanley).

A seção sobre spam (página 9) é particularmente relevante para este projeto:
cita drift virtual (mudança de conteúdo) e drift real (mudança de
comportamento dos spammers e da própria atitude do usuário).

### 3. Taxonomia em quatro módulos

A contribuição central do survey é apresentar sistemas adaptativos como uma
composição de quatro módulos (Figuras 3 e 4):

- **Memory** — gerenciamento e esquecimento de dados (instância única, janelas
  fixas, janelas adaptativas, fading factors).
- **Change detection** — detectores explícitos de drift baseados em
  monitoramento sequencial.
- **Learning** — modelo preditivo (online, incremental, ensemble, retraining).
- **Loss estimation** — estimativa contínua do erro/performance.

### 4. Detectores de drift discutidos

O survey e seu apêndice apresentam, entre outros:

- **Page-Hinkley (PH)** — teste sequencial baseado em desvio acumulado da média
  do erro.
- **ADWIN (Adaptive Windowing)** — janela adaptativa que detecta mudança via
  comparação de médias entre subjanelas.
- **SPC / DDM** — Statistical Process Control: monitora média e desvio do erro
  e dispara estados *warning* e *out-of-control*.
- **CUSUM / Geometric Moving Average** — soma cumulativa de desvios.
- **EDDM** — variante baseada na distância entre erros consecutivos.
- **Detectores baseados em distribuição** — testes estatísticos sobre $P(X)$
  (e.g., Kolmogorov-Smirnov).

### 5. Estratégias de adaptação ("correção")

Não são chamadas de "corretor" no survey, mas de mecanismos de adaptação:

- Janelas deslizantes de tamanho fixo ou variável (FLORA, FLORA2-4).
- Fading factors / weighting por idade da amostra.
- Retreinamento total ou parcial após detecção (SPC).
- Ensembles dinâmicos: **DWM** (Dynamic Weighted Majority) e **AddExp**, que
  ajustam pesos dos especialistas e adicionam/removem modelos.
- Árvores adaptativas: **CVFDT**.
- Mecanismos para drift recorrente: memória de modelos antigos.

### 6. Avaliação

O survey enfatiza o uso de métricas como **acurácia** e **F1-score** ao longo
do stream (prequential evaluation), com atenção especial a classes
desbalanceadas — ponto também central no projeto, em que F1 é a métrica
principal por causa do desbalanceamento ham/spam.

## Respostas diretas

> **Observação:** As perguntas pressupõem um único experimento, mas o
> documento é um *survey*. As respostas abaixo dão (a) a posição do survey e
> (b) o mapeamento para a implementação concreta do projeto deste workspace
> ([src/](src/)), que usa o survey como referência teórica.

- **Cenário utilizado no documento.**
  No survey, o cenário é genérico: **aprendizado online supervisionado sobre
  data streams** sujeito a concept drift, com diversas aplicações ilustrativas
  (monitoramento, planejamento, assistência pessoal, ambientes ubíquos). Entre
  elas, o survey discute explicitamente **filtragem de e-mail e detecção de
  spam** (Seção 2.5.3) — exatamente o cenário materializado no projeto, que
  implementa **detecção de spam com Aprendizado Federado horizontal (FedAvg)**
  sobre dataset sintético de 20 features ([src/data.py](src/data.py),
  [src/federated.py](src/federated.py), [src/scenarios.py](src/scenarios.py)).

- **Tipo de drift tratado.**
  O survey foca em **real concept drift** e cobre todas as formas temporais:
  súbito, gradual, incremental e recorrente. O projeto implementa **três
  desses tipos** em cenários separados (`src/scenarios.py`):
  - **Drift súbito** — troca abrupta da fase A (spam clássico) para a fase B
    (spam moderno) na rodada `DRIFT_ROUND = 8`.
  - **Drift gradual** — mistura progressiva da fase B controlada por `alpha`.
  - **Drift recorrente** — alternância cíclica entre A e B a cada
    `CYCLE_LEN = 4` rodadas.

- **Detector de drift utilizado.**
  O survey cataloga vários detectores (Page-Hinkley, ADWIN, SPC/DDM, EDDM,
  CUSUM, testes de distribuição). O projeto **não usa um deles diretamente**;
  utiliza um detector composto, configurado em
  [src/config.py](src/config.py) por `DRIFT_DETECTOR_TYPE = "composite"` e
  implementado em [src/drift-detector.py](src/drift-detector.py) como
  **`CompositeDriftDetector`**, que combina três detectores na política `any`
  com `cooldown = 2`:
  1. **`PerformanceDropDetector`** — queda de F1 ≥ `DRIFT_MIN_DROP_PP = 8.0`
     pp (família "loss-based", inspirada em DDM/SPC do survey).
  2. **`FeatureKSTestDetector`** — teste Kolmogorov-Smirnov por feature com
     limiar `DRIFT_KS_THRESHOLD = 0.35` (família "distribution-based").
  3. **`MeanShiftDetector`** — distância euclidiana normalizada entre médias
     com limiar `DRIFT_MEAN_SHIFT_THRESHOLD = 0.18`.

- **Corretor de drift utilizado.**
  No vocabulário do survey, "correção" corresponde às estratégias de
  adaptação (janelas, fading, retreinamento, ensembles). O projeto define
  uma camada explícita de corretores em
  [src/drift-correction.py](src/drift-correction.py). O corretor padrão,
  configurado em `DRIFT_CORRECTOR_TYPE = "severity_adaptive"`, é o
  **`SeverityBasedCorrector`**, que ajusta dinamicamente — em função da
  severidade detectada (baixa / média / alta) — três alavancas de adaptação
  durante um número limitado de rodadas:
  1. **Taxa de aprendizado** (multiplicador `DRIFT_LR_MULTIPLIER = 1.8`).
  2. **Épocas locais extras** (`DRIFT_EXTRA_EPOCHS = 1`).
  3. **Replay recente** de dados (`DRIFT_REPLAY_MEMORY_SIZE = 2`,
     `DRIFT_REPLAY_RATIO = 0.35`).

  Esse corretor combina, na prática, ideias de *fading / windowing* (replay) e
  de *retreinamento intensificado* (LR e épocas) descritas no survey, adaptando
  a intensidade da resposta à magnitude do drift.
