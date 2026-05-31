# Documentação da Aplicação

## Objetivo

Esta aplicação simula um sistema de detecção de spam em e-mails usando Aprendizado Federado (Federated Learning, ou FL) em um cenário com concept drift. O objetivo é classificar cada e-mail como:

- `ham`: e-mail legítimo.
- `spam`: e-mail indesejado, fraudulento ou malicioso.

O projeto demonstra como mudanças no comportamento dos spammers afetam o desempenho do modelo ao longo das rodadas federadas, e compara o resultado com e sem mecanismos de detecção e correção de drift.

## Arquitetura Geral

A aplicação é composta por quatro partes principais:

- Geração de dados sintéticos de e-mail em `src/data.py`.
- Modelo neural MLP para classificação binária em `src/model.py`.
- Treinamento federado com FedAvg em `src/federated.py`.
- Detecção e correção de concept drift em `src/drift-detector.py` e `src/drift-correction.py`.

O fluxo geral é:

1. O sistema gera datasets sintéticos representando e-mails legítimos e spam.
2. Os dados de treino são divididos entre clientes federados.
3. Cada cliente treina localmente uma cópia do modelo global.
4. O servidor agrega os pesos dos clientes usando FedAvg.
5. O modelo é avaliado a cada rodada.
6. Os detectores analisam métricas e distribuição das features.
7. Quando drift é detectado, os corretores ajustam temporariamente o treinamento.

## Dataset Sintético de Spam

O dataset possui 20 features numéricas por e-mail. Elas simulam características que um filtro de spam real poderia extrair, como frequência de palavras suspeitas, número de links, quantidade de exclamações, uso de HTML e reputação do remetente.

As classes são:

- `0`: ham.
- `1`: spam.

A proporção padrão é definida por `SPAM_RATIO = 0.4`, ou seja, 40% das amostras são spam.

As features mais importantes para representar o drift são divididas em dois grupos:

- Spam clássico, fase A: `word_free`, `word_win`, `word_prize`, `word_click`, `word_offer`.
- Spam moderno, fase B: `word_crypto`, `word_investment`, `word_profit`, `word_urgent`, `word_verify`.

Na fase A, os spams abusam de palavras típicas de golpes antigos, como ofertas, prêmios e chamadas para clicar. Na fase B, os spams passam a usar padrões mais modernos, como phishing, criptoativos e verificação de conta. Essa mudança de padrão cria o concept drift.

## Cenários de Concept Drift

A aplicação executa quatro cenários principais.

### FL Padrão

No cenário padrão, não há drift. A distribuição dos dados permanece na fase A durante todas as rodadas. Ele serve como linha de base para medir o desempenho do FL em uma situação estacionária.

### Drift Súbito

No drift súbito, a distribuição muda abruptamente a partir de `DRIFT_ROUND = 8`. Antes dessa rodada, os dados seguem a fase A. A partir dela, os spams passam para a fase B.

Esse cenário representa mudanças repentinas, como uma nova campanha global de phishing ou um golpe explorando um evento recente.

### Drift Gradual

No drift gradual, a fase B entra aos poucos depois da rodada de drift. O parâmetro `alpha` aumenta progressivamente, misturando spam clássico e moderno até chegar a 100% de fase B.

Esse cenário simula a evolução natural dos spammers, em que novas técnicas aparecem devagar e coexistem com técnicas antigas por algum tempo.

### Drift Recorrente

No drift recorrente, a distribuição alterna entre fase A e fase B em ciclos definidos por `CYCLE_LEN = 4`.

Esse cenário representa sazonalidade, como golpes que voltam em períodos específicos do ano, por exemplo Black Friday, Natal ou imposto de renda.

## Aprendizado Federado Utilizado

O projeto usa Federated Learning com agregação FedAvg.

Em cada rodada:

1. O servidor mantém um modelo global.
2. Cada cliente recebe uma cópia do modelo global.
3. Cada cliente treina localmente com seus próprios dados.
4. Cada cliente envia apenas os pesos atualizados e a quantidade de amostras.
5. O servidor calcula uma média ponderada dos pesos usando `fed_avg`.
6. O modelo global é atualizado com os pesos agregados.

O treino local é feito pela função `local_train`, usando SGD com momentum, `CrossEntropyLoss` e batches definidos por `BATCH_SIZE`.

## FL Horizontal ou Vertical?

O FL utilizado nesta aplicação é **Federated Learning horizontal**.

Ele é horizontal porque todos os clientes possuem o mesmo espaço de features e o mesmo tipo de label. Em outras palavras, cada cliente trabalha com e-mails descritos pelas mesmas 20 features:

- frequência de palavras suspeitas;
- número de links;
- tamanho do e-mail;
- proporção de letras maiúsculas;
- reputação do remetente;
- demais atributos definidos em `FEATURE_NAMES`.

A diferença entre os clientes está nas amostras. Cada cliente recebe um subconjunto diferente de e-mails, mas todos usam o mesmo formato de entrada e a mesma tarefa de classificação spam/ham.

Isso é diferente de FL vertical. Em FL vertical, diferentes instituições teriam features diferentes sobre as mesmas entidades. Por exemplo, em um cenário vertical, um provedor de e-mail poderia ter features textuais do e-mail, enquanto outra instituição teria informações de reputação do remetente para os mesmos usuários ou mensagens. Esse não é o caso aqui: neste projeto, todos os clientes têm o mesmo tipo de dado e treinam a mesma tarefa com amostras diferentes.

Portanto, no cenário de spam desta aplicação, o uso correto é FL horizontal.

## Modelo de Machine Learning

O modelo usado é `SpamMLP`, definido em `src/model.py`.

Ele é uma rede neural MLP com:

- entrada de dimensão `FEATURE_DIM = 20`;
- duas camadas ocultas;
- ativações ReLU;
- dropout para regularização;
- saída de dimensão 2 para classificação binária.

A avaliação usa duas métricas:

- Acurácia: percentual total de classificações corretas.
- F1-score: equilíbrio entre precisão e revocação para a classe spam.

O F1-score é especialmente importante porque, em detecção de spam, não basta acertar muitos e-mails legítimos. O modelo também precisa capturar bem os spams.

## Detectores de Drift

Os detectores estão implementados em `src/drift-detector.py`. Todos seguem a interface `BaseDriftDetector` e retornam um `DriftDetectionResult`.

O resultado contém:

- `detected`: indica se houve drift.
- `round_id`: rodada em que o detector foi executado.
- `detector_name`: nome do detector.
- `severity`: severidade estimada, como `baixa`, `média` ou `alta`.
- `score`: pontuação calculada pelo detector.
- `message`: explicação textual usada no log.

O detector padrão configurado é:

```python
DRIFT_DETECTOR_TYPE = "composite"
```

Ou seja, a aplicação usa uma combinação de detectores.

### PerformanceDropDetector

O `PerformanceDropDetector` detecta drift observando queda de desempenho.

Ele compara:

- uma janela de referência inicial;
- uma janela recente de métricas.

Por padrão, ele acompanha o F1-score. Se a média recente cair pelo menos `DRIFT_MIN_DROP_PP = 8.0` pontos percentuais em relação à referência, o detector sinaliza drift.

Esse detector é útil porque concept drift normalmente aparece como perda de desempenho: o modelo aprendeu o conceito antigo, mas passa a errar quando o conceito muda.

Pontos fortes:

- simples de interpretar;
- diretamente ligado ao impacto no modelo;
- bom para detectar queda real de qualidade.

Limitações:

- depende de labels ou de uma métrica de avaliação confiável;
- pode detectar o problema depois que a performance já caiu.

### FeatureKSTestDetector

O `FeatureKSTestDetector` compara a distribuição das features atuais com a distribuição de referência usando a estatística Kolmogorov-Smirnov.

Ele calcula o KS para cada feature e usa o maior valor encontrado. Se o score passar de `DRIFT_KS_THRESHOLD = 0.35`, o detector sinaliza drift.

Quando `scipy` está disponível, o código usa `scipy.stats.ks_2samp`. Quando não está, usa uma implementação fallback baseada em CDF empírica.

Esse detector é importante porque consegue perceber mudança na distribuição dos dados mesmo antes de o impacto aparecer fortemente na acurácia ou no F1-score.

Pontos fortes:

- detecta mudança estatística nas features;
- não depende diretamente da queda de desempenho;
- é útil para identificar drift de covariáveis.

Limitações:

- pode sinalizar mudança de distribuição que nem sempre prejudica o modelo;
- usa o maior KS entre features, podendo ser sensível a features isoladas.

### MeanShiftDetector

O `MeanShiftDetector` mede o deslocamento entre a média das features atuais e a média das features de referência.

O score é calculado como uma distância euclidiana normalizada entre os vetores médios. Se passar de `DRIFT_MEAN_SHIFT_THRESHOLD = 0.18`, o detector sinaliza drift.

Esse detector captura mudanças globais no centro da distribuição. No cenário de spam, ele ajuda a perceber quando as features associadas ao padrão moderno passam a ficar mais frequentes.

Pontos fortes:

- computacionalmente simples;
- bom para mudanças globais de distribuição;
- fácil de explicar.

Limitações:

- pode não detectar mudanças que preservam a média;
- é menos sensível a mudanças localizadas ou multimodais.

### CompositeDriftDetector

O `CompositeDriftDetector` combina múltiplos detectores.

Na configuração atual, ele usa:

- `PerformanceDropDetector`;
- `FeatureKSTestDetector`;
- `MeanShiftDetector`.

A política padrão é:

```python
DRIFT_DETECTOR_POLICY = "any"
```

Com essa política, basta um detector sinalizar drift para o detector composto sinalizar drift. Também existe suporte à política `majority`, em que a maioria dos detectores precisa concordar.

O detector composto também usa cooldown:

```python
DRIFT_DETECTOR_COOLDOWN = 2
```

Isso evita disparos repetidos em rodadas imediatamente consecutivas.

## Corretores de Drift

Os corretores estão implementados em `src/drift-correction.py`. Todos seguem a interface `BaseDriftCorrector` e retornam um `CorrectionState`.

O estado de correção contém:

- `active`: indica se a correção está ativa;
- `learning_rate`: taxa de aprendizado a ser usada;
- `local_epochs`: número de épocas locais;
- `replay_ratio`: proporção de replay usada;
- `remaining_rounds`: quantas rodadas de correção ainda restam;
- `message`: texto exibido no log.

O corretor padrão configurado é:

```python
DRIFT_CORRECTOR_TYPE = "severity_adaptive"
```

Ou seja, a aplicação usa uma correção baseada na severidade detectada.

### AdaptiveLearningRateCorrector

O `AdaptiveLearningRateCorrector` aumenta temporariamente a taxa de aprendizado após a detecção de drift.

A ideia é permitir que o modelo se adapte mais rápido ao novo padrão. O multiplicador padrão é:

```python
DRIFT_LR_MULTIPLIER = 1.8
```

Pontos fortes:

- simples;
- acelera a adaptação;
- não muda a arquitetura do modelo.

Risco:

- learning rate alto demais pode causar instabilidade.

### AdaptiveEpochCorrector

O `AdaptiveEpochCorrector` aumenta temporariamente o número de épocas locais.

O valor padrão é:

```python
DRIFT_EXTRA_EPOCHS = 1
```

Isso significa que, durante a correção, cada cliente treina por mais épocas antes de enviar seus pesos ao servidor.

Pontos fortes:

- melhora adaptação usando os dados locais atuais;
- simples de implementar.

Risco:

- aumenta custo computacional nos clientes;
- pode aumentar overfitting local se usado por muitas rodadas.

### RecentReplayCorrector

O `RecentReplayCorrector` mantém uma memória curta de datasets recentes dos clientes e mistura uma parte desses dados com os dados atuais.

Os parâmetros relevantes são:

```python
DRIFT_REPLAY_MEMORY_SIZE = 2
DRIFT_REPLAY_RATIO = 0.35
```

O replay ajuda a reduzir esquecimento abrupto e suaviza a transição entre conceitos.

Pontos fortes:

- ajuda em drift recorrente;
- preserva informação recente;
- reduz adaptação excessivamente brusca.

Risco:

- se a memória contiver dados do conceito antigo em excesso, pode atrasar a adaptação ao conceito novo.

### SeverityBasedCorrector

O `SeverityBasedCorrector` é o corretor principal da aplicação.

Ele ajusta automaticamente:

- taxa de aprendizado;
- número de épocas locais;
- proporção de replay.

A decisão depende da severidade do drift:

- drift baixo: aumento leve de learning rate e replay reduzido;
- drift médio: usa `DRIFT_LR_MULTIPLIER`, épocas extras e replay padrão;
- drift alto: aumenta mais agressivamente learning rate, épocas e replay.

Esse corretor é o mais completo porque combina as ideias dos outros corretores e adapta a resposta ao tamanho da mudança detectada.

## Fluxo de Execução

O arquivo principal é `src/main.py`.

Ele executa:

- `FL Padrão`;
- `Súbito sem correção`;
- `Súbito com correção`;
- `Gradual sem correção`;
- `Gradual com correção`;
- `Recorrente sem correção`;
- `Recorrente com correção`.

Cada cenário retorna históricos de acurácia e F1-score. Esses históricos são usados para imprimir um resumo final e gerar gráficos.

Comando via npm:

```bash
npm start
```

Comando direto pelo ambiente virtual Windows do projeto:

```bash
venv\Scripts\python.exe src\main.py
```

Em ambiente Linux/WSL com dependências instaladas:

```bash
python3 src/main.py
```

## Gráficos Gerados

A aplicação gera três imagens:

- `fl_spam_drift_results.png`: gráfico combinado com todos os cenários.
- `fl_spam_drift_sem_correcao.png`: gráfico apenas com FL padrão e cenários sem correção.
- `fl_spam_drift_com_correcao.png`: gráfico apenas com FL padrão e cenários com correção.

Cada gráfico contém dois painéis:

- acurácia por rodada;
- F1-score por rodada.

A linha vertical marca a rodada de início do drift, definida por `DRIFT_ROUND = 8`.

## Interpretação dos Resultados

Nos cenários sem correção, o modelo ainda pode se recuperar com o tempo, porque continua recebendo dados novos nas rodadas seguintes. No entanto, essa recuperação tende a ser mais lenta, principalmente após drift súbito ou recorrente.

Nos cenários com correção, a aplicação responde ao drift aumentando temporariamente a capacidade de adaptação do treinamento. Isso normalmente reduz o tempo de recuperação ou melhora a estabilidade depois da mudança.

O F1-score é a métrica mais importante para observar o impacto do drift em spam. Uma queda de F1 indica que o modelo está deixando de identificar spams corretamente ou está confundindo e-mails legítimos com spam.

## Limitações

Esta aplicação é uma simulação didática. Algumas simplificações importantes são:

- o dataset é sintético;
- os clientes são divididos de forma IID;
- a avaliação usa labels disponíveis em todas as rodadas;
- o detector KS compara features do dataset de teste da rodada;
- não há privacidade diferencial nem criptografia segura;
- o servidor recebe pesos dos clientes, mas não há simulação de ataques ou falhas de comunicação.

Apesar disso, a aplicação é útil para demonstrar os principais conceitos de drift em FL: mudança de distribuição, queda de performance, detecção estatística, correção adaptativa e comparação entre cenários com e sem tratamento.

## Possíveis Extensões

Algumas melhorias possíveis são:

- simular clientes não IID;
- adicionar drift parcial por cliente;
- usar datasets reais de spam;
- implementar detectores online clássicos como DDM, EDDM ou ADWIN;
- criar memória de conceitos para drift recorrente;
- comparar diferentes políticas de agregação federada;
- adicionar métricas como precisão, revocação e matriz de confusão;
- avaliar custo computacional das correções.
