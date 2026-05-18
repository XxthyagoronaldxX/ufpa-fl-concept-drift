# Drifts de mudança no cenário de Spam de E-mails.
## 1. Drift Súbito (Transição Abrupta)
O Contexto Real: Eventos Globais ou "Zero-Day Campaigns"
Exemplo: O início da pandemia da COVID-19 (março de 2020) ou o vazamento de dados de um grande banco.
Como explicar: "Do dia para a noite, os spammers abandonaram campanhas antigas (ex: golpes do 'príncipe nigeriano' ou de 'ganhe um iPhone') e viraram a chave para um assunto urgente. Os e-mails mudaram abruptamente para: 'Compre máscaras N95', 'Veja aqui se você tem direito ao Auxílio Emergencial' ou 'Sua conta foi bloqueada no vazamento recente'. O filtro de spam, que estava ótimo em detectar as palavras clássicas de golpe na rodada anterior, acorda no dia seguinte totalmente obsoleto e cego para os novos padrões."

## 2. Drift Gradual (Transição Progressiva)
O Contexto Real: A evolução e o jogo de 'Gato e Rato'
Exemplo: A transição de spams de texto puro para táticas de ofuscação (ex: Spam em Imagens ou troca de caracteres).
Como explicar: "Os spammers percebem que o filtro está aprendendo suas palavras. Aos poucos, uma nova tática começa a ser testada: eles inserem o texto do spam dentro de uma imagem anexada ou começam a trocar letras por números (ex: v1agra ou cr1pto). Durante vários meses (rodadas), as duas táticas (A e B) coexistem na caixa de entrada dos usuários. Progressivamente, a tática antiga vai perdendo força e a nova ofuscação se torna o padrão dominante. O modelo começa a deixar passar falsos negativos aos poucos, como uma 'sangria' na acurácia."

## 3. Drift Recorrente (Alternância Cíclica)
O Contexto Real: Campanhas Sazonais (Sazonalidade)
Exemplo: Black Friday, Natal ou Época de Declaração do Imposto de Renda.
Como explicar: "Este é o comportamento que respeita o calendário do mundo real. Todo ano, entre março e maio, a rede sofre um bombardeio de mensagens de phishing com o tema 'Malha Fina do Imposto de Renda'. Em junho, esse padrão desaparece completamente, e os spams voltam a seguir temas genéricos (Tática A). O grande problema aqui é que um modelo de Machine Learning comum foca apenas nos dados recentes; logo, até chegar o próximo mês de março, ele já 'esqueceu' como é o padrão do Imposto de Renda. O Drift Recorrente prova que o modelo não pode sofrer de amnésia sazonal."

# DATASET
## Classificação Binária (Spam vs. Ham)
Em Machine Learning, "Classificação" é quando o modelo precisa colocar coisas em "caixas" (categorias).

Por que Binária? Porque existem apenas duas opções de saída. O e-mail só pode ser uma de duas coisas:
Spam: É o e-mail não solicitado, lixo eletrônico, fraude ou propaganda invasiva (Geralmente representado pelo número 1).
Ham: É o termo técnico na área de computação para designar o e-mail legítimo e bom, aquele que você realmente quer receber (Geralmente representado pelo número 0).
Resumo para falar: "O objetivo único da nossa rede neural é ler um e-mail e emitir uma resposta simples de sim ou não: isso é lixo (Spam) ou é normal (Ham)?"

## Características Sintéticas (20 features)
Um computador não "lê" texto como nós. Ele precisa que o e-mail seja transformado em números. Cada número que descreve o e-mail é uma feature (característica).

Em um caso real, essas 20 features poderiam ser coisas como:

Feature 1: Quantidade de links no corpo do e-mail.
Feature 2: Frequência da palavra "Grátis".
Feature 3: Se o remetente está na sua lista de contatos.
Feature 4: Proporção de letras MAIÚSCULAS no título.

# Justificativa da recuperação do DRIFT.

O modelo parece se recuperar após o drift porque, mesmo sem tratamento especial, ele continua sendo treinado com os novos dados. Com o tempo, ele aprende o novo padrão, mas de forma lenta e ineficiente. Em aplicações reais, essa adaptação pode ser muito mais difícil e demorada, justificando a necessidade de técnicas específicas para concept drift.

# Justificativa da geração de 500 dados de teste.

Os 500 dados de teste simulam e-mails inéditos, usados apenas para medir a acurácia e o F1-score do modelo. Assim, garantimos que o desempenho apresentado reflete a capacidade real do sistema de detectar spam em situações novas, e não apenas nos exemplos vistos durante o treinamento.

# O que é F1-Score.

O F1-score é uma métrica que avalia o equilíbrio entre precisão (quantos dos e-mails marcados como spam realmente são spam) e revocação (quantos dos spams reais o modelo conseguiu encontrar). Ele é especialmente útil quando as classes estão desbalanceadas, como em detecção de spam.

# Se necessário, explicação das constantes da Config:

- SEED: Garante que os resultados dos experimentos sejam sempre os mesmos, facilitando comparações e depuração.

- NUM_CLIENTS: Quantidade de clientes simulados no aprendizado federado.

- NUM_ROUNDS: Número total de rodadas de comunicação entre clientes e servidor.

- LOCAL_EPOCHS: Quantas vezes cada cliente treina localmente antes de enviar os resultados.

- BATCH_SIZE: Tamanho do lote de dados usado em cada passo de treinamento.

- LEARNING_RATE: Taxa de aprendizado do otimizador (o “passo” dado na atualização dos pesos).

- DRIFT_ROUND: Rodada em que ocorre a mudança de conceito (drift) no experimento.

- CYCLE_LEN: Duração de cada fase no cenário de drift recorrente (quantas rodadas cada padrão permanece ativo).

- N_TRAIN: Quantidade total de amostras de treino (divididas entre os clientes).

- N_TEST: Quantidade de amostras de teste (usadas apenas para avaliação).

- FEATURE_DIM: Número de características (features) de cada e-mail simulado.

- SPAM_RATIO: Proporção de e-mails de spam no dataset.

- DEVICE: Define se o código vai rodar na GPU (se disponível) ou na CPU.

# A diferença principal entre o Grafo de Acurácia e de F1-Score.

Acurácia mostra a porcentagem total de acertos do modelo (quantos e-mails ele classificou corretamente, sejam spam ou ham).
F1-score avalia o equilíbrio entre precisão e revocação, focando na qualidade da detecção de spam (especialmente importante quando as classes estão desbalanceadas). Classes desbalanceadas significam que há muito mais exemplos de uma categoria do que de outra. No caso de spam, é comum ter muito mais e-mails legítimos do que spams. Isso pode enganar a acurácia e dificultar a detecção dos casos raros, como o spam.