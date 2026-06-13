3. O cenário de Concept Drift
Concept drift é quando a relação entre features e target muda ao longo do tempo. No projeto, ele é simulado trocando a estação do ano:

Estação	Meses	Características físicas
Verão (JJA)	Jun, Jul, Ago	Vento mais fraco, ar menos denso → menor geração
Inverno (DJF)	Dez, Jan, Fev	Vento mais forte, ar mais denso → maior geração

A função f: features → P, que o modelo aprendeu no verão não vale mais no inverno, mesmo com features parecidas. O scenarios.py simula 4 padrões dessa mudança:

Standard: sem drift (baseline) — só verão.
Recurrent: alterna verão/inverno a cada 4 rodadas (sazonalidade cíclica).