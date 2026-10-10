# Tech Challenge — Fase 2 | POSTECH Data Analytics

> MICHEL FERNANDES DA SILVA
> michelfernandes@bb.com.br
> LUIZ GUSTAVO FERREIRA
> luizgustavoferreira@gmail.com
> JOSE LUIZ DAGOSTINI FILHO
> j.dagostini@bb.com.br
> ALEXANDRE SIMOES DE CARVALHO COSTA
> alexandre.costa.279@bb.com.br

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 2DTATBB |
| Grupo | Grupo 08 |
| Data de entrega | 10/10/2026 |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Michel Fernandes da Silva | RM377742 | michelfernandes@bb.com.br |
| Luiz Gustavo Ferreira | RM377790 | luizgustavoferreira@gmail.com |
| José Luiz D'Agostini | RM377746 | j.dagostini@bb.com.br |
| Alexandre Simões de Carvalho Costa | RM377733 | alexandre.costa.279@bb.com.br |

---

## 2. Links da entrega

Estes três links são **obrigatórios** e devem ser idênticos aos do PDF de submissão.

| Item | Link |
|---|---|
| Repositório | https://github.com/zedagostini/Tech-Challenge-Fase-2-POSTECH-Data-Analytics |
| Vídeo executivo (≤ 5 min) | https://www.youtube.com/watch?v=89olBBzNPR4 |
| Apresentação | https://github.com/zedagostini/Tech-Challenge-Fase-2-POSTECH-Data-Analytics/blob/ef60e35481fda18bd7d1090c48feae2637fb7820/docs/apresentacao_executiva.pdf |

> ⚠️ Repositório privado ou inacessível inviabiliza a avaliação da entrega.
> Confira o acesso em uma janela anônima antes de enviar.

---

## 3. O problema

Instituições que concedem crédito precisam decidir, com base apenas nos dados cadastrais do cliente, quem merece uma análise mais cuidadosa antes (ou logo depois) da concessão. Fazer essa triagem manualmente para toda a carteira é caro e pouco escalável; conceder crédito a quem vai atrasar gera perda financeira.

Este projeto usa Machine Learning para **estimar a probabilidade de um cliente entrar em atraso grave** e, com isso, **priorizar quais cadastros devem ser encaminhados para análise**. O modelo funciona como triador: ele não aprova nem nega crédito sozinho.

### Variável alvo

`ALVO` é uma variável binária por cliente:

- **1** — o cliente teve ao menos um mês com `STATUS` 2, 3, 4 ou 5 (**atraso de 60 dias ou mais**) na janela observada;
- **0** — nenhum mês com atraso de 60+ dias na janela.

Meses com `STATUS` `C` (pago no mês) e `X` (sem empréstimo no mês) **não** contam como atraso.

**Limiar de binarização (60+ dias).** O notebook comparou os cortes de 30+ e 60+ dias. Com histórico mínimo de 6 meses, 30+ dias marca 14,0% dos clientes como inadimplentes e 60+ dias marca 1,8%. O grupo adotou **60+ dias** (atraso grave).

**Janela final.** O modelo final prevê atraso de 60+ dias nos **primeiros 12 meses de histórico de cada cliente**. Foram elegíveis 21.731 clientes (janela de 12 meses completa e não composta só por `X`); destes, **370 (1,70%)** são classe 1. A janela de 6 meses foi testada e tem apenas 0,69% de positivos.

**Distribuição das classes.** A base é muito desbalanceada: cerca de 98% de classe 0 e 2% de classe 1. Um classificador que sempre responde "sem atraso" (`DummyClassifier`) alcança ~97,9% de acurácia e recall zero, por isso a acurácia não é usada para escolher o modelo (ver seção 5).

### Dataset

| Campo | Valor |
|---|---|
| Fonte | Credit Card Approval Prediction (Kaggle): https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction — arquivos `application_record.csv` e `credit_record.csv` |
| Linhas × colunas | `application_record`: 438.557 × 18 (438.510 clientes únicos após remover 47 IDs duplicados). `credit_record`: 1.048.575 linhas × 3 colunas (45.985 clientes) |
| Período / versão | Histórico mensal relativo, de `MONTHS_BALANCE` = −60 (cinco anos atrás) a 0 (mês de referência); até 61 meses por cliente, média de 22,8 |
| Licença de uso | <!-- PREENCHER: confirmar a licença na página do dataset --> |

**Base de modelagem.** Após o tratamento: 45.985 clientes com histórico → 41.449 sem os 4.536 que só têm status `X` → 29.029 com histórico mínimo de 6 meses e cadastro correspondente (base usada nos experimentos de comparação) → 21.731 elegíveis para a janela de 12 meses (modelo final).

Descrição das variáveis:

| Variável | Tipo | Descrição |
|---|---|---|
| `ID` | Inteiro | Identificador do cliente (chave entre as duas tabelas) |
| `CODE_GENDER` | Categórica | Gênero (M/F) |
| `FLAG_OWN_CAR` | Categórica | Possui carro (Y/N) |
| `FLAG_OWN_REALTY` | Categórica | Possui imóvel (Y/N) |
| `CNT_CHILDREN` | Inteiro | Número de filhos |
| `AMT_INCOME_TOTAL` | Numérica | Renda anual total |
| `NAME_INCOME_TYPE` | Categórica | Tipo de renda (5 categorias, ex.: Working, Pensioner) |
| `NAME_EDUCATION_TYPE` | Categórica | Escolaridade (5 categorias) |
| `NAME_FAMILY_STATUS` | Categórica | Estado civil (5 categorias) |
| `NAME_HOUSING_TYPE` | Categórica | Tipo de moradia (6 categorias) |
| `DAYS_BIRTH` | Inteiro | Dias desde o nascimento, contados para trás a partir de hoje (valor negativo); convertido em idade em anos na análise |
| `DAYS_EMPLOYED` | Inteiro | Dias de emprego atual, em valor negativo; `365243` é código para quem não tem emprego (75.329 registros, em geral aposentados) |
| `FLAG_MOBIL` | Binária | Possui celular. Constante (sempre 1), removida do modelo final |
| `FLAG_WORK_PHONE` | Binária | Possui telefone comercial |
| `FLAG_PHONE` | Binária | Possui telefone residencial |
| `FLAG_EMAIL` | Binária | Possui e-mail |
| `OCCUPATION_TYPE` | Categórica | Ocupação (18 categorias); 30,6% ausente, tratado como `APOSENTADO` (aposentados) ou `N_INFORMADO` (demais) |
| `CNT_FAM_MEMBERS` | Numérica | Número de membros da família |
| `MONTHS_BALANCE` | Inteiro | (`credit_record`) Mês do registro, relativo ao mês de referência (0 = atual, −1 = mês anterior…) |
| `STATUS` | Categórica | (`credit_record`) 0: atraso de 1–29 dias; 1: 30–59; 2: 60–89; 3: 90–119; 4: 120–149; 5: mais de 150 dias ou baixa; C: pago no mês; X: sem empréstimo no mês |

---

## 4. Como reproduzir

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Baixe o dataset e coloque os arquivos brutos (`application_record.csv` e `credit_record.csv`) em `data/raw/` (os dados **não** são versionados —
veja `data/README.md`).

Depois execute os notebooks nesta ordem:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória: diagnóstico das duas tabelas, status de pagamento, correlações |
| 2 | `notebooks/02_preprocessamento.ipynb` | Limpeza, escala e feature engineering: remoção de IDs duplicados e constantes, tratamento de ocupação e `DAYS_EMPLOYED`, construção do alvo |
| 3 | `notebooks/03_modelagem.ipynb` | Treino e comparação dos modelos: seis modelos, validação cruzada de 5 folds sem perfis repetidos entre treino e teste |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas, importância de variáveis e conclusões: janelas de 6 e 12 meses, limiar, busca de hiperparâmetros, importância por permutação, modelo final |

**Semente fixa:** `SEMENTE = 42`, declarada na primeira célula de cada notebook.
Rodar os notebooks na ordem acima, a partir de um ambiente limpo, deve reproduzir
exatamente os números da seção 5.

<!-- ATENÇÃO (apagar antes de enviar): o notebook do Colab enviado é um único arquivo (TrabalhoPosTech_Fase2.ipynb) que lê os CSVs de /content/drive/MyDrive/DataScienceFIAP/. Para a tabela acima ser verdadeira, divida-o nos 4 notebooks e troque esse caminho por data/raw/. O nome da constante no código é SEMENTE (não RANDOM_STATE). -->

---

## 5. Resultados

Validação cruzada de 5 folds (`StratifiedGroupKFold`, agrupando por perfil cadastral idêntico para que o mesmo perfil nunca apareça em treino e teste). Métricas da **classe 1 (atraso 60+)**, média dos folds. Base de comparação: 29.029 clientes, histórico mínimo de 6 meses.

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Dummy (sempre classe 0) | 0,979 | 0,000 | 0,000 | 0,000 | 0,500 |
| Regressão Logística (balanced) | 0,627 | 0,022 | 0,389 | 0,041 | 0,521 |
| Árvore de Decisão (balanced) | 0,712 | 0,026 | 0,349 | 0,048 | 0,545 |
| Random Forest (balanced_subsample) | 0,939 | 0,038 | 0,079 | 0,051 | 0,552 |
| KNN (15 vizinhos, distance) | 0,978 | 0,015 | 0,001 | 0,003 | 0,541 |
| **Hist Gradient Boosting (balanced)** | 0,941 | **0,089** | 0,199 | **0,123** | **0,593** |

Modelo final, avaliado na janela de 12 meses (21.731 clientes, 370 positivos), limiar 0,50, média de 5 folds:

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Hist Gradient Boosting — 12 meses | 0,961 | 0,119 | 0,197 | 0,146 | 0,621 |

Average Precision do modelo final: 0,122, contra 0,017 de um palpite aleatório (≈7 vezes). Em contagens somadas dos 5 testes externos: 72 atrasos identificados, 298 não identificados e 540 falsos alarmes (612 clientes sinalizados, 2,8% do total).

> **Cuidado ao comparar protocolos.** Com uma divisão simples por ID, KNN e Random Forest chegam a ROC-AUC de 0,74 e 0,71 porque 87,2% dos clientes do teste têm cadastro idêntico a algum cliente do treino. Ao separar perfis iguais entre treino e teste, esses ganhos desaparecem (ROC-AUC de 0,53 e 0,51); os números acima usam sempre a divisão por perfil.

**Modelo escolhido:** Hist Gradient Boosting (`class_weight` balanceado), janela de 12 meses, limiar 0,50, 16 preditores (`FLAG_MOBIL` removida) — foi o modelo com melhor ROC-AUC, F1, precisão e Average Precision na validação cruzada sem perfis repetidos. A busca de hiperparâmetros testada piorou o F1 (0,146 → 0,078), por isso foram mantidos os parâmetros originais.

**Métricas priorizadas:** Average Precision, precisão e recall da classe 1, F1 e ROC-AUC. Com ~98% de clientes sem atraso, a acurácia é enganosa (o `Dummy` tem 97,9%). O custo dos erros é assimétrico: um falso negativo é um atraso grave que passou sem análise; um falso positivo é uma análise manual desnecessária. O notebook comparou limiares otimizados por F1, F2 e F3: o limiar padrão 0,50 equilibra os dois erros (recall 19,7% analisando 2,8% da carteira), enquanto buscar recall de 54% exigiria analisar 40% da carteira. A escolha final do limiar deve refletir o custo de análise da instituição.

---

## 6. Principais conclusões

1. **O modelo é um triador, não um decisor.** Ele ordena clientes por risco com ganho real sobre o acaso (Average Precision ≈7× maior), mas identifica só cerca de 1 em cada 5 atrasos graves, e cerca de 1 em cada 9 sinalizações é de fato um atraso.
2. **Idade, renda e tempo de emprego são as variáveis que mais pesam.** Na importância por permutação, `DAYS_BIRTH`, `AMT_INCOME_TOTAL` e `DAYS_EMPLOYED` são as três primeiras em quase todos os folds; `OCCUPATION_TYPE`, `NAME_FAMILY_STATUS` e `NAME_INCOME_TYPE` vêm em seguida. Na prática, o perfil de vida e de renda do cliente é o que o modelo usa; variáveis como e-mail, telefone comercial e número de filhos contribuem pouco. A importância mostra quanto o modelo depende da variável, não o sentido do efeito nem causalidade.
3. **Os dados cadastrais carregam sinal fraco.** As correlações entre variáveis são baixas (só filhos × membros da família passa de 0,5, com 0,89, e é redundância) e o ROC-AUC fica em ~0,62. O desempenho aparente de modelos como KNN e Random Forest vinha de perfis repetidos entre treino e teste.
4. **Perfis cadastrais idênticos têm desfechos diferentes.** Dos 8.744 perfis distintos, 248 têm clientes bons e ruins ao mesmo tempo; 63% dos inadimplentes estão nesses perfis. Clientes que atrasam também têm histórico mais longo, o que indica que só o cadastro não separa bem os dois grupos.
5. **Aplicado à carteira inteira,** o modelo sinalizou 14.255 de 438.510 cadastros (3,25%) para análise. Esse número serve para dimensionar o esforço de análise, não como medida de desempenho, pois inclui clientes usados no treino.

### Limitações e próximos passos

- Só variáveis cadastrais foram usadas; o histórico de pagamento serviu apenas para construir o alvo. Incluir variáveis comportamentais e de relacionamento deve ser o maior ganho esperado.
- Poucos positivos (370 em 21.731) e variação grande entre folds (desvio padrão do recall de 0,08): as métricas têm incerteza considerável.
- Parte dos perfis cadastrais é ambígua (mesmo cadastro, desfechos diferentes), o que limita o teto de qualquer modelo baseado só nesses campos.
- O limiar de 0,50 não foi calibrado com custos reais; definir o custo de uma análise e de um atraso permitiria escolher o limiar de forma otimizada.
- A busca de hiperparâmetros foi pequena (8 configurações); explorar um espaço maior e outras famílias de modelos pode ser testado.
- Monitorar o modelo em produção (deriva dos dados e das taxas de atraso) e reavaliar periodicamente.

---

## 7. Estrutura do repositório

```
.
├── data/          dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/     análise em ordem numerada
└── docs/          apresentação executiva
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

Python 3, Google Colab, pandas, NumPy, scikit-learn (pipelines, `HistGradientBoostingClassifier`, `StratifiedGroupKFold`, importância por permutação), matplotlib, seaborn, joblib, Jupyter Notebook. <!-- PREENCHER: versões exatas (o notebook não as imprime; rode `pip freeze` no Colab e copie para requirements.txt) -->
