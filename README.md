# Sistema de Análise de Safra com Aprendizado por Reforço

## Descrição
Este projeto implementa um sistema completo de análise de dados agrícolas que combina:
- Análise exploratória de dados climáticos
- Modelos de Machine Learning (regressão e classificação)
- Sistema de aprendizado por reforço (Q-learning) para decisões de irrigação

## Estrutura do Código Reorganizado

### Funções Principais

#### 1. Carregamento e Preparação de Dados
- `carregar_dados()`: Carrega o dataset e renomeia colunas
- `criar_variaveis_adicionais()`: Cria variáveis derivadas (chuva relativa, anomalia binária, etc.)

#### 2. Análise Exploratória
- `bloxplot()`: Visualiza produtividade vs eventos ENSO
- `scatterplot()`: Temperatura vs produtividade
- `HistogramaDeVariaveisNumericas()`: Distribuições das variáveis
- `HeatmapDeCorrelacao()`: Matriz de correlação
- `pairplot()`: Matriz de dispersão entre variáveis

#### 3. Modelos de Regressão
- `preparar_dados_regressao()`: Prepara dados para regressão
- `screeplot()`: Análise de componentes principais (PCA)
- `treinar_modelos_regressao()`: Treina modelos Linear e Ridge
- `visualizar_comparacao_lambda()`: Compara RMSE vs Lambda
- `plot_funcao_custo_1D()`, `plot_funcao_custo_2D()`: Visualiza funções de custo
- `visualizar_residuos()`: Análise de resíduos

#### 4. Modelos de Classificação
- `preparar_dados_classificacao()`: Prepara dados para classificação
- `treinar_modelos_classificacao()`: Treina modelos de classificação
- `avaliar_modelos_classificacao()`: Avalia desempenho com métricas
- `visualizar_fronteiras_decisao()`: Visualiza fronteiras de decisão

#### 5. Aprendizado por Reforço
- `aprendizado_por_reforco()`: Implementa Q-learning para irrigação
  - Estados: muito_seco, seco, ideal, encharcado
  - Ações: muita_agua, regar, pouca_agua, nao_regar
  - Modos: padrão ou extremo (diferentes tabelas de recompensa)

#### 6. Menu Principal
- `menu_principal()`: Interface interativa para acessar todas as funcionalidades
- `executar_pipeline_completo()`: Executa todas as análises em sequência

## Como Usar

1. Execute o arquivo:
   ```
   py aty_organizado.py
   ```

2. Use o menu interativo para:
   - Carregar e preparar dados (opção 1)
   - Executar análises exploratórias (opção 2)
   - Treinar e avaliar modelos de regressão (opção 3)
   - Treinar e avaliar modelos de classificação (opção 4)
   - Executar aprendizado por reforço (opção 5)
   - Visualizar resultados (opção 6)
   - Executar pipeline completo (opção 7)

## Requisitos
- Python 3.x
- Bibliotecas: pandas, numpy, matplotlib, seaborn, scikit-learn

## Arquivos Necessários
- `DataSet.csv`: Arquivo de dados com informações climáticas e de produtividade
- `aty_organizado.py`: Código principal reorganizado

## Melhorias Implementadas
- Código totalmente modularizado em funções
- Menu interativo para facilitar o uso
- Possibilidade de executar análises individuais ou pipeline completo
- Estrutura clara e organizada
