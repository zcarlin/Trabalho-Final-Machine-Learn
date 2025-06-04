# Trabalho-Final-Machine-Learn
Trabalho Final 
# Análise de Safra com Aprendizado por Reforço

Este projeto utiliza técnicas de aprendizado por reforço para analisar e prever a produtividade de safra agrícola com base em variáveis climáticas e outros fatores.

## Funcionalidades Principais

1. **Análise de Dados**
   - Leitura e processamento de dados do arquivo `DataSet.csv`
   - Classificação das safras em três categorias: baixa, média e alta
   - Análise de variáveis climáticas (chuva, temperatura, etc.)

2. **Modelos de Classificação**
   - Implementação de dois modelos:
     - Modelo clássico com variáveis originais
     - Modelo com Análise de Componentes Principais (PCA)
   - Uso do OneVsRestClassifier com LogisticRegression

3. **Aprendizado por Reforço**
   - Implementação de algoritmo Q-learning
   - Tabela de Q-values para tomada de decisões
   - Visualização do histórico das decisões

## Estrutura do Projeto

- `aty.py`: Arquivo principal contendo todo o código do projeto
- `DataSet.csv`: Arquivo de dados com informações das safras

## Requisitos

- Python 3.x
- Bibliotecas necessárias:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - IPython

## Como Executar

1. Certifique-se de ter todas as dependências instaladas:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install IPython
```

2. Execute o arquivo principal:
```bash
python aty.py
```

## Métricas de Avaliação

O projeto inclui várias métricas de avaliação:
- Matriz de Confusão
- Curva ROC multiclasse
- Acurácia
- F1-Score
- Relatório de classificação completo

## Visualizações

O projeto inclui várias visualizações para ajudar na análise:
- Gráficos de dispersão
- Curvas ROC
- Matrizes de confusão
- Tabela de Q-values
