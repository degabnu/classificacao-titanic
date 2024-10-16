# Análise e Classificação de Dados do Titanic

**Aluno:** Douglas Cristiano da Silva  
**Instituição:** Unisociesc Blumenau - RA 152211260

Este projeto utiliza Python e o algoritmo k-Nearest Neighbors (k-NN) para realizar a classificação de sobreviventes no dataset do Titanic. O objetivo é treinar um modelo que prevê se um passageiro sobreviveu ao desastre, com base em atributos como classe, idade, sexo e tarifa paga.

## Instruções de Instalação

Para executar o script, primeiro instale as dependências necessárias:

```bash
pip install scikit-learn pandas
```

Em seguida, execute o script python:
```bash
python .\titanic.py
```

## Carregamento e Pré-processamento dos Dados

1. **Carregamento dos Dados:** Utilizamos a biblioteca `pandas` para carregar e explorar o dataset.
   - Comandos como `df.head()`, `df.info()` e `df.describe()` foram usados para obter uma visão geral dos dados e de seus tipos.
   
2. **Limpeza dos Dados:**
   - Identificamos valores ausentes usando `df.isnull().sum()`. 
   - A coluna `Age` continha valores nulos que foram preenchidos com a média de idades. Da mesma forma, a coluna `Fare` teve valores ausentes preenchidos com sua média.
   - Outras colunas não essenciais para esta análise foram removidas para simplificar o conjunto de dados.

3. **Seleção de Variáveis e Codificação:**
   - Selecionamos as variáveis `Pclass`, `Sex`, `Age`, `Fare` e `Survived`.
   - Convertendo a variável categórica `Sex` em numérica usando `pd.get_dummies()` para facilitar a análise.

## Implementação do Algoritmo k-NN

- Utilizamos o algoritmo k-NN da biblioteca `scikit-learn` com k=3 para classificar os dados.
- Dividimos o conjunto de dados em dados de treino e teste, reservando 70% para treino e 30% para teste.
- O modelo foi treinado com os dados de treino (`X_train` e `y_train`) e, em seguida, fez previsões para os dados de teste (`X_test`).

## Avaliação de Desempenho

1. **Acurácia:**
   - A acurácia do modelo foi de aproximadamente 69%, indicando que o modelo classificou corretamente 69% dos casos no conjunto de teste.

2. **Matriz de Confusão:**
   - A matriz de confusão mostra as classificações corretas e incorretas:
     ```
     [[131  26]
      [ 58  53]]
     ```
   - Estes resultados indicam que o modelo fez previsões corretas para 131 sobreviventes e 53 não sobreviventes, enquanto houve 26 e 58 previsões incorretas, respectivamente.

## Conclusão

Este projeto demonstrou uma análise básica e a implementação do k-NN para classificação de sobreviventes no Titanic. Confesso que tive muita dificuldade no processamento dos dados, principalmente com a questão de NaN, pra ser sincero, ainda não consegui resolver 100%.

