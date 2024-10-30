
# Árvore de Decisão e Avaliação de Desempenho

**Aluno:** Douglas Cristiano da Silva  
**Instituição:** Unisociesc Blumenau - RA 152211260

Este projeto utiliza Python e o algoritmo de Árvore de Decisão para classificar os dados do dataset Iris. O objetivo é treinar um modelo que prevê a espécie de uma flor com base em suas características como comprimento e largura das pétalas e sépalas.

## Instruções de Instalação

Para executar o script, primeiro instale as dependências necessárias:

```bash
pip install scikit-learn pandas numpy matplotlib
```

Em seguida, execute o script Python na raiz do projeto:

```bash
python .\src\arvoredecisao\arvoredecisao.py
```

## Resultados

A avaliação do modelo utilizando diferentes valores de `max_depth` e critérios de divisão trouxe os seguintes resultados:

### Acurácia com `max_depth` Variando:

- **Acurácia (sem especificar max_depth):** 1.0
  - **Matriz de Confusão:**
    ```
    [[19  0  0]  -> classe 0: Setosa
    [ 0 13  0]  -> classe 1: Versicolor
    [ 0  0 13]] -> classe 2: Virginica
    ```
  - **Relatório de Classificação:**
    ```
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00        19
               1       1.00      1.00      1.00        13
               2       1.00      1.00      1.00        13

        accuracy                           1.00        45
       macro avg       1.00      1.00      1.00        45
    weighted avg       1.00      1.00      1.00        45
    ```

- **Acurácia com `max_depth=2`:** 0.9778
  - **Matriz de Confusão:**
    ```
    [[19  0  0]
     [ 0 12  1]
     [ 0  0 13]]
    ```

- **Acurácia com `max_depth=30`:** 1.0
  - **Matriz de Confusão:**
    ```
    [[19  0  0]
     [ 0 13  0]
     [ 0  0 13]]
    ```

- **Acurácia com `max_depth=500`:** 1.0
  - **Matriz de Confusão:**
    ```
    [[19  0  0]
     [ 0 13  0]
     [ 0  0 13]]
    ```

- **Acurácia com `max_depth=10000`:** 1.0
  - **Matriz de Confusão:**
    ```
    [[19  0  0]
     [ 0 13  0]
     [ 0  0 13]]
    ```

### Acurácia com Diferentes Critérios de Divisão:

- **Acurácia com critério `entropy`:** 0.9778
  - **Matriz de Confusão (Entropy):**
    ```
    [[19  0  0]
     [ 0 12  1]
     [ 0  0 13]]
    ```
  - **Relatório de Classificação (Entropy):**
    ```
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00        19
               1       1.00      0.92      0.96        13
               2       0.93      1.00      0.96        13

        accuracy                           0.98        45
       macro avg       0.98      0.97      0.97        45
    weighted avg       0.98      0.98      0.98        45
    ```

## Conclusão

Os testes realizados confirmaram que a profundidade da árvore (max_depth) não teve um impacto significativo na performance do modelo para o dataset utilizado. Quando max_depth foi ajustado para valores muito altos, como 30, 500 e até 10.000, a acurácia permaneceu proxima de 1.0 e a matriz de confusão indicou previsões corretas para todas as classes. 