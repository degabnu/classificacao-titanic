from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Carregar o conjunto de dados de câncer de mama
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Criar um DataFrame para facilitar a análise
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

# Verificar a quantidade de amostras benignas e malignas
print("Quantidade de amostras:")
malignos = (y == 0).sum()
benignos = (y == 1).sum()
print(f"Malignas: {malignos}")
print(f"Benignas: {benignos}")

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Criar e treinar o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {accuracy:.2f}")

# Imprimir o relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['maligno', 'benigno']))

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Imprimir a matriz de confusão no terminal
print("\nMatriz de Confusão:")
print(cm)

# Imprimir a matriz de confusão com rótulos
print("\nMatriz de Confusão com Rótulos:")
cm_labeled = pd.DataFrame(cm, index=['maligno', 'benigno'], columns=['maligno', 'benigno'])
print(cm_labeled)

# # Plotar a matriz de confusão
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=['maligno', 'benigno'], 
#             yticklabels=['maligno', 'benigno'])
# plt.xlabel('Previsão')
# plt.ylabel('Valor Real')
# plt.title('Matriz de Confusão')
# plt.show()