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
df['diagnóstico'] = y

# Mapear os valores do alvo para 'maligno' e 'benigno'
df['diagnóstico'] = df['diagnóstico'].map({0: 'maligno', 1: 'benigno'})

# Mostrar as 10 primeiras linhas
print("10 primeiras linhas do conjunto de dados:")
print(df.head(10))


# Verificar a quantidade de amostras benignas e malignas
print("\nQuantidade de amostras:")
print(df['diagnóstico'].value_counts())

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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
