# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data  
y = iris.target  

# treino (70%) teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) #, random_state=42

# profundidade máxima de 3
model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(X_train, y_train)

# Visualizar arvore de decisão
plt.figure(figsize=(12,8))
tree.plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

# previsões no conjunto de teste
y_pred = model.predict(X_test)

# acurácia, matriz de confusão e classificação
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# testar com diferentes valores de 'max_depth'

for depth in [2, 30, 500, 10000]:
    model = DecisionTreeClassifier(criterion='gini', max_depth=depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nAcurácia com max_depth={depth}: {accuracy_score(y_test, y_pred)}")
    print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# testar com 'entropy' na divisão
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)

print("\nAcurácia com critério 'entropy':", accuracy_score(y_test, y_pred_entropy))
print("Matriz de Confusão (Entropy):\n", confusion_matrix(y_test, y_pred_entropy))
print("Relatório de Classificação (Entropy):\n", classification_report(y_test, y_pred_entropy))
