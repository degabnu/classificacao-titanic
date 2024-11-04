import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data = load_wine()
X = data.data
y = data.target
print("Características do Dataset:", data.feature_names)
print("Classes:", data.target_names)
print("\nExemplo de Dados:\n", pd.DataFrame(X, columns=data.feature_names).head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}


results = {}
for model_name, model in models.items():

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nModelo: {model_name}")
    print("Acurácia:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Matriz de Confusão:\n", conf_matrix)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=data.target_names))
    
    results[model_name] = {
        "Acurácia": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

print("\nComparação dos Modelos:")
comparison_df = pd.DataFrame(results).T
print(comparison_df)

best_model = comparison_df['Acurácia'].idxmax()
print("\nConclusão:")
print(f"O modelo com melhor desempenho foi o {best_model}, com uma acurácia de {comparison_df.loc[best_model, 'Acurácia']:.2f}.")
print("A precisão, recall e F1-score também foram analisados para verificar a consistência do desempenho do modelo.")
