import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

train_data = pd.read_csv('./data/train.csv')

print(train_data.head())
print(train_data.info())
print(train_data.describe())

print("Valores nulos antes do tratamento:")
print(train_data.isnull().sum())

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())

train_data = train_data.dropna(subset=['Pclass', 'Sex', 'Survived'])

data = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

X = data[['Pclass', 'Age', 'Fare', 'Sex_male']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)
