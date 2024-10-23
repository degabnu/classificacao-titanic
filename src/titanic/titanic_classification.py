import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Função para carregar e pré-processar os dados
def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=',', quotechar='"', decimal='.', on_bad_lines='skip', header=0, skipinitialspace=True)
        print(f"Dados carregados de {file_path}:")
        print(df.head())
        print(df.info())
        print(df.describe())

        # Preenchendo valores nulos nas colunas relevantes
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].median())
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        # Convertendo a coluna 'Sex' para numérica (0 = female, 1 = male)
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

        return df

    except pd.errors.EmptyDataError:
        print(f"O arquivo {file_path} está vazio.")
        return None
    except pd.errors.ParserError:
        print(f"Erro ao parsear o arquivo {file_path}.")
        return None
    except Exception as e:
        print(f"Erro inesperado ao carregar o arquivo {file_path}: {e}")
        return None

# Carregar os conjuntos de dados
train_df = load_and_process_data('src/titanic/data/train.csv')
test_df = load_and_process_data('src/titanic/data/test.csv')
gender_submission_df = load_and_process_data('src/titanic/data/gender_submission.csv')

# Verificando se o DataFrame de treino foi carregado corretamente
if train_df is not None:
    # Seleção de variáveis (features) relevantes para o treino
    features = ['Pclass', 'Sex', 'Age', 'Fare']
    X = train_df[features]  # Dados de entrada
    y = train_df['Survived']  # Rótulo (alvo)

    # Divisão dos dados entre treino e validação (70% treino, 30% validação)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Verifique se X_train e y_train não estão vazios
    print("Verificando se existem valores nulos em X_train e y_train:")
    print(X_train.isnull().sum())
    print(y_train.isnull().sum())

    # Treinando o modelo k-NN com k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Fazendo previsões no conjunto de validação
    y_val_pred = knn.predict(X_val)

    # Avaliando a acurácia no conjunto de validação
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f'Acurácia no conjunto de validação: {accuracy_val:.2f}')

    # Matriz de confusão
    cm_val = confusion_matrix(y_val, y_val_pred)
    print('Matriz de Confusão no conjunto de validação:')
    print(cm_val)

# Processando o conjunto de teste se ele foi carregado corretamente
if test_df is not None:
    # Seleção de variáveis (features) para o conjunto de teste
    X_test = test_df[features]

    # Verificando se existem valores nulos em X_test
    print("Verificando se existem valores nulos em X_test:")
    print(X_test.isnull().sum())

    # Fazendo previsões no conjunto de teste
    y_test_pred = knn.predict(X_test)

    # Criando um DataFrame para comparar as previsões com o gender_submission
    submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_test_pred})
    submission_df.to_csv('submission.csv', index=False)

    # Verificando se o arquivo de submissão foi gerado corretamente
    print("Verificando se o arquivo de submissão foi gerado corretamente:")
    print(submission_df.head())
