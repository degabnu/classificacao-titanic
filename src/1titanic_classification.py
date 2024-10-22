import pandas as pd

# Define file paths
csv_files = ['src/data/gender_submission.csv', 'src/data/test.csv', 'src/data/train.csv']

for file_path in csv_files:
    try:
        # Read CSV with specific conditions for each file
        if file_path == "src/data/gender_submission.csv":
            df = pd.read_csv(file_path, sep=',', quotechar='"', engine='python', 
                             on_bad_lines='skip', encoding='utf-8').dropna(subset=['Survived'])  # Drop rows with NaN in Survived
        elif file_path == "src/data/test.csv":
            # Read only PassengerId and Pclass
            df = pd.read_csv(file_path, sep=',', quotechar='"', engine='python', 
                             on_bad_lines='skip', encoding='utf-8', usecols=['PassengerId', 'Pclass'])
            
            # Display data after processing
            print(f"Dados tratados de {file_path} com PassengerId e Pclass:")
            print(df)

    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
