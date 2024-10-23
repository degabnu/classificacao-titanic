import csv
import pandas as pd

# Define file path
file_path = 'src/data/test.csv'

try:
    # Read the CSV file using the csv module
    with open(file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        
        # Read the header
        header = next(reader)
        
        # Read the remaining rows
        data = [row for row in reader]
        
    # Create a DataFrame from the read data
    df = pd.DataFrame(data, columns=header)

    # Select only the PassengerId and Pclass columns
    df = df[['PassengerId', 'Pclass']]

    # Display the structure of the read data
    print("\nEstrutura dos dados lidos:")
    print(df.head())

except Exception as e:
    print(f"Erro ao ler {file_path}: {e}")
