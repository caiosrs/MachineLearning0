#pip install pandas numpy -U scikit-learn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Carregar os dados do Excel
file_path = r"C:\\Reviant\\Documentos\\Codes\\Python\\MachineLearning\\Evolução Preditiva.xlsx"
data = pd.read_excel(file_path)

# Renomear colunas e garantir formatação
data.rename(columns={'Mês/Ano Numérico': 'Mes_Ano_Numerico'}, inplace=True)
data['Mes_Ano_Numerico'] = pd.to_datetime(data['Mes_Ano_Numerico'], errors='coerce')
data['Mes_Ano'] = data['Mes_Ano_Numerico'].dt.strftime('%d/%m/%Y')

# Recalcular os dados de treinamento com o novo valor adicionado
train_data = data.dropna(subset=['Valores'])
X_train = train_data['Mes_Ano_Numerico'].map(lambda x: (x - train_data['Mes_Ano_Numerico'].min()).days).values.reshape(-1, 1)
y_train = train_data['Valores'].values

# Treinar o modelo novamente com os dados atualizados
model = LinearRegression()
model.fit(X_train, y_train)

# Recalcular os dados de previsão
predict_data = data[data['Valores'].isna()]
X_predict = predict_data['Mes_Ano_Numerico'].map(lambda x: (x - train_data['Mes_Ano_Numerico'].min()).days).values.reshape(-1, 1)
predicted_values = model.predict(X_predict)

# Adicionar as previsões ao DataFrame
data.loc[data['Valores'].isna(), 'Previsão'] = predicted_values
data['Valores'] = data['Valores'].fillna('valor previsto')
data['Previsão'] = data['Previsão'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else '')

# Salvar os resultados em Excel com a formatação desejada
output_path = r"C:\\Reviant\\Documentos\\Codes\\Python\\MachineLearning\\Evolução Preditiva_Resultado.xlsx"
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    data[['Mes_Ano', 'Valores', 'Previsão']].to_excel(writer, index=False, sheet_name='Resultado')

    # Ajustar formatação
    workbook = writer.book
    worksheet = writer.sheets['Resultado']
    number_format = workbook.add_format({'num_format': '#,##0.00', 'align': 'center'})
    text_format = workbook.add_format({'align': 'center'})

    worksheet.set_column('A:A', 12, text_format)  # Mês/Ano
    worksheet.set_column('B:B', 20, text_format)  # Valores
    worksheet.set_column('C:C', 20, number_format)  # Previsão

print(f"Resultados salvos em {output_path}")
