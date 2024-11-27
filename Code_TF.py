import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split

# Caminho do arquivo CSV
csv_path = r"Notas para Code_TF.csv"

# Leitura dos dados
dados = pd.read_csv(csv_path, header=None)

nota1 = dados[0].values
nota2 = dados[1].values
resultado = dados[2].values

notas = np.column_stack((nota1, nota2))

scaler = MinMaxScaler()
notas_normalizadas = scaler.fit_transform(notas)

X_train, X_test, y_train, y_test = train_test_split(notas_normalizadas, resultado, test_size=0.2, random_state=42)

#Leaky ReLU e Dropout
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, input_shape=[2]),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=32),
    LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=16),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

print("Treinando o modelo...")
modelo.fit(X_train, y_train, epochs=1000, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])

_, accuracy = modelo.evaluate(X_test, y_test)
print(f"Acurácia od teste: {accuracy:.2f}")

while True:
    nota1_input = input("\nDigite a primeira nota (ou 'q' para sair): ")
    if nota1_input.lower() == 'q': 
        print("Encerrando o teste.")
        break
    try:
        nota1 = float(nota1_input)
        nota2_input = input("Digite a segunda nota: ")
        nota2 = float(nota2_input)

        entrada = np.array([[nota1, nota2]])
        entrada_normalizada = scaler.transform(entrada)

        # Fazer a previsão
        predicao = modelo.predict(entrada_normalizada)
        status = "Passou" if predicao[0][0] >= 0.5 else "Não passou"
        print(f"\nNotas: [{nota1}, {nota2}], Previsão: {predicao[0][0]:.2f}, Status: {status}")
    except ValueError:
        print("Entrada inválida. Por favor, digite números válidos para as notas.")
