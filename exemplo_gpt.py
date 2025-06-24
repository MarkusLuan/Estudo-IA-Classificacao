from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Plot não tinha no GPT
import matplotlib.pyplot as plt

# GPT: Carrega o dataset Iris (flores)
iris = load_iris()
X, y = iris.data, iris.target

# GPT: Divide em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GPT: Cria e treina o modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# GPT: Faz previsões e avalia
y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

# Coloquei para a IA usar dados reais para testar
resultados = model.predict([
    [4.9, 3.1, 1.5, 0.1],
    [5.1, 3.5, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [5.8, 2.7, 5.1, 1.9],
    [5.9, 3.0, 5.1, 1.8],
    [7.0, 3.2, 4.7, 1.4]
])

# Testando a classificação com base no que passei
a = 0
for resultado in resultados:
    print(f"Classificou o resultado {a}, como: {iris.target_names[resultado]}")
    a += 1

# Plotando gráficos
_, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])

_ = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names,
    loc="lower right",
    title="Especie da Flor Irís"
)

plt.show()