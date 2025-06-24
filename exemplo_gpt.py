from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carrega o dataset Iris (flores)
X, y = load_iris(return_X_y=True)

# Divide em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina o modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Faz previsões e avalia
y_pred = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))