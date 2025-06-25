# IA para classificar a saúde de um feto
# Dados obitidos em: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas

df = pandas.read_csv("./datasets/fetal_health.csv")
X = df.drop("fetal_health", axis=1)
X.columns = range(X.shape[1])
y = df["fetal_health"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

ia_model = RandomForestClassifier(random_state=6)
ia_model.fit(X_train, y_train)

y_pred = ia_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

res = ia_model.predict([
    [132.0,0.006,0.0,0.006,0.003,0.0,0.0,17.0,2.1,0.0,10.4,130.0,68.0,198.0,6.0,1.0,141.0,136.0,140.0,12.0,0.0],
    [134.0,0.001,0.0,0.01,0.009,0.0,0.002,26.0,5.9,0.0,0.0,150.0,50.0,200.0,5.0,3.0,76.0,107.0,107.0,170.0,0.0],
    [140.0,0.001,0.0,0.006,0.0,0.0,0.0,78.0,0.4,27.0,7.0,66.0,103.0,169.0,6.0,0.0,152.0,147.0,151.0,4.0,1.0],
    [142.0,0.002,0.002,0.008,0.0,0.0,0.0,74.0,0.4,36.0,5.0,42.0,117.0,159.0,2.0,1.0,145.0,143.0,145.0,1.0,0.0],
    [128.0,0.0,0.0,0.008,0.01,0.0,0.0,63.0,4.2,0.0,0.0,90.0,66.0,156.0,5.0,0.0,69.0,73.0,118.0,128.0,0.0]
])

classificacao = {
    1: "Normal",
    2: "Suspeito",
    3: "Doente",
}

# Testando a classificação com base no que passei
a = 1
for r in res:
    print(f"Classificou o {a}º resultado, como: {classificacao[int(r)]}")
    a += 1