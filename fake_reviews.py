# IA para classificar a saúde de um feto
# Dados obitidos com o chatGPT - Prompt: "consegue gerar um arquivo cvc de dados fakes para estudar isso?\n Gere com 200 dados contendo essas colunas\n ProdID, Rating, Text"

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import pandas

df = pandas.read_csv("./datasets/fake_reviews.csv")

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Text"])
y = df["Rating"]

ia_model = LogisticRegression()
ia_model.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=200)
ia_model.fit(X_train, y_train)

y_pred = ia_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

classificacao = {
    1: "Totalmente Insatisfeito",
    2: "Instatisfeito",
    3: "Indiferente",
    4: "Satisfeito",
    5: "Super Satisfeito",
}

res = ia_model.predict(
    vectorizer.transform([
        "Não gostei", #1
        "horrivel", #2
        "Super recomendo", #3
        "gostei não", #4
        "gostei", #5
        "gostei demais", #6
        "Ótimo produto", #7
        "Ótimo notebook", #8
        "ótimo", #9
        "recomendo", #10
        "não gostei", #11
        "produto com problema", #12
        "jamais comprem isso", #13
        "não comprem essa porcaria", #14
        "comprei, e comprarei novamente" #15
    ])
)

# Testando a classificação com base no que passei
a = 1
for r in res:
    print(f"Classificou o {a}º resultado, como: {classificacao[int(r)]}")
    a += 1