import os
import pickle
import pandas as pd
from textblob import TextBlob
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from sklearn.linear_model import LinearRegression

colunas = ['tamanho', 'ano', 'garagem']
modelo = pickle.load(open('models/modelo.sav', 'rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return 'Minha primeira API'

@app.route('/sentiment/<frase>')
@basic_auth.required
def sentiment(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    return "Polaridade: {}".format(tb_en.sentiment.polarity)


#def linear_regression():
    # import dataset
    # df = pd.read_csv('casas.csv')

    # select columns
    # colunas = ['tamanho', 'ano', 'garagem']
    # df = df[colunas]

    # split data into explanatory and response variables (X and y)
    # X = df.drop('preco', axis=1)
    # y = df['preco']

    # split data into train and test 
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # create linear regression
    # modelo = LinearRegression()

    # fit using the data
    # modelo.fit(X_train, y_train)

    # return modelo, colunas

@app.route('/regression/', methods=['POST'])
def regression():
    # modelo, colunas = linear_regression()

    dados = request.get_json()
    dados_input = [dados[col] for col in colunas] 

    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')