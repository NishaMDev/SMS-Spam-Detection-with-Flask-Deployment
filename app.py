from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    model_dir = "models/"
    cv = open(model_dir + 'cv.pkl','rb')
    cv = joblib.load(cv)
    NB_spam_model = open(model_dir + 'NB_spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction, message = message)



if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=False)
