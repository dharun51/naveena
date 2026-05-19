import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
file_name="naveen.xlsx"
df=pd.read_excel(file_name)
print(df)
X=df['Question']
y=df['Answer']
model=Pipeline([('tfidf',TfidfVectorizer()),('classifier',MultinomialNB())])
model.fit(X, y)
joblib.dump(model, 'model,naveen(1)_model.pkl')
print("Model Saved Successfully!")