import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

news=pd.read_csv('news.csv')
print(news.shape)
print(news.head())

X=news['text']
y=news['label']
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=7)
tfidf=TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train_tfidf,y_train)
y_pred=pac.predict(X_test_tfidf)
print('score=', round(accuracy_score( y_test, y_pred)*100,3))
print(confusion_matrix( y_test, y_pred, labels=['FAKE','REAL']))