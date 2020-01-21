"""

@author: Akash

Dataset Source : UCI ML Repository SMS Spam collection DataSet

"""

import pandas as pd

#Data loading
messages = pd.read_csv('sms_spamcollection/SMSSpamCollection',sep='\t',names=['label','message'])

#Data cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

ps = PorterStemmer()
clnd_msgs = []
#wnl = WordNetLemmatizer()
for i in range(len(messages)):
    temp_msg = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    temp_msg = temp_msg.lower()
    temp_msg = temp_msg.split()
    temp_msg = [ps.stem(word) for word in temp_msg if word not in set(stopwords.words('english'))]
#    temp_msg = [wnl.lemmatize(word) for word in temp_msg if word not in set(stopwords.words('english'))]
    temp_msg = ' '.join(temp_msg)
    clnd_msgs.append(temp_msg)

#Creating BagOfWords model
from sklearn.feature_extraction.text import CountVectorizer as CV
cv = CV(max_features = 5000) #selecting random 5k feaures or columns or words
X = cv.fit_transform(clnd_msgs).toarray()

'''
creating TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer as TV
tv = TV(max_features = 5000)#selecting random 5k feaures or columns or words
X = tv.fit_transform(clnd_msgs).toarray()
'''

#Output data
Y = pd.get_dummies(messages['label'])
Y = Y.iloc[:,1].values

#Train Test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Train model with naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB().fit(X_train,Y_train)

#Prediction
y_pred = spam_model.predict(X_test)

#Accuracy
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(Y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,y_pred)
