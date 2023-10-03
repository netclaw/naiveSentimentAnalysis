import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.utils import shuffle

import string



# ## Read datasets

fake = pd.read_csv("AI_NLP/data/Fake.csv")

true = pd.read_csv("AI_NLP/data/True.csv")



fake.shape

true.shape



# Add flag to track fake and real

fake['target'] = 'fake'

true['target'] = 'true'



# Concatenate dataframes

data = pd.concat([fake, true]).reset_index(drop = True)

data.shape



# Shuffle the data

data = shuffle(data)

data = data.reset_index(drop=True)



# Check the data

data.head()



# Removing the date (we won't use it for the analysis)

data.drop(["date"],axis=1,inplace=True)

data.head()



# Removing the title (we will only use the text)

data.drop(["title"],axis=1,inplace=True)

data.head()



# Convert to lowercase

data['text'] = data['text'].apply(lambda x: x.lower())

data.head()



# Remove punctuation


def punctuation_removal(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    return text


#Insert code



data['text'] = data['text'].apply(punctuation_removal)



# Check

data.head()



# Removing stopwords
from nltk.corpus import stopwords

import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

data['text'] = [word for word in data['text'] if word not in stop_words]

#Insert code



data.head()





# ## Basic data exploration



# How many articles per subject?

print(data.groupby(['subject'])['text'].count())

data.groupby(['subject'])['text'].count().plot(kind="bar")

plt.show()





# In[17]:





# How many fake and real articles?

print(data.groupby(['target'])['text'].count())

data.groupby(['target'])['text'].count().plot(kind="bar")

plt.show()



# Most frequent words counter (Code adapted from https://www.kaggle.com/rodolfoluna/fake-news-detector)   

from nltk import tokenize



token_space = tokenize.WhitespaceTokenizer()



def counter(text, column_text, quantity):

    all_words = ' '.join([text for text in text[column_text]])

    token_phrase = token_space.tokenize(all_words)

    frequency = nltk.FreqDist(token_phrase)

    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),

                                   "Frequency": list(frequency.values())})

    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)

    plt.figure(figsize=(12,8))

    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')

    ax.set(ylabel = "Count")

    plt.xticks(rotation='vertical')

    plt.show()



# Most frequent words in fake news

counter(data[data["target"] == "fake"], "text", 20)



# Most frequent words in real news

counter(data[data["target"] == "true"], "text", 20)



# ### Peparing the data



# Split the data
X = data['text']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Insert code



# ## Modeling



# Function to plot the confusion matrix (code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

from sklearn import metrics

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')










# # **Naive Bayes**



print("------Naive Bayes-----")

dct = dict()



from sklearn.naive_bayes import MultinomialNB



NB_classifier = MultinomialNB()

pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', NB_classifier)])



model = pipe.fit(X_train, y_train)

prediction = model.predict(X_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))



dct['Naive Bayes'] = round(accuracy_score(y_test, prediction)*100,2)



cm = metrics.confusion_matrix(y_test, prediction)

plot_confusion_matrix(cm, classes=['Fake', 'Real'])



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Logistic Regression
print("------Logistic Regression-----")
logreg_classifier = LogisticRegression()
logreg_pipe = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('model', logreg_classifier)])
logreg_model = logreg_pipe.fit(X_train, y_train)
logreg_prediction = logreg_model.predict(X_test)
logreg_accuracy = round(accuracy_score(y_test, logreg_prediction) * 100, 2)
print("Accuracy: {}%".format(logreg_accuracy))
dct['Logistic Regression'] = logreg_accuracy

# Decision Tree
print("------Decision Tree-----")
dt_classifier = DecisionTreeClassifier()
dt_pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('model', dt_classifier)])
dt_model = dt_pipe.fit(X_train, y_train)
dt_prediction = dt_model.predict(X_test)
dt_accuracy = round(accuracy_score(y_test, dt_prediction) * 100, 2)
print("Accuracy: {}%".format(dt_accuracy))
dct['Decision Tree'] = dt_accuracy

# Random Forest
print("------Random Forest-----")
rf_classifier = RandomForestClassifier()
rf_pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('model', rf_classifier)])
rf_model = rf_pipe.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)
rf_accuracy = round(accuracy_score(y_test, rf_prediction) * 100, 2)
print("Accuracy: {}%".format(rf_accuracy))
dct['Random Forest'] = rf_accuracy

# SVM
print("------SVM-----")
svm_classifier = SVC()
svm_pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', svm_classifier)])
svm_model = svm_pipe.fit(X_train, y_train)
svm_prediction = svm_model.predict(X_test)
svm_accuracy = round(accuracy_score(y_test, svm_prediction) * 100, 2)
print("Accuracy: {}%".format(svm_accuracy))
dct['SVM'] = svm_accuracy


# # **Comparing** **Different Models**





import matplotlib.pyplot as plt

plt.figure(figsize=(8,7))

plt.bar(list(dct.keys()),list(dct.values()))

plt.ylim(90,100)

plt.yticks((91, 92, 93, 94, 95, 96, 97, 98, 99, 100))

plt.show()

