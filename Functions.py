# Basic Libraries Required
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Sklearn Library Tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
# Preprocessing Tools
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
# NLP Tools
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('stopwords')

# Split data: train 80%, test 20%
def split_data(data,target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=7)
    print('Training Sample:',X_train.shape)
    print('Test Sample:',X_test.shape)
    print('Target Training Sample:',y_train.shape)
    print('Target Test Sample:',y_test.shape)
    return X_train, X_test, y_train, y_test

# Initialise Vectorization, fit and transform Train Data
def vectorize(train,test):
    TFIDF_vec = TfidfVectorizer(stop_words='english',max_df=0.65,strip_accents='unicode')
    fitted_vectorizer = TFIDF_vec.fit(train)
    vec_train = fitted_vectorizer.transform(train)
    vec_test = fitted_vectorizer.transform(test)
    print('Transformed Train set:',vec_train.shape)
    print('Transformed Test set:',vec_test.shape)
    return fitted_vectorizer, vec_train, vec_test

# Initialize Machine Learning Classifier and Fit on Transformed Train Data set
def PAclassify(vectorized_train, target_train):
    PassiveAgg_class = PassiveAggressiveClassifier(early_stopping=False, max_iter=150)
    PassiveAgg_class.fit(vectorized_train, target_train)
    return PassiveAgg_class

# Examine Results in form of confusion matrix
def examine_results(target_test, target_pred_data):
    results = confusion_matrix(target_test, target_pred_data, labels=['FAKE', 'REAL'])
    report = classification_report(target_test, target_pred_data)
    return results,report

# Create Confusion Matrix Graph
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):

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

# Remove any non-alphanumeric characters and punctuation - lowercase and separate words in a body of text
def remove_punct(series,entry):
    text = re.sub('[^a-zA-Z]',' ',series[entry]).lower().split()
    return text

# Stemmatize tokenized Data
def stem(data):
    # Initialize stemmatizer and apply
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in data if not word in stopwords]
    stemmed = ' '.join(stemmed)
    return stemmed

# Preprocess data sequentially
def pre_process(target_data,max_vocab_count):
    # Create localized data copy 
    article = target_data.copy()
    
    corpus = []
    for entry in range(len(article)):
        # Remove punctuation & non-alphanumerics
        tokens = remove_punct(article['compiled'],entry)
        # Stemmatize & Store results
        stemmed_data = stem(tokens)
        corpus.append(stemmed_data)
    # One-Hot Encode & Pad Train Data to maintain dimensionality
    OH_batch = [one_hot(words,max_vocab_count)for words in corpus]
    padded_data = pad_sequences(OH_batch,padding='pre',maxlen=25) 
    print(padded_data.shape)
    return padded_data

def eval_subjectivity(x):
    if 0.0<=x<=0.4:
        y = "Objective"
    elif 0.4<x<=0.6:
        y = "Neutral"
    else:
        y = "Subjective"
    return y

def eval_polarity(x):
    if -1.0<=x<=-0.2:
        y = "Negative"
    elif -0.2<x<=0.2:
        y = "Neutral"
    else:
        y = "Positive"
    return y