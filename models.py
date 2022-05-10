from dataclasses import dataclass
from random import sample
import pandas as pd 
import numpy as np
import spacy 
import tensorflow as tf 
nlp = spacy.load("en_core_web_sm")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam , SGD

def gen_training_data(data): 
    sample_text = data['text'][4]
    label = data[["outcome"]]
    dataset = data.drop(['text', 'outcome'],  axis=1)

    #encode categorical data: persontype and alignment 
    le = LabelEncoder()
    dataset['persontype'] = le.fit_transform(dataset[['persontype']].astype(str))
    dataset['alignment'] = le.fit_transform(dataset[['alignment']].astype(str))

    #normalize numerical data
    min_max_scaler = MinMaxScaler()
    normalized = ["word_count", "sentence_count", "avg_sentence_length", "flesch", "smog", "successful_trigrams", "successful_quadgrams", "successful_pentagrams", "max_connections", "avg_connections_word", "avg_connections_sent"]
    dataset[normalized] = min_max_scaler.fit_transform(dataset[normalized])
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20)
    return x_train, x_test, y_train, y_test

def naive_bayes(dataset): 
    x_train, x_test, y_train, y_test = gen_training_data(dataset)
    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()

    print("GaussianNB: ")
    model = GaussianNB()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

    print("MultinomialNB: ")
    model = MultinomialNB()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

def svm(dataset): 
    x_train, x_test, y_train, y_test = gen_training_data(dataset)
    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()

    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

def FCNN(dataset):
    X_train, X_test, y_train, y_test = gen_training_data(dataset)
    X_train = np.asarray(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='sigmoid'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.fit(X_train, y_train, epochs=20, batch_size=200, verbose=2)
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.6f' %accuracy)
    print('Precision: %.6f' %precision)
    print('Recall: %.6f' %recall)


def main(): 
    final_dataset = pd.read_csv ('features.csv')
    #print("Naive Bayes: ")
    #naive_bayes(final_dataset)
    #print()
    #print("SVM: ")
    #svm(final_dataset)
    #print()
    print("FCNN: ")
    FCNN(final_dataset)
    #print()

if __name__ == "__main__":
   main()