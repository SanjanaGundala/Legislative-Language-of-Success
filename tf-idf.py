import pandas as pd 
import spacy 
nlp = spacy.load("en_core_web_sm")

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

def readFile(csv_name):
    df = pd.read_csv(csv_name, sep='\t')
    df.columns =['FirstName', 'LastName', 'PID', 'PersonType', 'Organization', 'BID', 'DID', 'Speech Type', 'Alignment','Date','VID','Ayes','Naes','Abstains','MID','motionText','doPass','Text']
    df = df.drop_duplicates()
    return df 

def filter_data(dataset):
    dataset = dataset.drop_duplicates(subset=dataset.columns.difference(['Organization', 'PersonType']))
    zero_ayes = dataset["Ayes"] != 0
    zero_naes = dataset["Naes"] != 0
    filtered_dataset = dataset[zero_ayes & zero_naes]
    options = ['Lobbyist', 'General Public'] 
    filtered_dataset = filtered_dataset[filtered_dataset['PersonType'].isin(options)]

    filtered_dataset = filtered_dataset.groupby(['FirstName', 'LastName','Organization' ,'PID', 'PersonType', 'BID', 'DID', 'Speech Type', 'Alignment','Date','VID','Ayes','Naes','Abstains','MID','motionText','doPass'])['Text'].apply(', '.join).reset_index()

    filtered_dataset.to_csv('filtered_dataset.csv', index=False)
    return filtered_dataset



def gen_input_tfidf(dataset): 
    all_texts = dataset[["Text", "PersonType", "Alignment", 'Ayes','Naes','Abstains']]
    all_texts = all_texts.drop_duplicates()
    speeches = all_texts['Text'].tolist()
    outcome_list = []
    for index, row in all_texts.iterrows():
        alignment = str(row['Alignment'])
        outcome = 0
        ayes = row['Ayes']
        naes = row['Naes']
        if (ayes > naes and ("For" in alignment or alignment == "Indeterminate" or alignment == "Neutral")) or (ayes < naes and ("Against" in alignment or alignment == "Indeterminate" or alignment == "Neutral")):
            outcome = 1
        outcome_list.append(outcome)    
    return speeches, outcome_list

def TF_IDF_classifier(dataset): 
    speeches, label = gen_input_tfidf(dataset)
    X_train, X_test, y_train, y_test = train_test_split(speeches, label, test_size=0.20)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tf = vectorizer.fit_transform(X_train)
    #print(X_train_tf[0])
    #print(speeches[0])
    X_train_tf = X_train_tf.toarray()
    X_test_tf = vectorizer.transform(X_test)
    X_test_tf = X_test_tf.toarray()

    print("Multinomial Naive Bayes: ")
    model1 = MultinomialNB()
    model1.fit(X_train_tf,y_train)
    y_pred1 = model1.predict(X_test_tf)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
    print("Precision:",metrics.precision_score(y_test, y_pred1))
    print("Recall:",metrics.recall_score(y_test, y_pred1))

    print("Guassian Naive Bayes: ")
    model2 = GaussianNB()
    model2.fit(X_train_tf,y_train)
    y_pred2 = model2.predict(X_test_tf)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
    print("Precision:",metrics.precision_score(y_test, y_pred2))
    print("Recall:",metrics.recall_score(y_test, y_pred2))


def main(): 
    dataset = readFile("CA20172018_alignments.tsv")
    filtered_dataset = filter_data(dataset)
    TF_IDF_classifier(filtered_dataset)
    
if __name__ == "__main__":
   main()