import pandas as pd
import numpy as np
import spacy 
import scipy
from scipy import stats
nlp = spacy.load("en_core_web_sm")

def avgs(df): 
    
    aligns = df[df['outcome'] == 1]
    not_aligns = df[df['outcome'] == 0]
    aligns_avg = aligns.mean()
    not_aligns_avg = not_aligns.mean()
    print("ALIGNS WITH BILL AVGS: ")
    print(aligns_avg)
    print("DOES NOT ALIGN WITH BILL AVGS: ")
    print(not_aligns_avg)


def p_value(df): 
    columns = ['word_count', 'sentence_count', 'avg_sentence_length', 'flesch', 'smog', 'successful_trigrams', 'successful_quadgrams', 'successful_pentagrams','DT', 'JJ', 'NN', 'PRP', 'VB','max_connections','avg_connections_word', 'avg_connections_sent','outcome']
    aligns = df[df['outcome'] == 1]
    not_aligns = df[df['outcome'] == 0]
    
    for column in columns: 
        print(column)  
        #print(stats.ttest_ind(aligns[column], not_aligns[column]))
        x = np.array(aligns[column])
        y = np.array(not_aligns[column])

        f = np.var(x, ddof=1)/np.var(y, ddof=1)
        nun = x.size-1
        dun = y.size-1
        p_value = 1-scipy.stats.f.cdf(f, nun, dun)
        print(p_value)
  
    
def main(): 
    final_dataset = pd.read_csv ('features.csv')
    p_value(final_dataset)
    #to_remove = np.random.choice(final_dataset[final_dataset['outcome']==1].index,size=5660,replace=False)
    #final_dataset = final_dataset.drop(to_remove)
    #p_value(final_dataset)
    #avgs(final_dataset)
    

if __name__ == "__main__":
   main()