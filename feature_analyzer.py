import pandas as pd 
import spacy 
nlp = spacy.load("en_core_web_sm")

def avgs(df): 
    #[['DT','JJ','NN', 'PRP', 'VB']]
    aligns = df[df['outcome'] == 1]
    not_aligns = df[df['outcome'] == 0]
    aligns_avg = aligns.mean()
    not_aligns_avg = not_aligns.mean()
    print("ALIGNS WITH BILL AVGS: ")
    print(aligns_avg)
    print("DOES NOT ALIGN WITH BILL AVGS: ")
    print(not_aligns_avg)


def main(): 
    final_dataset = pd.read_csv ('features.csv')
    avgs(final_dataset)
    

if __name__ == "__main__":
   main()