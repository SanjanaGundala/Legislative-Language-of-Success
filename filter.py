import pandas as pd 

def readFile(csv_name):
    df = pd.read_csv(csv_name, sep='\t')
    df.columns =['FirstName', 'LastName', 'PID', 'PersonType', 'Organization', 'BID', 'DID', 'Speech Type', 'Alignment','Date','VID','Ayes','Naes','Abstains','MID','motionText','doPass','Text']
    print(df)
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

    filtered_dataset.to_csv('filtered_dataset_10k.csv', index=False)
    return filtered_dataset

def check_alignments(dataset): 
    df = dataset[['Alignment','Text']]
    df = df.sample(n = 63)
    df.to_csv('alignment_check.csv', index=False)

def main(): 
    dataset = readFile("CA20172018_alignments.tsv")
    filtered_dataset = filter_data(dataset)
    #check_alignments(filtered_dataset)
    
    
if __name__ == "__main__":
   main()