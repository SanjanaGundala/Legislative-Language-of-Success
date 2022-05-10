import pandas as pd 
import string 
import nltk
from nltk.util import ngrams
import spacy 
nlp = spacy.load("en_core_web_sm")
from textstat.textstat import textstatistics,legacy_round
from collections import Counter
import csv

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


def generate_ngrams_with_NER(text, num): 
    exclist = string.punctuation
    table_ = str.maketrans('', '', exclist)
    #majorly slows down process 
    doc=nlp(text)
    newtext = text
    for e in reversed(doc.ents): 
        start = e.start_char
        end = start + len(e.text)
        newtext = newtext[:start] + e.label_ + newtext[end:]

    no_punc_text = newtext.translate(table_)
    tokens = [token.lower() for token in no_punc_text.split()]
    NGRAMS = list(ngrams(tokens, num))
    return NGRAMS

def POS_TAGGING(text): 
    exclist = string.punctuation
    table_ = str.maketrans('', '', exclist)
    text.lower()
    no_punc_text = text.translate(table_)
    tokens = nltk.word_tokenize(no_punc_text)
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text)
    return tagged 


def generate_POS_ngrams(text, num):
    tags = POS_TAGGING(text)
    pos_tags = [x[1] for x in tags]
    flattened_sent = ' '.join(pos_tags)  
    tokens = [token for token in flattened_sent.split()]
    NGRAMS = list(ngrams(tokens, num))
    return NGRAMS

def generate_successful_phrases(dataset, num): 
    all_ngrams = []
    ngrams_dict = {}

    for_bill_pass ={}
    against_bill_fail = {}  

    alignment_for = {}
    alignment_against = {}

    for index, row in dataset.iterrows():
        text = row['Text']
        ayes = row['Ayes']
        naes = row['Naes']
        alignment = str(row['Alignment'])

        NGRAMS = generate_POS_ngrams(text, num)
        all_ngrams = all_ngrams + NGRAMS
        
        for entry in NGRAMS: 
            entry = tuple(entry)
            if "For" in alignment or "Indeterminate" in alignment or "Neutral" in alignment:
                if entry in alignment_for:  
                    alignment_for[entry] = alignment_for[entry] + 1
                else: 
                    alignment_for[entry] = 1
                if ayes > naes: 
                    if entry in for_bill_pass:  
                        for_bill_pass[entry] = for_bill_pass[entry] + 1
                    else: 
                        for_bill_pass[entry] = 1
            if "Against" in alignment or "Indeterminate" in alignment or "Neutral" in alignment:
                if entry in alignment_against:  
                    alignment_against[entry] = alignment_against[entry] + 1
                else: 
                    alignment_against[entry] = 1
                if ayes < naes: 
                    if entry in against_bill_fail:  
                        against_bill_fail[entry] = against_bill_fail[entry] + 1
                    else: 
                        against_bill_fail[entry] = 1
            else: 
                for_bill_pass[entry] = 0
                against_bill_fail[entry] = 0  
                alignment_for[entry] = 1
                alignment_against[entry] = 1

    for ngram in all_ngrams: 
        tuple_ngram = tuple(ngram)
        if tuple_ngram in ngrams_dict: 
            ngrams_dict[tuple_ngram] += 1
        else: 
            ngrams_dict[tuple_ngram] = 1
    ngrams_dict = {key:val for key, val in ngrams_dict.items() if val > 10}
    
    pos_success_rate = {}
    neg_success_rate = {}
    for ngram in ngrams_dict.keys(): 
        if ngram in for_bill_pass: 
            pos_success_rate[ngram] = for_bill_pass[ngram]/alignment_for[ngram]
        else: 
            pos_success_rate[ngram] = 0
        if ngram in against_bill_fail: 
            neg_success_rate[ngram] = against_bill_fail[ngram]/alignment_against[ngram]
        else: 
            neg_success_rate[ngram] = 0
    pos_success_rate = {key:val for key, val in pos_success_rate.items() if val > 0.49} 
    neg_success_rate = {key:val for key, val in neg_success_rate.items() if val > 0.49}

    #remove phrases which are both positively and negatively successful
    pos_success_rate = {key:val for key, val in pos_success_rate.items() if key not in neg_success_rate.keys()} 
    neg_success_rate = {key:val for key, val in neg_success_rate.items() if key not in pos_success_rate.keys()}

    print(pos_success_rate)
    print(neg_success_rate)
    '''
    with open('pos_successfulphrases5.csv', 'w') as f:
        writer = csv.writer(f)
        for row in pos_success_rate.items():
            writer.writerow(row)
    with open('neg_successfulphrases5.csv', 'w') as f:
        writer = csv.writer(f)
        for row in neg_success_rate.items():
            writer.writerow(row)
    '''
    pos_successful_phrases = pos_success_rate.keys()
    neg_successful_phrases = neg_success_rate.keys()
    return pos_successful_phrases, neg_successful_phrases

def count_successful_phrases(text, alignment,pos_successful_phrases,neg_successful_phrases):    
    if "For" in alignment or "Indeterminate" in alignment or "Neutral" in alignment:
        if len(pos_successful_phrases) != 0:
            num = len(list(pos_successful_phrases)[0])
            text_ngrams = generate_ngrams_with_NER(text, num)  
            res = len(set(text_ngrams).intersection(pos_successful_phrases))
            return res
        else: 
            return 0 
    if "Against" in alignment or "Indeterminate" in alignment or "Neutral" in alignment: 
        if len(neg_successful_phrases) != 0:
            num = len(list(neg_successful_phrases)[0])
            text_ngrams = generate_ngrams_with_NER(text, num)  
            res = len(set(text_ngrams).intersection(neg_successful_phrases))
            return res
        else: 
            return 0 
    else: 
        return 0 

def break_sentences(text):
    doc = nlp(text)
    return list(doc.sents)
 
# Returns Number of Words in the text
def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words
 
# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)
 
# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length


def syllables_count(word):
    word = str(word)
    try: 
        return textstatistics().syllable_count(word)
    except TypeError: 
        print(word, type(word))
        return 1

# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)
 
# Return total Difficult Words in a text
def difficult_words(text):
    # Find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()
     
    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)
 
    return len(diff_words_set)
 
def poly_syllable_count(text):
    count = 0
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]
     
    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count
 
def flesch_reading_ease(text):
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) -\
          float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)
 
def smog_index(text):
    if sentence_count(text) >= 3:
        poly_syllab = poly_syllable_count(text)
        num_sentences = sentence_count(text)
        avg_poly_syllab = poly_syllab/num_sentences

        if num_sentences < 30: 
            short_30 = 30 - num_sentences
            SMOG = 3 + ((poly_syllab + (avg_poly_syllab*short_30))**0.5)
        else: 
            SMOG = 1.0430 * ((30 * poly_syllab / num_sentences)**0.5) + 3.1291
        
        return legacy_round(SMOG, 2)
    else:
        return 0

def count_pos(text): 
    tagged = POS_TAGGING(text)
    counts = Counter(tag for word,tag in tagged)
    total = sum(counts.values())
    props = dict((word, float(count)/total) for word,count in counts.items())
    pos_tags ={}
    wanted_tags = ['DT', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    for tag in wanted_tags: 
        if tag in props: 
            pos_tags[tag] = props[tag]
        else: 
            pos_tags[tag] = 0
    pos_tags['JJ'] = pos_tags['JJ'] + pos_tags['JJR'] + pos_tags['JJS']
    pos_tags.pop('JJR', None)
    pos_tags.pop('JJS', None)

    pos_tags['NN'] = pos_tags['NN'] + pos_tags['NNS'] + pos_tags['NNP'] + pos_tags['NNPS']
    pos_tags.pop('NNS', None)
    pos_tags.pop('NNP', None)
    pos_tags.pop('NNPS', None)

    pos_tags['PRP'] = pos_tags['PRP'] + pos_tags['PRP$']
    pos_tags.pop('PRP$', None)

    pos_tags['VB'] = pos_tags['VB'] + pos_tags['VBD'] + pos_tags['VBG'] + pos_tags['VBN'] + pos_tags['VBP'] + pos_tags['VBZ']
    pos_tags.pop('VBD', None)
    pos_tags.pop('VBG', None)
    pos_tags.pop('VBN', None)
    pos_tags.pop('VBP', None)
    pos_tags.pop('VBZ', None)
    return pos_tags

def count_dependencies(text): 
    max = 0
    total = 0
    sentences = break_sentences(text)
    num_sents = len(sentences)
    num_words = word_count(text)
    for sentence in sentences:
        for token in sentence: 
            num_connections = len([child for child in token.children])
            if num_connections > max: 
                max = num_connections
            total = total + num_connections
    avg_per_word = total/num_words
    avg_per_sent = total/num_sents
    res = {'max_connections': max,'avg_connections_word': avg_per_word, 'avg_connections_sent': avg_per_sent}
    return res

def extract_features(dataset,pos_successful_trigrams,neg_successful_trigrams, pos_successful_quadgrams, neg_successful_quadgrams, pos_successful_pentagrams, neg_successful_pentagrams): 
    features = pd.DataFrame(columns = ['text', 'persontype','alignment' ,'word_count', 'sentence_count', 'avg_sentence_length', 'flesch', 'smog', 'successful_trigrams', 'successful_quadgrams', 'successful_pentagrams','DT', 'JJ', 'NN', 'PRP', 'VB','max_connections','avg_connections_word', 'avg_connections_sent','outcome'])
    all_texts = dataset[["Text", "PersonType", "Alignment", 'Ayes','Naes','Abstains']]
    all_texts = all_texts.drop_duplicates()

    count = 1
    for index, row in all_texts.iterrows():
        speech = row['Text']
        alignment = str(row['Alignment'])

        outcome = 0
        ayes = row['Ayes']
        naes = row['Naes']
        if (ayes > naes and ("For" in alignment or alignment == "Indeterminate" or alignment == "Neutral")) or (ayes < naes and ("Against" in alignment or alignment == "Indeterminate" or alignment == "Neutral")):
            outcome = 1

        dict1 = {'text': row['Text'], 'persontype': row['PersonType'], 'alignment': row['Alignment'], 'word_count': word_count(speech), 'sentence_count': sentence_count(speech), 'avg_sentence_length': avg_sentence_length(speech),'flesch' : flesch_reading_ease(speech), 'smog': smog_index(speech)}
        dict2 = {'successful_trigrams': count_successful_phrases(speech, alignment,pos_successful_trigrams,neg_successful_trigrams), 'successful_quadgrams': count_successful_phrases(speech, alignment,pos_successful_quadgrams, neg_successful_quadgrams),'successful_pentagrams': count_successful_phrases(speech, alignment,pos_successful_pentagrams, neg_successful_pentagrams)}
        dict3 = count_dependencies(speech)
        dict4 = count_pos(speech)
        dict5 = {'outcome': outcome}
        dict6 = {**dict1, **dict2, **dict3, **dict4, **dict5}
        features = features.append(dict6, ignore_index = True)
        print(count)
        count = count + 1
    return features

def main(): 
    dataset = readFile("CA20172018_alignments.tsv")
    filtered_dataset = filter_data(dataset)
    print(filtered_dataset)
    pos_successful_trigrams, neg_successful_trigrams = generate_successful_phrases(filtered_dataset, 3)
    #pos_successful_quadgrams, neg_successful_quadgrams = generate_successful_phrases(filtered_dataset, 4)
    #pos_successful_pentagrams, neg_successful_pentagrams = generate_successful_phrases(filtered_dataset, 5)
    #features = extract_features(filtered_dataset,pos_successful_trigrams,neg_successful_trigrams, pos_successful_quadgrams, neg_successful_quadgrams, pos_successful_pentagrams, neg_successful_pentagrams)
    #print(features)
    #features.to_csv('features.csv', index=False)    
    
if __name__ == "__main__":
   main()