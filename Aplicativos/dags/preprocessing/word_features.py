def word_count(x):
    lista = word_tokenize(x)
    return len(lista)

def word_count_unique(x):
    lista = set(word_tokenize(x))
    return len(lista)

def sentence_count(x):
    lista = sent_tokenize(x)
    return len(lista)

def generate_word_sentence_features(df,column = 'new_text'):
    df['word_count'] = df[column].apply(lambda x: word_count(x))
    df['unique_word_count'] = df[column].apply(lambda x: word_count_unique(x))
    df['sentence_count'] = df[column].apply(lambda x: sentence_count(x))
    
    return df