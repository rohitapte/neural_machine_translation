from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from string import punctuation

#just read the file contents into an list
#we use this to load the source and target text
def read_data_from_file(filename):
    text=[]
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            text.append(line.rstrip())
    return text

#convert sentence to wordids and pad
def sents2sequences(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):
    encoded_text=tokenizer.texts_to_sequences(sentences)
    preproc_text=pad_sequences(encoded_text, padding=padding_type, maxlen=pad_length)
    if reverse:
        preproc_text = np.flip(preproc_text, axis=1)
    return preproc_text

#load data from files, add start and end token to target text
def get_data(source_file,target_file):
    source_lang_text_data=read_data_from_file(source_file)
    target_lang_text_data=read_data_from_file(target_file)
    source_lang_text_data=[item.rstrip() for item in source_lang_text_data]
    target_lang_text_data=[item.rstrip() for item in target_lang_text_data]
    target_lang_text_data=['sentencestart '+sent[:-1]+' sentenceend .' if sent.endswith('.') else 'sentencestart '+sent+' sentenceend .' for sent in target_lang_text_data]
    print("Length of text {}".format(len(source_lang_text_data)))
    return source_lang_text_data,target_lang_text_data

#load data from file, tokenize words and fit on texts
#do train_test_split
def build_tokenizer_and_split_text(source_file="data/europarl-v7.fr-en_small.fr",target_file="data/europarl-v7.fr-en_small.en",src_min_words=1,tgt_min_words=1):
    source_lang_text_data,target_lang_text_data=get_data(source_file,target_file)

    split_condition = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    source_words = Counter()
    for sentence in source_lang_text_data:
        words=[word for word in split_condition.findall(sentence.lower()) if word not in punctuation]
        for word in words:
            source_words[word] += 1
    target_words = Counter()
    for sentence in target_lang_text_data:
        words = [word for word in split_condition.findall(sentence.lower()) if word not in punctuation]
        for word in words:
            target_words[word] += 1
    print("Total unique words in source lang: "+str(len(source_words)))
    print("Total unique words in target lang: "+str(len(target_words)))
    source_vocab_count=0
    for word in source_words:
        if source_words[word]>=src_min_words:source_vocab_count+=1
    target_vocab_count=0
    for word in target_words:
        if target_words[word]>=tgt_min_words:target_vocab_count+=1
    print("Total unique source words with min count "+str(src_min_words)+': '+str(source_vocab_count))
    print("Total unique target words with min count "+str(tgt_min_words)+': '+str(target_vocab_count))

    source_tokenizer=keras.preprocessing.text.Tokenizer(num_words=source_vocab_count+1,oov_token='UNK')
    source_tokenizer.fit_on_texts(source_lang_text_data)
    target_tokenizer=keras.preprocessing.text.Tokenizer(num_words=target_vocab_count+1,oov_token='UNK')
    target_tokenizer.fit_on_texts(target_lang_text_data)
    src_train, src_test, tgt_train, tgt_test = train_test_split(source_lang_text_data, target_lang_text_data, test_size=0.1)
    return src_train,src_test,tgt_train,tgt_test,source_tokenizer,target_tokenizer

if __name__ == '__main__':
    src_train, src_test, tgt_train, tgt_test, source_tokenizer, target_tokenizer = build_tokenizer_and_split_text(source_file="../data/europarl-v7.fr-en_small.fr",target_file="../data/europarl-v7.fr-en_small.en")
    print(len(src_train))