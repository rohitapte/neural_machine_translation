from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras
from sklearn.model_selection import train_test_split

#just read the file contents into an list
def read_data_from_file(filename):
    text=[]
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            text.append(line.rstrip())
    return text

def sents2sequences(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):
    encoded_text=tokenizer.texts_to_sequences(sentences)
    #preproc_text=pad_sequences(encoded_text, padding=padding_type, maxlen=pad_length)
    preproc_text=encoded_text
    if reverse:
        preproc_text = np.flip(preproc_text, axis=1)
    return preproc_text

#convert sentence to sequences for each language, return it
def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
    en_seq=sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='post', pad_length=en_timesteps)
    fr_seq=sents2sequences(fr_tokenizer, fr_text, pad_length=fr_timesteps)
    print('Vocabulary size (English): {}'.format(np.max(en_seq)+1))
    print('Vocabulary size (French): {}'.format(np.max(fr_seq)+1))
    print('En text shape: {}'.format(en_seq.shape))
    print('Fr text shape: {}'.format(fr_seq.shape))

    return en_seq, fr_seq

def get_data():
    source_lang_text_data=read_data_from_file("data/europarl-v7.fr-en_small.fr")
    target_lang_text_data=read_data_from_file("data/europarl-v7.fr-en_small.en")
    source_lang_text_data=[item.strip() for item in source_lang_text_data]
    target_lang_text_data=['<START> '+item.strip()+' <END>' for item in target_lang_text_data]
    print("Length of text {}".format(len(source_lang_text_data)))
    return source_lang_text_data,target_lang_text_data

def tokenize_sentences_and_split(source_timesteps,target_timesteps):
    source_lang_text_data,target_lang_text_data=get_data()
    source_tokenizer=keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    source_tokenizer.fit_on_texts(source_lang_text_data)
    target_tokenizer=keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    target_tokenizer.fit_on_texts(target_lang_text_data)

    """ Getting preprocessed data """
    src_seq, tgt_seq = preprocess_data(source_tokenizer,target_tokenizer,source_lang_text_data,target_lang_text_data,source_timesteps,target_timesteps)
    src_seq_train,src_seq_test,tgt_seq_train,tgt_seq_test=train_test_split(src_seq,tgt_seq,test_size=0.1)
    return src_seq_train,src_seq_test,tgt_seq_train,tgt_seq_test,source_tokenizer,target_tokenizer

def build_tokenizer_and_split_text():
    source_lang_text_data,target_lang_text_data=get_data()
    source_tokenizer=keras.preprocessing.text.Tokenizer(num_words=29014,oov_token='<UNK>')
    source_tokenizer.fit_on_texts(source_lang_text_data)
    target_tokenizer=keras.preprocessing.text.Tokenizer(num_words=39371,oov_token='<UNK>')
    target_tokenizer.fit_on_texts(target_lang_text_data)
    src_train, src_test, tgt_train, tgt_test = train_test_split(source_lang_text_data, target_lang_text_data, test_size=0.1)
    return src_train,src_test,tgt_train,tgt_test,source_tokenizer,target_tokenizer