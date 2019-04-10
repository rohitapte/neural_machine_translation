import unicodedata
import re
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = w.replace('¿', '')
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)

class SPADataObject(object):
    def __init__(self,path,numexamples=None,maxlen_english=54,maxlen_spanish=59,test_size=0.2):
        en, sp = create_dataset(path,numexamples)
        index=1
        self.englishDict={}
        self.reverseEnglishDict={}
        self.englishDict['<PAD>']=0
        self.reverseEnglishDict[0]='<PAD>'
        self.englishDict['<UNK>']=1
        self.reverseEnglishDict[1]='<UNK>'
        for sentence in en:
            words=sentence.split(" ")
            for word in words:
                if word not in self.englishDict:
                    index+=1
                    self.englishDict[word]=index
                    self.reverseEnglishDict[index]=word
        index=1
        self.spanishDict={}
        self.reverseSpanishDict={}
        self.spanishDict['<PAD>'] = 0
        self.reverseSpanishDict[0]='<PAD>'
        self.spanishDict['<UNK>']=1
        self.reverseSpanishDict[1]='<UNK>'
        for sentence in sp:
            words=sentence.split(" ")
            for word in words:
                if word not in self.spanishDict:
                    index+=1
                    self.spanishDict[word]=index
                    self.reverseSpanishDict[index]=word

        en_encoded_text=np.array(self.encode_text(en,self.englishDict,maxlen_english))
        sp_encoded_text=np.array(self.encode_text(sp,self.spanishDict,maxlen_spanish))
        self.en_encoded_train,self.en_encoded_test,self.sp_encoded_train,self.sp_encoded_test=train_test_split(en_encoded_text,sp_encoded_text,test_size=test_size)

    def encode_text(self,text,langDict,padLength):
        encoded_data=[]
        for sentence in text:
            endcoded_sentence=[langDict.get(word,langDict['<UNK>']) for word in sentence.split(" ")]
            endcoded_sentence=endcoded_sentence+[langDict['<PAD>']]*(padLength-len(endcoded_sentence))
            encoded_data.append(endcoded_sentence)
        return encoded_data

    def decode_text(self,list_of_ids,langDict):
        return ' '.join([langDict.get(item,'<UNK>') for item in list_of_ids])

    def generate_one_epoch(self,batch_size):
        num_batches = int(self.en_encoded_train.shape[0]) // batch_size
        if batch_size*num_batches < self.en_encoded_train.shape[0]: num_batches += 1
        self.en_encoded_train, self.sp_encoded_train = shuffle(self.en_encoded_train, self.sp_encoded_train)
        for i in range(num_batches):
            input_mask=(self.en_encoded_train[i*batch_size:(i+1)*batch_size]!=self.englishDict['<PAD>']).astype(np.int32)
            yield self.en_encoded_train[i*batch_size:(i+1)*batch_size],self.sp_encoded_train[i*batch_size:(i+1)*batch_size],input_mask

    def generate_test_epoch(self,batch_size):
        num_batches=int(self.en_encoded_test.shape[0]) // batch_size
        if batch_size*num_batches<self.en_encoded_test.shape[0]: num_batches += 1
        for i in range(num_batches):
            input_mask=(self.en_encoded_test[i*batch_size:(i+1)*batch_size]!=self.englishDict['<PAD>']).astype(np.int32)
            yield self.en_encoded_test[i*batch_size:(i+1)*batch_size],self.sp_encoded_test[i*batch_size:(i+1)*batch_size],input_mask

#temp=SPADataObject('data/spa.txt')
#for en,sp,input_mask in temp.generate_one_epoch(1000):
#    print(en.shape,sp.shape,input_mask.shape)
#    aa=en[1,:]
#    bb=sp[1,:]
#    print(temp.decode_text(aa,temp.reverseEnglishDict))
#    print(temp.decode_text(bb,temp.reverseSpanishDict))
#    print(input_mask[1:])

