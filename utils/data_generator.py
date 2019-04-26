from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn.utils import shuffle
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

class DataGenerator(Sequence):
    def __init__(self,source_text,target_text,source_tokenizer,target_tokenizer,target_vocab_size,source_timesteps,target_timesteps,batch_size=32,shuffle=True):
        self.source_text=source_text
        self.target_text=target_text
        self.source_tokenizer=source_tokenizer
        self.target_tokenizer=target_tokenizer
        self.target_vocab_size=target_vocab_size
        self.source_timesteps=source_timesteps
        self.target_timesteps=target_timesteps
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.source_text)/float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle==True:
            self.source_text,self.target_text=shuffle(self.source_text,self.target_text)

    def __getitem__(self,idx):
        source_text=self.source_text[idx * self.batch_size:(idx + 1) * self.batch_size]
        target_text=self.target_text[idx * self.batch_size:(idx + 1) * self.batch_size]
        source_text_encoded = self.source_tokenizer.texts_to_sequences(source_text)
        target_text_encoded = self.target_tokenizer.texts_to_sequences(target_text)
        source_preproc_text = pad_sequences(source_text_encoded, padding='post', maxlen=self.source_timesteps)
        target_preproc_text = pad_sequences(target_text_encoded, padding='post', maxlen=self.target_timesteps)
        target_categorical=to_categorical(target_preproc_text,num_classes=self.target_vocab_size)
        return [source_preproc_text,target_preproc_text[:,:-1]],target_categorical[:,1:,:]


class DataGeneratorOneHot(Sequence):
    def __init__(self,source_text,target_text,source_tokenizer,source_vocab_size,target_tokenizer,target_vocab_size,source_timesteps,target_timesteps,batch_size=32,shuffle=True):
        self.source_text=source_text
        self.target_text=target_text
        self.source_tokenizer=source_tokenizer
        self.source_vocab_size=source_vocab_size
        self.target_tokenizer=target_tokenizer
        self.target_vocab_size=target_vocab_size
        self.source_timesteps=source_timesteps
        self.target_timesteps=target_timesteps
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.source_text)/float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle==True:
            self.source_text,self.target_text=shuffle(self.source_text,self.target_text)

    def __getitem__(self,idx):
        source_text=self.source_text[idx * self.batch_size:(idx + 1) * self.batch_size]
        target_text=self.target_text[idx * self.batch_size:(idx + 1) * self.batch_size]
        source_text_encoded = self.source_tokenizer.texts_to_sequences(source_text)
        target_text_encoded = self.target_tokenizer.texts_to_sequences(target_text)
        source_preproc_text = pad_sequences(source_text_encoded, padding='post', maxlen=self.source_timesteps)
        target_preproc_text = pad_sequences(target_text_encoded, padding='post', maxlen=self.target_timesteps)
        source_categorical=to_categorical(source_preproc_text,num_classes=self.source_vocab_size)
        target_categorical=to_categorical(target_preproc_text,num_classes=self.target_vocab_size)
        return [source_categorical,target_categorical[:,:-1,:]],target_categorical[:,1:,:]