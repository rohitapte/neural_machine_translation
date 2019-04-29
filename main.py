import io
import json
import os
from utils.data_utils import build_tokenizer_and_split_text
from utils.data_generator import DataGenerator
from tensorflow.python.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json

WHICH_MODEL="GRU"
saveParams={}
if WHICH_MODEL=='LSTM':
    from models.encoder_decoder_lstm import define_nmt,translate
elif WHICH_MODEL=="GRU":
    from models.encoder_decoder_gru import define_nmt,translate

def save_model(dir_hash,model_dict,full_model,encoder_model,decoder_model,source_tokenizer,target_tokenizer):
    if not os.path.exists('h5.models/' + dir_hash):
        os.makedirs('h5.models/' + dir_hash)
    with open('h5.models/' + dir_hash + "/model_params.json", 'w') as f:
        f.write(json.dumps(model_dict))
    tokenizer_json=source_tokenizer.to_json()
    with io.open('h5.models/'+dir_hash+'/source_tokenizer.json','w',encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json,ensure_ascii=False))
    tokenizer_json=target_tokenizer.to_json()
    with io.open('h5.models/'+dir_hash+'/target_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    full_model.save('h5.models/'+dir_hash+'/full_model.h5')
    encoder_model.save('h5.models/'+dir_hash+'/encoder_model.h5')
    decoder_model.save('h5.models/'+dir_hash+'/decoder_model.h5')

def load_saved_model(dir_hash):
    with open('h5.models/'+dir_hash+'/model_params.json','r') as f:
        for line in f:
            data=json.loads(line)
    with open('h5.models/'+dir_hash+'/source_tokenizer.json',encoding='utf-8') as f:
        temp=json.load(f)
        source_tokenizer=tokenizer_from_json(temp)
    with open('h5.models/'+dir_hash+'/target_tokenizer.json',encoding='utf-8') as f:
        temp=json.load(f)
        target_tokenizer=tokenizer_from_json(temp)
    full_model=load_model('h5.models/'+dir_hash+'/full_model.h5')
    encoder_model=load_model('h5.models/'+dir_hash+'/encoder_model.h5')
    decoder_model=load_model('h5.models/'+dir_hash+'/decoder_model.h5')
    return data,source_tokenizer,target_tokenizer,full_model,encoder_model,decoder_model

if __name__ == '__main__':
    #Define model parameters
    WHICH_GPU="1"

    MODE="TRAIN"
    SOURCE_TIMESTEPS,TARGET_TIMESTEPS=20,20
    HIDDEN_SIZE=128
    EMBEDDING_DIM=100
    NUM_EPOCHS=10
    BATCH_SIZE=64
    DROPOUT=0.5
    src_min_words=tgt_min_words=10
    source_file = 'data/europarl-v7.fr-en_small.fr'
    target_file = 'data/europarl-v7.fr-en_small.en'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = WHICH_GPU;

    model_params = WHICH_MODEL + "_" + str(HIDDEN_SIZE) + "_" + str(EMBEDDING_DIM) + "_" + str(SOURCE_TIMESTEPS) + "_" + str(TARGET_TIMESTEPS)
    dir_hash = model_params
    if os.path.exists('h5.models/' + dir_hash):
        print("Loading saved model")
        model_dict, source_tokenizer, target_tokenizer, full_model, encoder_model, decoder_model=load_saved_model(dir_hash)
        src_train,src_test,tgt_train,tgt_test,_, _=build_tokenizer_and_split_text(source_file=source_file,target_file=target_file,src_min_words=src_min_words,tgt_min_words=tgt_min_words)
        src_vsize=model_dict['SourceVocab']
        tgt_vsize=model_dict['TargetVocab']
        SOURCE_TIMESTEPS=model_dict['SourceTimeSteps']
        TARGET_TIMESTEPS=model_dict['TargetTimeSteps']
        HIDDEN_SIZE=model_dict['HiddenSize']
        EMBEDDING_DIM=model_dict['EmbeddingDim']
    else:
        print("Creating new model")
        src_train,src_test,tgt_train,tgt_test,source_tokenizer,target_tokenizer=build_tokenizer_and_split_text(source_file=source_file,target_file=target_file,src_min_words=src_min_words,tgt_min_words=tgt_min_words)
        if source_tokenizer.num_words is None:
            src_vsize = max(source_tokenizer.index_word.keys()) + 1
        else:
            if (max(source_tokenizer.index_word.keys()) + 1) < source_tokenizer.num_words:
                src_vsize = max(source_tokenizer.index_word.keys()) + 1
            else:
                src_vsize = source_tokenizer.num_words
        if target_tokenizer.num_words is None:
            tgt_vsize = max(target_tokenizer.index_word.keys()) + 1
        else:
            if max(target_tokenizer.index_word.keys()) + 1 < target_tokenizer.num_words:
                tgt_vsize = max(target_tokenizer.index_word.keys()) + 1
            else:
                tgt_vsize = target_tokenizer.num_words
        model_dict = {
            'Model': WHICH_MODEL,
            'HiddenSize': HIDDEN_SIZE,
            'EmbeddingDim': EMBEDDING_DIM,
            'SourceTimeSteps': SOURCE_TIMESTEPS,
            'TargetTimeSteps': TARGET_TIMESTEPS,
            'SourceVocab':src_vsize,
            'TargetVocab':tgt_vsize,
        }
        full_model, encoder_model, decoder_model = define_nmt(hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM,
                                                              source_lang_timesteps=SOURCE_TIMESTEPS,
                                                              source_lang_vocab_size=src_vsize,
                                                              target_lang_timesteps=TARGET_TIMESTEPS,
                                                              target_lang_vocab_size=tgt_vsize, dropout=DROPOUT)

    print('Source Vocab {}'.format(src_vsize))
    print('Target Vocab {}'.format(tgt_vsize))
    full_model.summary(line_length=225)
    encoder_model.summary(line_length=225)
    decoder_model.summary(line_length=225)

    if MODE=='TRAIN':
        training_generator=DataGenerator(source_text=src_train,target_text=tgt_train,source_tokenizer=source_tokenizer,target_tokenizer=target_tokenizer,
                                         target_vocab_size=tgt_vsize,source_timesteps=SOURCE_TIMESTEPS,target_timesteps=TARGET_TIMESTEPS,batch_size=BATCH_SIZE,shuffle=True)
        validation_generator=DataGenerator(source_text=src_test,target_text=tgt_test,source_tokenizer=source_tokenizer,target_tokenizer=target_tokenizer,
                                         target_vocab_size=tgt_vsize,source_timesteps=SOURCE_TIMESTEPS,target_timesteps=TARGET_TIMESTEPS,batch_size=BATCH_SIZE,shuffle=True)

        full_model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=6,epochs=NUM_EPOCHS)
        save_model(dir_hash,model_dict,full_model,encoder_model,decoder_model,source_tokenizer,target_tokenizer)

        sentence="Aux dires de son Président, la Commission serait en mesure de le faire."
        expected="According to its President, it is in a position to do so."
        translation=translate(sentence, encoder_model, decoder_model, source_tokenizer, target_tokenizer, src_vsize, tgt_vsize,SOURCE_TIMESTEPS,TARGET_TIMESTEPS)
        print("French: "+sentence)
        print("English: "+expected)
        print("Translation: "+translation)
