import io
import json
import os
from utils.data_utils import build_tokenizer_and_split_text
from utils.data_generator import DataGenerator
from tensorflow.python.keras.models import load_model


WHICH_MODEL="LSTM"
saveParams={}
if WHICH_MODEL=='LSTM':
    from models.encoder_decoder_lstm import define_nmt,translate

def save_model(dir_hash,full_model,encoder_model,decoder_model,source_tokenizer,target_tokenizer):
    tokenizer_json=source_tokenizer.to_json()
    with io.open('h5.models/'+dir_hash+'/source_tokenizer.json','w',encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json,ensure_ascii=False))
    tokenizer_json=target_tokenizer.to_json()
    with io.open('h5.models/'+dir_hash+'/target_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    full_model.save('h5.models/'+dir_hash+'/full_model.h5')
    encoder_model.save('h5.models/'+dir_hash+'/encoder_model.h5')
    decoder_model.save('h5.models/'+dir_hash+'/decoder_model.h5')

if __name__ == '__main__':
    #Define model parameters
    WHICH_GPU="1"

    SOURCE_TIMESTEPS,TARGET_TIMESTEPS=50,50
    HIDDEN_SIZE=128
    EMBEDDING_DIM=100
    NUM_EPOCHS=30
    BATCH_SIZE=64
    src_min_words=tgt_min_words=20

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = WHICH_GPU;

    src_train,src_test,tgt_train,tgt_test,source_tokenizer,target_tokenizer=build_tokenizer_and_split_text(source_file='data/europarl-v7.fr-en_50.fr',target_file='data/europarl-v7.fr-en_50.en',src_min_words=src_min_words,tgt_min_words=tgt_min_words)
    if source_tokenizer.num_words is None:
        src_vsize=max(source_tokenizer.index_word.keys())+1
    else:
        if (max(source_tokenizer.index_word.keys()) + 1) < source_tokenizer.num_words:
            src_vsize=max(source_tokenizer.index_word.keys())+1
        else:
            src_vsize=source_tokenizer.num_words
        #src_vsize=source_tokenizer.num_words
    if target_tokenizer.num_words is None:
        tgt_vsize=max(target_tokenizer.index_word.keys())+1
    else:
        if max(target_tokenizer.index_word.keys())+1<target_tokenizer.num_words:
            tgt_vsize=max(target_tokenizer.index_word.keys())+1
        else:
            tgt_vsize=target_tokenizer.num_words
        #tgt_vsize=target_tokenizer.num_words
    print('Source Vocab {}'.format(src_vsize))
    print('Target Vocab {}'.format(tgt_vsize))

    model_params = WHICH_MODEL + "_" + str(HIDDEN_SIZE) + "_" + str(EMBEDDING_DIM) + "_" + str(
        SOURCE_TIMESTEPS) + "_" + str(TARGET_TIMESTEPS)+"_"+str(src_vsize)+"_"+str(tgt_vsize)
    dir_hash = model_params

    if not os.path.exists('h5.models/' + dir_hash):
        os.makedirs('h5.models/' + dir_hash)
    model_dict = {
        'Model': WHICH_MODEL,
        'HiddenSize': HIDDEN_SIZE,
        'EmbeddingDim': EMBEDDING_DIM,
        'SourceTimeSteps': SOURCE_TIMESTEPS,
        'TargetTimeSteps': TARGET_TIMESTEPS,
        'SourceVocab':src_vsize,
        'TargetVocab':tgt_vsize,
    }
    with open('h5.models/' + dir_hash + "/model_params.json",'w') as f:
        f.write(json.dumps(model_dict))

    training_generator=DataGenerator(source_text=src_train,target_text=tgt_train,source_tokenizer=source_tokenizer,target_tokenizer=target_tokenizer,
                                     target_vocab_size=tgt_vsize,source_timesteps=SOURCE_TIMESTEPS,target_timesteps=TARGET_TIMESTEPS,batch_size=BATCH_SIZE,shuffle=True)
    validation_generator=DataGenerator(source_text=src_test,target_text=tgt_test,source_tokenizer=source_tokenizer,target_tokenizer=target_tokenizer,
                                     target_vocab_size=tgt_vsize,source_timesteps=SOURCE_TIMESTEPS,target_timesteps=TARGET_TIMESTEPS,batch_size=BATCH_SIZE,shuffle=True)

    if os.path.isfile('h5.models/' + dir_hash + "/model.h5"):
        print("Existing model found. loading")
        full_model=load_model('h5.models/' + dir_hash + "/model.h5")
    else:
        print("No existing model found. Creating from scratch.")
        full_model,encoder_model,decoder_model=define_nmt(hidden_size=HIDDEN_SIZE,embedding_dim=EMBEDDING_DIM,
                          source_lang_timesteps=SOURCE_TIMESTEPS,source_lang_vocab_size=src_vsize,
                          target_lang_timesteps=TARGET_TIMESTEPS,target_lang_vocab_size=tgt_vsize)

    full_model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=6,epochs=NUM_EPOCHS)
    save_model(dir_hash,full_model,encoder_model,decoder_model,source_tokenizer,target_tokenizer)

    sentence="Ce n' est pas demander beaucoup."
    expected="It is not a lot to ask."
    translation=translate(sentence, encoder_model, decoder_model, source_tokenizer, target_tokenizer, src_vsize, tgt_vsize,SOURCE_TIMESTEPS,TARGET_TIMESTEPS)
    print("French: "+sentence)
    print("English: "+expected)
    print("Translation: "+translation)
