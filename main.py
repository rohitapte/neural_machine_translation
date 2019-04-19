import io
import json
import os
from utils.data_utils import build_tokenizer_and_split_text
from tensorflow.python.keras.utils import to_categorical
from utils.data_generator import DataGenerator
from models.encoder_decoder_lstm import define_nmt

if __name__ == '__main__':
    #Define model parameters
    WHICH_GPU="1"

    SOURCE_TIMESTEPS,TARGET_TIMESTEPS=100,100
    HIDDEN_SIZE=128
    EMBEDDING_DIM=100
    NUM_EPOCHS=10
    BATCH_SIZE=32

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = WHICH_GPU;

    def save_model(model,source_tokenizer,target_tokenizer):
        if not os.path.exists(os.path.join('..', 'h5.models')):
            os.mkdir(os.path.join('..', 'h5.models'))
        tokenizer_json=source_tokenizer.to_json()
        with io.open('h5.models/source_tokenizer.json','w',encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json,ensure_ascii=False))
        tokenizer_json=target_tokenizer.to_json()
        with io.open('h5.models/target_tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        model.save('h5.models/model.h5')

    src_train,src_test,tgt_train,tgt_test,source_tokenizer,target_tokenizer=build_tokenizer_and_split_text()
    src_vsize=max(source_tokenizer.index_word.keys())+1
    tgt_vsize=max(target_tokenizer.index_word.keys())+1
    print('Source Vocab {}'.format(src_vsize))
    print('Target Vocab {}'.format(tgt_vsize))
    training_generator=DataGenerator(source_text=src_train,target_text=tgt_train,source_tokenizer=source_tokenizer,target_tokenizer=target_tokenizer,
                                     target_vocab_size=tgt_vsize,source_timesteps=SOURCE_TIMESTEPS,target_timesteps=TARGET_TIMESTEPS,batch_size=BATCH_SIZE,shuffle=True)
    validation_generator=DataGenerator(source_text=src_test,target_text=tgt_test,source_tokenizer=source_tokenizer,target_tokenizer=target_tokenizer,
                                     target_vocab_size=tgt_vsize,source_timesteps=SOURCE_TIMESTEPS,target_timesteps=TARGET_TIMESTEPS,batch_size=BATCH_SIZE,shuffle=True)

    full_model=define_nmt(hidden_size=HIDDEN_SIZE,embedding_dim=EMBEDDING_DIM,
                          source_lang_timesteps=SOURCE_TIMESTEPS,source_lang_vocab_size=src_vsize,
                          target_lang_timesteps=TARGET_TIMESTEPS,target_lang_vocab_size=tgt_vsize)

    full_model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=6)
    save_model(full_model,source_tokenizer,target_tokenizer)