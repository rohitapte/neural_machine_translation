from utils.data_utils import tokenize_sentences_and_split
import io
import json
import os
from tensorflow.python.keras.utils import to_categorical
from models.model_vanilla_encoder import define_nmt

#Define model parameters
WHICH_GPU="1"

SOURCE_TIMESTEPS,TARGET_TIMESTEPS=100,100
HIDDEN_SIZE=256
EMBEDDING_DIM=300
NUM_EPOCHS=10
BATCH_SIZE=256

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

src_seq_train,src_seq_test,tgt_seq_train,tgt_seq_test,source_tokenizer,target_tokenizer=tokenize_sentences_and_split(SOURCE_TIMESTEPS,TARGET_TIMESTEPS)
src_vsize=max(source_tokenizer.index_word.keys())+1
tgt_vsize=max(target_tokenizer.index_word.keys())+1
tgt_seq_train_categorical=to_categorical(tgt_seq_train,num_classes=tgt_vsize)

#print(src_seq_train.shape)
#print(tgt_seq_train.shape)
#print(tgt_seq_train[:,:-1].shape)
#print(tgt_seq_train_categorical.shape)


full_model=define_nmt(hidden_size=HIDDEN_SIZE,embedding_dim=EMBEDDING_DIM,
                      source_lang_timesteps=SOURCE_TIMESTEPS,source_lang_vocab_size=src_vsize,
                      target_lang_timesteps=TARGET_TIMESTEPS,target_lang_vocab_size=tgt_vsize)
full_model.fit(x=[src_seq_train,tgt_seq_train[:,:-1]],y=tgt_seq_train_categorical[:,1:,:],batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_split=0.1)
save_model(full_model,source_tokenizer,target_tokenizer)