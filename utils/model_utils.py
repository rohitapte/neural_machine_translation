import os
import io
import json
from tensorflow.python.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from layers.attention import AttentionLayer

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
    full_model=load_model('h5.models/'+dir_hash+'/full_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    encoder_model=load_model('h5.models/'+dir_hash+'/encoder_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    decoder_model=load_model('h5.models/'+dir_hash+'/decoder_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    return data,source_tokenizer,target_tokenizer,full_model,encoder_model,decoder_model