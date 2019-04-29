from tensorflow.python.keras.layers import Input,Embedding,LSTM,Dense,TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np

def define_nmt(hidden_size,embedding_dim,source_lang_timesteps,source_lang_vocab_size,target_lang_timesteps,target_lang_vocab_size,dropout):

    encoder_inputs=Input(shape=(None,),name='encoder_inputs')
    decoder_inputs=Input(shape=(None,),name='decoder_inputs')

    encoder_embedding_layer=Embedding(input_dim=source_lang_vocab_size,output_dim=embedding_dim,mask_zero=True,input_length=source_lang_timesteps)
    encoder_embedded=encoder_embedding_layer(encoder_inputs)
    decoder_embedding_layer=Embedding(input_dim=target_lang_vocab_size,output_dim=embedding_dim, mask_zero=True,input_length=target_lang_timesteps)
    decoder_embedded=decoder_embedding_layer(decoder_inputs)

    #encoder LSTM
    encoder_lstm=LSTM(hidden_size,return_sequences=True,return_state=True,name='encoder_lstm',dropout=dropout, recurrent_dropout=dropout)
    encoder_out,encoder_state_h,encoder_state_c=encoder_lstm(encoder_embedded)
    encoder_states = [encoder_state_h, encoder_state_c]

    #decoder LSTM
    decoder_lstm=LSTM(hidden_size,return_sequences=True,return_state=True,name='decoder_lstm',dropout=dropout, recurrent_dropout=dropout)
    decoder_out,decoder_state_h,decoder_state_c=decoder_lstm(decoder_embedded,initial_state=encoder_states)

    #dense layer
    dense=Dense(target_lang_vocab_size,activation='softmax',name='softmax_layer')
    dense_time=TimeDistributed(dense,name='time_distributed_layer')
    decoder_pred=dense_time(decoder_out)

    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    encoder_model=Model(encoder_inputs,[encoder_out]+encoder_states)

    inf_decoder_state_input_h = Input(shape=(hidden_size,))
    inf_decoder_state_input_c = Input(shape=(hidden_size,))
    inf_decoder_states_inputs = [inf_decoder_state_input_h, inf_decoder_state_input_c]
    inf_decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    inf_decoder_embedded = decoder_embedding_layer(inf_decoder_inputs)

    inf_decoder_out, inf_decoder_state_h, inf_decoder_state_c = decoder_lstm(inf_decoder_embedded, initial_state=inf_decoder_states_inputs)
    inf_decoder_states = [inf_decoder_state_h, inf_decoder_state_c]
    inf_decoder_pred=dense_time(inf_decoder_out)
    decoder_model=Model(inputs=[inf_decoder_inputs]+inf_decoder_states_inputs,outputs=[inf_decoder_pred]+inf_decoder_states)
    return full_model,encoder_model,decoder_model

def translate(sentence,encoder_model,decoder_model,source_tokenizer,target_tokenizer,src_vsize,tgt_vsize,source_timesteps,target_timesteps):
    target="sentencestart"
    source_text_encoded = source_tokenizer.texts_to_sequences([sentence])
    target_text_encoded = target_tokenizer.texts_to_sequences([target])
    source_preproc_text = pad_sequences(source_text_encoded, padding='post', maxlen=source_timesteps)
    target_preproc_text=pad_sequences(target_text_encoded,padding='post',maxlen=1)
    encoder_out,enc_last_state_h,enc_last_state_c=encoder_model.predict(source_preproc_text)
    enc_last_state=[enc_last_state_h,enc_last_state_c]
    continuePrediction=True
    output_sentence=''
    total=0
    while continuePrediction:
        decoder_pred,decoder_state_h,decoder_state_c=decoder_model.predict([target_preproc_text]+enc_last_state)
        index_value = np.argmax(decoder_pred, axis=-1)[0, 0]
        sTemp = target_tokenizer.index_word.get(index_value, 'UNK')
        output_sentence += sTemp + ' '
        total += 1
        if total >= target_timesteps or sTemp == 'sentenceend':
            continuePrediction = False
        enc_last_state=[decoder_state_h,decoder_state_c]
        target_preproc_text[0,0]=index_value
    return output_sentence

if __name__ == '__main__':
    """ Checking nmt model for toy example """
    define_nmt(hidden_size=64,embedding_dim=300, source_lang_timesteps=100,source_lang_vocab_size=254,target_lang_timesteps=100,target_lang_vocab_size=321)