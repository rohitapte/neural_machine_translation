from tensorflow.python.keras.layers import Input,Embedding,GRU,Dense,TimeDistributed,Bidirectional,Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
from layers.attention import AttentionLayer

def define_nmt(hidden_size,embedding_dim,source_lang_timesteps,source_lang_vocab_size,target_lang_timesteps,target_lang_vocab_size,dropout):

    encoder_inputs=Input(shape=(source_lang_timesteps,),name='encoder_inputs')
    decoder_inputs=Input(shape=(target_lang_timesteps-1,),name='decoder_inputs')

    encoder_embedding_layer = Embedding(input_dim=source_lang_vocab_size, output_dim=embedding_dim)
    encoder_embedded = encoder_embedding_layer(encoder_inputs)
    decoder_embedding_layer = Embedding(input_dim=target_lang_vocab_size, output_dim=embedding_dim)
    decoder_embedded = decoder_embedding_layer(decoder_inputs)

    #encoder GRU
    # encoder_gru=GRU(hidden_size,return_sequences=True,return_state=True,name='encoder_gru',dropout=dropout, recurrent_dropout=dropout)
    encoder_gru1 = Bidirectional(GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru1'))
    encoder_out1, encoder_state_f1, encoder_state_b1 = encoder_gru1(encoder_embedded)
    encoder_state1 = Concatenate()([encoder_state_f1, encoder_state_b1])
    encoder_gru2 = Bidirectional(GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru2'))
    encoder_out2, encoder_state_f2, encoder_state_b2 = encoder_gru2(encoder_out1)
    encoder_state2 = Concatenate()([encoder_state_f2, encoder_state_b2])
    encoder_states = [encoder_state1, encoder_state2]

    #decoder GRU
    # decoder_gru=GRU(hidden_size,return_sequences=True,return_state=True,name='decoder_gru',dropout=dropout, recurrent_dropout=dropout)
    decoder_gru1 = GRU(2 * hidden_size, return_sequences=True, return_state=True, name='decoder_gru1')
    decoder_out1, decoder_state1 = decoder_gru1(decoder_embedded, initial_state=encoder_state1)
    decoder_gru2 = GRU(2 * hidden_size, return_sequences=True, return_state=True, name='decoder_gru2')
    decoder_out2, decoder_state2 = decoder_gru2(decoder_out1, initial_state=encoder_state2)

    #attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out2, decoder_out2])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out2, attn_out])

    #dense layer
    dense=Dense(target_lang_vocab_size,activation='softmax',name='softmax_layer')
    dense_time=TimeDistributed(dense,name='time_distributed_layer')
    decoder_pred=dense_time(decoder_concat_input)

    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')
    full_model.summary(line_length=225)

    encoder_model = Model(encoder_inputs, [encoder_out2]+encoder_states)
    encoder_model.summary(line_length=225)

    inf_decoder_state1 = Input(shape=(2*hidden_size,))
    inf_decoder_state2 = Input(shape=(2*hidden_size,))
    inf_decoder_inputs = Input(shape=(1,), name='decoder_inputs')
    inf_encoder_outputs=Input(shape=(source_lang_timesteps,2*hidden_size,))
    inf_decoder_embedded = decoder_embedding_layer(inf_decoder_inputs)

    inf_decoder_out1, inf_decoder_state_out1= decoder_gru1(inf_decoder_embedded,initial_state=inf_decoder_state1)
    inf_decoder_out2, inf_decoder_state_out2 = decoder_gru2(inf_decoder_out1, initial_state=inf_decoder_state2)
    inf_attn_out, inf_attn_states = attn_layer([inf_encoder_outputs, inf_decoder_out2])
    inf_decoder_concat = Concatenate(axis=-1, name='concat')([inf_decoder_out2, inf_attn_out])

    inf_decoder_pred = TimeDistributed(dense)(inf_decoder_concat)
    decoder_model = Model(inputs=[inf_encoder_outputs,inf_decoder_inputs,inf_decoder_state1,inf_decoder_state2],outputs=[inf_decoder_pred,inf_attn_states,inf_decoder_state_out1,inf_decoder_state_out2])
    decoder_model.summary(line_length=225)
    return full_model, encoder_model, decoder_model

def translate(sentence,encoder_model,decoder_model,source_tokenizer,target_tokenizer,src_vsize,tgt_vsize,source_timesteps,target_timesteps):
    target="sentencestart"
    source_text_encoded = source_tokenizer.texts_to_sequences([sentence])
    target_text_encoded = target_tokenizer.texts_to_sequences([target])
    source_preproc_text = pad_sequences(source_text_encoded, padding='post', maxlen=source_timesteps)
    target_preproc_text=pad_sequences(target_text_encoded,padding='post',maxlen=1)
    encoder_out,enc_last_state1,enc_last_state2=encoder_model.predict(source_preproc_text)
    continuePrediction=True
    output_sentence=''
    total=0
    while continuePrediction:
        decoder_pred,attn_state,decoder_state1,decoder_state2=decoder_model.predict([encoder_out,target_preproc_text,enc_last_state1,enc_last_state2])
        index_value = np.argmax(decoder_pred, axis=-1)[0, 0]
        sTemp = target_tokenizer.index_word.get(index_value, 'UNK')
        output_sentence += sTemp + ' '
        total += 1
        if total >= target_timesteps or sTemp == 'sentenceend':
            continuePrediction = False
        enc_last_state1=decoder_state1
        enc_last_state2=decoder_state2
        target_preproc_text[0,0]=index_value
    return output_sentence

if __name__ == '__main__':
    """ Checking nmt model for toy example """
    full_model,encoder_model,decoder_model=define_nmt(hidden_size=128,embedding_dim=100,source_lang_timesteps=20,source_lang_vocab_size=254,target_lang_timesteps=20,target_lang_vocab_size=321,dropout=0.1)
    full_model.summary(line_length=225)
    encoder_model.summary(line_length=225)
    decoder_model.summary(line_length=225)