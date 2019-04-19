from tensorflow.python.keras.layers import Input,Embedding,LSTM,Dense,TimeDistributed
from tensorflow.python.keras.models import Model

def define_nmt(hidden_size,embedding_dim,source_lang_timesteps,source_lang_vocab_size,target_lang_timesteps,target_lang_vocab_size):

    encoder_inputs=Input(shape=(None,),name='encoder_inputs')
    decoder_inputs=Input(shape=(None,),name='decoder_inputs')

    encoder_embedded=Embedding(source_lang_vocab_size,embedding_dim)(encoder_inputs)
    decoder_embedded=Embedding(target_lang_vocab_size,embedding_dim)(decoder_inputs)

    #encoder LSTM
    encoder_lstm=LSTM(hidden_size,return_sequences=True,return_state=True,name='encoder_lstm')
    encoder_out,encoder_state_h,encoder_state_c=encoder_lstm(encoder_embedded)
    encoder_states = [encoder_state_h, encoder_state_c]

    #decoder LSTM
    decoder_lstm=LSTM(hidden_size,return_sequences=True,return_state=True,name='decoder_lstm')
    decoder_out,decoder_state_h,decoder_state_c=decoder_lstm(decoder_embedded,initial_state=encoder_states)

    #dense layer
    dense=Dense(target_lang_vocab_size,activation='softmax',name='softmax_layer')
    dense_time=TimeDistributed(dense,name='time_distributed_layer')
    decoder_pred=dense_time(decoder_out)


    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary(line_length=225)

    return full_model


if __name__ == '__main__':
    """ Checking nmt model for toy example """
    define_nmt(hidden_size=64,embedding_dim=300, source_lang_timesteps=100,source_lang_vocab_size=254,target_lang_timesteps=100,target_lang_vocab_size=321)