from __future__ import absolute_import
from __future__ import division
from data_batcher_spa import SPADataObject
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import rnn_cell

class NMTModel(object):
    def __init__(self,FLAGS,name="NMTModel"):
        self.FLAGS=FLAGS
        self.name=name
        self.dataObject=SPADataObject(FLAGS.path,numexamples=30000,maxlen_english=FLAGS.max_len_english,maxlen_spanish=FLAGS.max_len_spanish,test_size=FLAGS.test_size)
        self.english_vocabulary_size=len(self.dataObject.englishDict)
        self.spanish_vocabulary_size = len(self.dataObject.spanishDict)
        self.add_placeholders()
        self.add_embedding_layer()
        self.build_graph()

    def add_placeholders(self):
        self.input_ids=tf.placeholder(tf.int32,shape=[None,self.FLAGS.max_len_english])
        self.input_mask=tf.placeholder(tf.int32,shape=[None,self.FLAGS.max_len_english])
        self.output_ids=tf.placeholder(tf.int32,shape=[None,self.FLAGS.max_len_spanish])
        self.keep_prob=tf.placeholder_with_default(1.0,shape=())

    def add_embedding_layer(self):
        self.word_embeddings=tf.get_variable("word_embeddings",[self.english_vocabulary_size,self.FLAGS.embedding_size])
        self.embedded_word_ids=tf.nn.embedding_lookup(self.word_embeddings,self.input_mask)   #batch_size,max_len_english,embedding_size

    def build_graph(self):
        with vs.variable_scope("RNNEncoder"):
            self.rnn_cell_fw=rnn_cell.LSTMCell(self.FLAGS.hidden_size)
            self.rnn_cell_fw=DropoutWrapper(self.rnn_cell_fw,input_keep_prob=self.keep_prob)
            self.rnn_cell_bw=rnn_cell.LSTMCell(self.FLAGS.hidden_size)
            self.rnn_cell_bw=DropoutWrapper(self.rnn_cell_bw,input_keep_prob=self.keep_prob)

            input_lens = tf.reduce_sum(self.input_mask, reduction_indices=1)  # shape (batch_size)
            (fw_out, bw_out),(fw_final,bw_final) = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw,self.embedded_word_ids,input_lens,dtype=tf.float32)
            final_state= tf.concat([fw_final,bw_final],2)
            final_state=tf.nn.dropout(final_state,self.keep_prob)
            self.final_state = tf.identity(final_state, name='final_state')

