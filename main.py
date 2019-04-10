import tensorflow as tf
import os
from nmt_model import NMTModel

#system specific stuff
tf.app.flags.DEFINE_integer("gpu", 1, "Which GPU to use, if you have multiple.")

#Deep learning related
tf.app.flags.DEFINE_string("path","data/spa.txt","Path to Spanish-english file")
tf.app.flags.DEFINE_integer("max_len_english",54,"Maximum number of words in English sentences")
tf.app.flags.DEFINE_integer("max_len_spanish",59,"Maximum number of words in Spanish sentences")
tf.app.flags.DEFINE_float("test_size",0.2,"Size of the test set")
tf.app.flags.DEFINE_integer("embedding_size",128,"Dimension of word embedding")
tf.app.flags.DEFINE_integer("hidden_size",512,"Size of the LSTM Hidden states")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
config=tf.ConfigProto()
config.gpu_options.allow_growth = True

model=NMTModel(FLAGS)