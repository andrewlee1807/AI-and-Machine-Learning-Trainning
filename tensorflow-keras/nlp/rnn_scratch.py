import keras as K
import tensorflow as tf
import pandas as pd

class RNN_scratch(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(RNN_scratch, self).__init__()

        # initialize weight matries
        self.W_xh = self.add_weight([rnn_units,input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        # initialize hidden state to zeros
        self.h = tf.zeros([rnn_units, 1])

        
    def call(self, x): 
        # update hiden state
        self.h = tf.tanh(self.W_hh*self.h + self.W_xh*x)
        output = self.W_hy*self.h

        return (output, self.h)


# Data Preparation
data = pd.read_csv(r'D:\\AI\\AI-and-Machine-Learning-Trainning\\Pytorch\\nlp\\data\\neural_network_patent_query.csv')


st1 = data.loc[0].values

rnn = RNN_scratch(50,5,1)
sentence = ["i", "love", "recurrent", "neural"]

for word in sentence:
    predict, hidden_state = rnn.call(word)

print(predict)

print(tf.__version__)
print(tf.test.is_gpu_available())