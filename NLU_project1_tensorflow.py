'''
NLU project 1
'''

import tensorflow as tf
import numpy as np
from load_embeddings import load_embedding
from Preprocessing import preprocessing

class RNN_class(object):
    """ built the RNN """
    def __init__(self, rnn_settings, training_mode):
        self.sentence_length = rnn_settings['sentence_length']    # every word corresponds to a time step
        self.batch_size = rnn_settings['batch_size']
        self.embedding_size = rnn_settings['embedding_size']
        self.lstm_size = rnn_settings['lstm_size']
        self.vocabulary_size = rnn_settings['vocabulary_size']
        self.learning_rate = rnn_settings['learning_rate']
        self.epoch_size = rnn_settings['epoch_size']
        self.clip_gradient = rnn_settings['clip_gradient']
        self.num_batches = int(self.vocabulary_size/self.batch_size)
        self.training_mode = training_mode 
        
        # initialize the placeholders
        self.input_x = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int32) # [batch_size, sentence_length]
        self.input_y = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int64)
        
        with tf.variable_scope("embedding"):
            # create embeddings
            embedding_matrix= tf.get_variable(name="embedding", initializer = tf.random_uniform([self.vocabulary_size, self.embedding_size], -0.1, 0.1))
            embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, self.input_x) # [None, sentence_length, vocab_size, embedding_size]
            
        with tf.variable_scope('rnn_cell'):
            # Initial state of the LSTM memory.
            lstm = tf.contrib.rnn.LSTMCell(self.lstm_size, forget_bias=0.0, state_is_tuple=True, reuse=not training_mode)
        
        with tf.variable_scope('rnn_operations'):
            # rnn operation
            self.initial_state = lstm.zero_state(self.batch_size, dtype = tf.float32)
            state = self.initial_state
        
            outputs = []
            for word in range(self.sentence_length):
                if word > 0: 
                     tf.get_variable_scope().reuse_variables()
                (lstm_output, state) = lstm(embedded_inputs[:, word], state)
                outputs.append(lstm_output)
            self.output = tf.reshape(tf.concat(outputs, 1), [-1, self.lstm_size])


        with tf.variable_scope('softmax_variables'):
            # Set model weights
            W = tf.get_variable(name = "W", shape=[self.lstm_size, self.vocabulary_size], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            
            b = tf.get_variable(name = "b", shape=[self.vocabulary_size], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            
#            if task_C:
#                W_pro = tf.get_variable(name = "W_projection", shape=[self.lstm_size_2, self.lstm_size], 
#                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
#            
#                b_pro = tf.get_variable(name = "b_projection", shape=[self.vocabulary_size], 
#                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
#            
        with tf.variable_scope('logits'):
            logits = tf.add(tf.matmul(self.output, W),b) 
            logits = tf.reshape(logits, [self.batch_size, self.sentence_length, self.vocabulary_size])
        
        with tf.variable_scope('loss'):
            # loss
            # TODO: reduce sum or mean? piazza says mean
            # negative or not?
            
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.input_y)
            self.loss =  tf.reduce_sum(self.loss, 1)
        
        if training_mode:
        
            with tf.variable_scope('predictions'):
                self.predictions = tf.argmax(logits,2)  # [batch_size, sentence_length]
                self.predictions = tf.cast(self.predictions, tf.int64)
                correct = tf.equal(self.predictions, self.input_y)
            
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
            
            with tf.variable_scope('training_operations'):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_gradient)
                optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
                
            print('output', self.output.get_shape(), self.output.dtype)
            print('x size', self.input_x.get_shape(), self.input_x.dtype)
            print('y size', self.input_y.get_shape(), self.input_y.dtype)
            print('logit', logits.get_shape(), logits.dtype)
            print('loss size', self.loss.get_shape(), self.loss.dtype)
            print('pred size', self.predictions.get_shape(), self.predictions.dtype)
            #print (tf.trainable_variables())
            
            self.perplexity = self.calculate_perplexity(logits = logits, 
                                                        sentence_length = self.sentence_length,
                                                        batch_size = self.batch_size)
    
#    def calculate_perplexity(self, y_true, y_pred):
#        
#        def log2(x):
#          x = tf.cast(x, tf.float64)
#          numerator = tf.log(x)
#          denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
#          return tf.cast(numerator / denominator, tf.int64)
#        
#        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * log2(y_pred), reduction_indices=[1]))
#        perplexity = tf.pow(tf.cast(2, tf.int64), cross_entropy)
#        return perplexity     
    
    def calculate_perplexity(self, logits, sentence_length, batch_size):
        sentence_probability = tf.nn.softmax(logits)
        
        def log2(x):
          x = tf.cast(x, tf.float64)
          numerator = tf.log(x)
          denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
          return tf.cast(numerator / denominator, tf.int64)
      
        log_calc = log2(sentence_probability)
        log_sum = tf.reduce_sum(log_calc,[1,2])/sentence_length
        log_sum = tf.reshape(log_sum, [batch_size])
        print('per', log_sum.get_shape())
        return tf.pow(tf.cast(2, tf.float64), -log_sum)
    
    def train_rnn(self, model, session, X, y):
        """Runs the model on the given data."""

        costs = 0.0
        iters = 0
    
        # iterate over all epochs
        for epoch in range(self.epoch_size):
             print ('epoch: ', epoch, self.num_batches)
             # iterate over all batches
             for batch_i in range(self.num_batches):
                # get batches
                start = batch_i * self.batch_size
                end = min((batch_i + 1) * self.batch_size, len(X))
                    
                _, loss, acc, pred, per_new = session.run([self.train_op, self.loss, self.accuracy, self.predictions, self.perplexity], 
                                           feed_dict = {self.input_x: X[start:end],
                                                        self.input_y: y[start:end]})
                costs += loss
                iters += self.sentence_length # we don't need this, since we are taking the mean
                # TODO: check perplexity
                perplexity = np.exp(costs/iters) * np.log(2) # convert nats to bits
                print('Training: batch: ', batch_i , 'loss: ', np.sum(loss), 'accuracy: ', acc, 'perplexity: ', np.sum(perplexity))
               

def main_train(train_X, train_Y, eval_X, eval_Y, word_dict):            
    # reset the built graph
    tf.reset_default_graph()
        
    rnn_settings = {
        'sentence_length' : 30-1,        
        'batch_size' : 64, # given
        'embedding_size' : 100, #given
        'lstm_size' : 512,
        'vocabulary_size' : 20000,
        'learning_rate' : 0.001, # default
        'epoch_size' : 1,
        'clip_gradient' : 5.0
        }
    
    rnn_train = RNN_class(rnn_settings, training_mode = True)
    rnn_valid = RNN_class(rnn_settings, training_mode = False)

    # Launch the graph
    with tf.Session() as session:
        # Initialize the variables 
        session.run(tf.global_variables_initializer())
        
        
        task_B = True
        if task_B:   
            embedding_matrix= tf.get_variable(name="embedding", initializer = tf.random_uniform([20000, 100], -0.1, 0.1))
            load_embedding(session = session, 
                           vocab = word_dict, 
                           emb = embedding_matrix, 
                           path = r'C:\Users\mauro\Desktop\CAS\_Natural Language Understanding\Project\data\wordembeddings-dim100.word2vec', 
                           dim_embedding = 100, 
                           vocab_size = 20000)
        

        # train the model
        rnn_train.train_rnn(rnn_train, session, train_X, train_Y)
        
        # validate the model
        rnn_valid.test_rnn(rnn_valid, session, eval_X, eval_Y)
        # evaluate the model
        #model.train(model, session, )

        
        
            
# run the main method
pathData = r'C:\Users\mauro\Desktop\CAS\_Natural Language Understanding\Project\data'
train_file = pathData + '\sentences.train'
test_file = pathData + '\sentences.eval'

train_X, train_Y, word_dict = preprocessing(train_file)      
#eval_X, eval_Y, word_dict = preprocessing(test_file)      

main_train(train_X, train_Y, eval_X, eval_Y, word_dict)            


