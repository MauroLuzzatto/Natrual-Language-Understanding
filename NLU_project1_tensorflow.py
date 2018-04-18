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
        self.lstm_size_down_projected = rnn_settings['lstim_size_down_projected']
        self.vocabulary_size = rnn_settings['vocabulary_size']
        self.learning_rate = rnn_settings['learning_rate']
        self.number_of_epochs = rnn_settings['number_of_epochs']
        self.clip_gradient = rnn_settings['clip_gradient']
        self.num_batches = int(self.vocabulary_size/self.batch_size) + 1
        self.training_mode = training_mode
        self.task = rnn_settings['Task']
        
        # initialize the placeholders
        self.input_x = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int32) # [batch_size, sentence_length]
        self.input_y = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int64)
        
        with tf.variable_scope("embedding"):
            # create embeddings
            embedding_matrix= tf.get_variable(name="embedding", initializer = tf.random_uniform([self.vocabulary_size, self.embedding_size], -0.1, 0.1))
            embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, self.input_x) # [None, sentence_length, vocab_size, embedding_size]
            
        with tf.variable_scope('rnn_cell'):
            # Initial state of the LSTM memory.
            lstm = tf.contrib.rnn.LSTMCell(self.lstm_size, forget_bias=0.0, state_is_tuple=True)
        
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
          
        
        if self.task == 'C':
            with tf.variable_scope('softmax_variables'):
                # Set model weights
                W_down_pro = tf.get_variable(name = "W_down_projection", shape=[self.lstm_size, self.lstm_size_down_projected], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            
                b_down_pro = tf.get_variable(name = "b_down_projection", shape=[self.lstm_size_down_projected], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                
                W = tf.get_variable(name = "W", shape=[self.lstm_size_down_projected, self.vocabulary_size], 
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                
                b = tf.get_variable(name = "b", shape=[self.vocabulary_size], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                
            with tf.variable_scope('logits'):
                down_projected = tf.add(tf.matmul(self.output, W_down_pro),b_down_pro) # [29*64, 1024] *[1024,512] + [1,512] 
                logits = tf.add(tf.matmul(down_projected, W),b) # [29*64, 512] * [512, 20000] + [1*20000]
                logits = tf.reshape(logits, [self.batch_size, self.sentence_length, self.vocabulary_size])
                self.sentence_probability = tf.nn.softmax(logits)
            
            print('check tensor size task C')
            print('W_down_prjected', W_down_pro.get_shape(), W_down_pro.dtype)
            print('down_projected', down_projected.get_shape(), down_projected.dtype)
            print('logits', logits.get_shape(), logits.dtype)
        
        else:    
          with tf.variable_scope('softmax_variables'):
                # Set model weights
                W = tf.get_variable(name = "W", shape=[self.lstm_size, self.vocabulary_size], 
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b = tf.get_variable(name = "b", shape=[self.vocabulary_size], 
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        
          with tf.variable_scope('logits'):
                logits = tf.add(tf.matmul(self.output, W),b) 
                logits = tf.reshape(logits, [self.batch_size, self.sentence_length, self.vocabulary_size])
                self.sentence_probability = tf.nn.softmax(logits)
        
        with tf.variable_scope('loss'):
            # TODO: reduce sum or mean? piazza says mean
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.input_y)
            self.loss =  tf.reduce_sum(self.loss, 1)
            #tf.summary.scaler('loss', tf.reduce_sum(self.loss))
        
            with tf.variable_scope('predictions'):
                self.predictions = tf.argmax(logits,2)  # [batch_size, sentence_length]
                self.predictions = tf.cast(self.predictions, tf.int64)
                correct = tf.equal(self.predictions, self.input_y)
            
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
                tf.summary.scalar('accuracy', self.accuracy)
                
            if training_mode:
                with tf.variable_scope('training_operations'):
                    tvars = tf.trainable_variables()
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_gradient)
                    optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
                    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
                
            print('output tensor: ', self.output.get_shape(), self.output.dtype)
            print('input_x tensor: ', self.input_x.get_shape(), self.input_x.dtype)
            print('input_y tensor: ', self.input_y.get_shape(), self.input_y.dtype)
            print('logits tensor: ', logits.get_shape(), logits.dtype)
            print('loss tensor: ', self.loss.get_shape(), self.loss.dtype)
            print('predictions tensor: ', self.predictions.get_shape(), self.predictions.dtype)
            #print (tf.trainable_variables())
            
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            self.merged = tf.summary.merge_all()
            
                   
    def create_submission_file(self, perplexity_list, task = 'Missing'):
        # create the submission file
        fileName = 'groupXX.perplexity' + task
        with open(fileName, 'w') as file_handler:
            for item in perplexity_list:
                file_handler.write("{}\n".format(item))
        print('output file created for task: ', task)
    
    
    def calculate_perplexity(self, sentence_probability, input_y):
        """
        sentence_length = 29
        probability = sentence_probability[:,:,input_y]
        prob = probability[0,0,:,:]

#        prob1 = probability[0,0,:,:]
#        prob2 = probability[:,0,0,:]
#        prob3 = probability[0,:,0,:]
#        prob4 = probability[:,0,:,0]
        
        log_sum = np.sum(np.log(prob),1)/sentence_length

#        print (np.mean(np.exp(-np.sum(np.log(prob1),1)/sentence_length)))
#        print (np.mean(np.exp(-np.sum(np.log(prob2),1)/sentence_length)))
#        print (np.mean(np.exp(-np.sum(np.log(prob3),1)/sentence_length)))
#        print (np.mean(np.exp(-np.sum(np.log(prob4),1)/sentence_length)))

        print('prob', np.shape(prob))
        #np.power(2, -log_sum)
        return np.exp(-log_sum) 
 
        probability = sentence_probability[:,:,input_y][0,:,0,:]
        batch_size = np.shape(probability)[0]
        print('prob', np.shape(probability))
        
        pad_index = 0 # index of eos
        bos_index = 1
        # remove pad
        
        index_pad = ~(np.in1d(probability, pad_index).reshape(probability.shape))
        index_bos = ~(np.in1d(probability, bos_index).reshape(probability.shape))
        
        sum_index_pad = np.sum(index_pad, axis = 1)
        sum_index_bos = np.sum(index_bos, axis = 1)
        
        #print(index_pad)
        print('index_pad', index_pad.shape)
        print('sum_index_pad', sum_index_pad.shape)
        

        sum_all = np.sum([sum_index_pad, sum_index_eos, sum_index_bos], axis = 0)
        #print(sum_all)
        log_sum = np.sum(np.multiply(index_eos, np.multiply(index_bos, np.multiply(index_pad, np.log2(probability)))), axis = 1)
        #print(log_sum)
        print(sum_all.shape, log_sum.shape)
        perplexity = np.power(2, - np.divide(log_sum, sum_all))
        """
        
        batch_size = sentence_probability.shape[0]
        sentence_length = sentence_probability.shape[1]
        perplexity = np.zeros(batch_size)
        pad_index = 0 # index of pad
        
        for sentence_i in range(batch_size):
            word_index = 1 # start at 1 to discard bos
            log_sum = 0
            while ((word_index < sentence_length) and (input_y[sentence_i][word_index] != pad_index)): # stop when the first pad token is reached
                
                log_sum += np.log2(sentence_probability[sentence_i, word_index, 
                                                        input_y[sentence_i][word_index]])
                word_index += 1 
            word_index -= 1 # remove one count because of discarded bos
            perplexity[sentence_i] = np.power(2, -(log_sum/word_index))

        return perplexity
    
    
#    def calculate_perplexity(probabilities, sentences, pad_index):
#    """probabilities: matrix of shape [n_sentences, sentence length, voc_size]
#       sentences: array of shape [n_sentences, sentence length], containing word indexes of sentence batch
#       pad_index: the index of the '<pad>' word - given directly so we don't have to access the vocabulary here
#
#       returns list of length n_sentences with the perplexities"""
#
#    n_sentences = sentences.shape[0]
#    sentence_length = sentences.shape[1]
#
#    perplexities = np.empty(n_sentences)
#
#    for sent in range(n_sentences): # step through sentences of a batch
#
#        logsum = 0
#        n = 0
#
#       # print (np.sum(sentences[sent,:]==pad_index))
#
#        while(n < sentence_length and sentences[sent,n] != pad_index): # step through words of this sentence
#
#            index = sentences[sent, n]
#            prob = probabilities[sent, n, index]
#
#            logsum += np.log2(prob)
#            n += 1
#
#        logsum *= -1/n
#        perplexities[sent] = np.exp2(logsum)
#
#    return perplexities
    
    
    def test_rnn(self, model, session, X, y, task):
        """Runs the model on the test data"""
                
        pathToLog = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\Natrual-Language-Understanding\log'
        writer = tf.summary.FileWriter(pathToLog)
        writer.add_graph(session.graph)
        
        perplexity_list = []
        costs = 0.0
        iters = 0.0
        
        num_test_batches = int(len(y)/self.batch_size)
        
        for batch_i in range(10): #num_test_batches):
            # get batches
            start = batch_i * self.batch_size
            end = min((batch_i + 1) * self.batch_size, len(y))
            
            feed_dict = {self.input_x: X[start:end],
                         self.input_y: y[start:end]}
                
            loss, acc, pred, summary, \
            sentence_probability = session.run([self.loss, self.accuracy, self.predictions, self.merged, self.sentence_probability], 
                                               feed_dict)

            per_test = self.calculate_perplexity(sentence_probability, y[start:end])
            print ('test preplexity: ', np.mean(per_test))
            
            costs += loss
            iters += self.sentence_length # we don't need this, since we are taking the mean
            # TODO: check perplexity
            perplexity = np.exp(costs/iters) 
            print('Test set: loss: ', np.sum(loss), 'accuracy: ', acc, 'perplexity: ', np.mean(perplexity))
            writer.add_summary(summary, batch_i)
            perplexity_list.extend(perplexity)
        
        # create final suubmission file for a specific task
        self.create_submission_file(perplexity_list = perplexity_list, task = task)
        writer.add_summary(summary)
        writer.flush()
        writer.close()
       
        
    def train_rnn(self, model, session, X, y):
        """Runs the model on the given data."""
        costs = 0.0
        iters = 0.0
        
        pathToLog = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\Natrual-Language-Understanding\log'
        writer = tf.summary.FileWriter(pathToLog)
        writer.add_graph(session.graph)
                
        # iterate over all epochs
        for epoch_i in range(self.number_of_epochs):
             ### shuffle?
             print('#epoch: ', epoch_i) 
             # iterate over all batches
             for batch_i in range(10): #self.num_batches):
                # get batches
                start = batch_i * self.batch_size
                end = min((batch_i + 1) * self.batch_size, len(y))
                
                feed_dict = {self.input_x: X[start:end],
                             self.input_y: y[start:end]}
                    
                _, loss, acc, pred, summary, \
                sentence_probability = session.run([self.train_op, self.loss, self.accuracy, self.predictions, self.merged, self.sentence_probability], 
                                                   feed_dict)
    
                per_test = self.calculate_perplexity(sentence_probability, y[start:end])
                print ('test preplexity: ', np.mean(per_test))
                
                costs += loss
                iters += self.sentence_length # we don't need this, since we are taking the mean
                # TODO: check perplexity
                perplexity = np.exp(costs/iters) 
                print('Training: batch: ', batch_i , 'loss: ', np.sum(loss), 'accuracy: ', acc, 'perplexity: ', np.mean(perplexity))
                writer.add_summary(summary, epoch_i)
                writer.flush()
        
        writer.close()
       


def main_train(train_X, train_Y, eval_X, eval_Y, words_to_idx):            
   
    rnn_settings = {
        'sentence_length' : 30-1,        
        'batch_size' : 64, # given
        'embedding_size' : 100, #given
        'lstm_size' : 1024,
        'lstim_size_down_projected': 512, # lstm_size = 1024 for task C
        'vocabulary_size' : 20000,
        'learning_rate' : 0.001, # default
        'number_of_epochs' : 200,
        'clip_gradient' : 5.0,
        'Training_mode': True,
        'Task': 'A'
        }

    training_mode = rnn_settings['Training_mode']
    task = rnn_settings['Task']
    
    rnn_train = RNN_class(rnn_settings, training_mode = True)
    # Launch the graph
    with tf.Session() as session:

        if training_mode:    
            saver = tf.train.Saver()
            # Initialize the variables 
            session.run(tf.global_variables_initializer())
            
            if task == 'B':   
                embedding_matrix= tf.get_variable(name="embedding", initializer = tf.random_uniform([20000, 100], -0.1, 0.1))
                load_embedding(session = session, 
                               vocab = words_to_idx, 
                               emb = embedding_matrix, 
                               path = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\Data\wordembeddings-dim100.word2vec', 
                               dim_embedding = 100, 
                               vocab_size = 20000)
            
            
            # train the model
            rnn_train.train_rnn(rnn_train, session, train_X, train_Y)
            # export the trained meta-graph
            saver.save(session, 'graph\model.ckpt')
        else:
                    
            saver = tf.train.Saver()
            saver.restore(session, 'graph\model.ckpt')
            print('model restored')

        
            # validate the model
            rnn_train.test_rnn(rnn_train, session, eval_X, eval_Y, task)
            
      
 # reset the built graph
tf.reset_default_graph()

# run the main method
#pathData = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\data'
#pathData=r"/home/dario/Desktop/NLU/Projekt/Data"
trainFile = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\Data\sentences.train'
testFile =  r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\Data\sentences.eval'


#train_X, train_Y, words_to_idx, word_dict = preprocessing(pathToFile = trainFile, 
#                                                          training_mode = True, 
#                                                          words_to_idx = None, 
#                                                          word_dict = None)
#  
#eval_X, eval_Y, words_to_idx, word_dict = preprocessing(pathToFile = testFile, 
#                                                        training_mode = False, 
#                                                        words_to_idx = words_to_idx, 
#                                                        word_dict = word_dict)     

main_train(train_X, train_Y, eval_X, eval_Y, words_to_idx)            




      




#

#        
