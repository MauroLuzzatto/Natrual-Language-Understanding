'''
NLU project 1

authors: Dario Kneubühler, Mauro Luzzatto, Thomas Brunschwiler
group: 12

'''

import tensorflow as tf
import numpy as np
import os

from load_embedding import load_embedding
from Preprocessing import preprocessing

class RNN_class(object):
    """ main class that builds the RNN with an LSTM cell """
    def __init__(self, rnn_settings):
        self.sentence_length = rnn_settings['sentence_length']    # every word corresponds to a time step
        self.embedding_size = rnn_settings['embedding_size']
        self.lstm_size = rnn_settings['lstm_size']
        self.lstm_size_down_projected = rnn_settings['lstm_size_down_projected']
        self.vocabulary_size = rnn_settings['vocabulary_size']
        self.learning_rate = rnn_settings['learning_rate']
        self.number_of_epochs = rnn_settings['number_of_epochs']
        self.clip_gradient = rnn_settings['clip_gradient']
        self.training_mode = rnn_settings['Training_mode'] 
        self.task = rnn_settings['Task']
        reuseVar=False
       
        # initialize the placeholders
        self.input_x = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int32) # [batch_size, sentence_length]
        self.input_y = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int64)
        self.batch_size = tf.shape(self.input_x)[0]
        
        with tf.variable_scope("embedding", reuse = reuseVar):
            # create embeddings
            embedding_matrix= tf.get_variable(name="embedding", initializer = tf.random_uniform([self.vocabulary_size, self.embedding_size], -0.1, 0.1))
            embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, self.input_x) # [None, sentence_length, vocab_size, embedding_size]
            
        with tf.variable_scope('rnn_cell', reuse = reuseVar):
            # Initial state of the LSTM memory
            lstm = tf.contrib.rnn.LSTMCell(self.lstm_size, forget_bias=0.0, state_is_tuple=True)
        
        with tf.variable_scope('rnn_operations', reuse = reuseVar):
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
            
            print('check tensor size task C: ')
            print('W_down_prjected: ', W_down_pro.get_shape(), W_down_pro.dtype)
            print('down_projected: ', down_projected.get_shape(), down_projected.dtype)
            print('logits: ', logits.get_shape(), logits.dtype)

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
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.input_y)
            self.loss =  tf.reduce_mean(self.loss, 1)
        
            with tf.variable_scope('predictions'):
                self.predictions = tf.argmax(logits,2)  # [batch_size, sentence_length]
                self.predictions = tf.cast(self.predictions, tf.int64)
                correct = tf.equal(self.predictions, self.input_y)
            
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
                tf.summary.scalar('accuracy', self.accuracy)
                
            if self.training_mode:
                with tf.variable_scope('training_operations'):
                    tvars = tf.trainable_variables()
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_gradient)
                    optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
                    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
                
        
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            self.merged = tf.summary.merge_all()
        
            
    def create_submission_file(self, perplexity_list, task = 'A'):
        """create a submission text file """
        
        if task in ['A', 'B', 'C']:
            fileName = 'group12.perplexity' + task
        else:
            fileName = 'group12.' + task
            
        with open(fileName, 'w') as file_handler:
            for item in perplexity_list:
                if not(self.task=='1.2'):
                    if np.isnan(item): continue #skip nans comming from dummy scentences with only pads
                    file_handler.write("{}\n".format(item))
        print('output file created for task: ', task)
    
    
    def calculate_perplexity_new(self, sentence_probability, input_y, words_to_idx):
        """method that calculates the perplexity without taking into 
            account <pad> and <bos> tokens"""
            
        batch_size = sentence_probability.shape[0]
        sentence_length = sentence_probability.shape[1]
        perplexity = np.zeros(batch_size)
        pad_index = words_to_idx['<pad>'] # index of pad
        
        for sentence_i in range(batch_size):
            word_index = 1 # start at 1 to discard bos
            log_sum = 0
            while ((word_index < sentence_length) and (input_y[sentence_i][word_index] != pad_index)): # stop when the first pad token is reached
                
                log_sum += np.log2(sentence_probability[sentence_i, word_index, 
                                                        input_y[sentence_i][word_index]])
                word_index += 1 
            word_index -= 1 # remove one count because of discarded bos
            try:
                # catch sentence with all pad's
                perplexity[sentence_i] = np.power(2, -(log_sum/word_index))
            except:
                print(input_y[sentence_i])
                print('sencentece with just pads')
                perplexity[sentence_i] = np.nan                
        
        return perplexity
    
    def test_rnn(self, model, session, X, y, task, rnn_settings, words_to_idx):
        """Runs the model on the test data"""
                
        pathToLog = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\logTest'
        writer = tf.summary.FileWriter(pathToLog)
        writer.add_graph(session.graph)
        
        batch_size = rnn_settings['batch_size']
        perplexity_list = []        
        num_test_batches = int(len(y)/batch_size)
        
        for batch_i in range(num_test_batches):
            # get batches
            start = batch_i * batch_size
            end = min((batch_i + 1) * batch_size, len(y))
            
            feed_dict = {self.input_x: X[start:end],
                         self.input_y: y[start:end]}
                
            loss, acc, pred, summary, \
            sentence_probability = session.run([self.loss, self.accuracy, self.predictions, 
                                                self.merged, self.sentence_probability], 
                                                feed_dict)

            perplexity = self.calculate_perplexity_new(sentence_probability, y[start:end], words_to_idx)
            print('Test set: loss: ', np.sum(loss), 'accuracy: ', acc, \
                  'perplexity: ', np.nanmean(perplexity))
            writer.add_summary(summary, batch_i)
            perplexity_list.extend(perplexity)
        
        # create final suubmission file for a specific task
        self.create_submission_file(perplexity_list = perplexity_list, task = task)
        writer.add_summary(summary)
        writer.flush()
        writer.close()
       
        
    def train_rnn(self, model, session, X, y, rnn_settings, words_to_idx):
        """Runs the model on the given data."""
        batch_size = rnn_settings['batch_size']
        num_batches = int(len(y)/batch_size)
        
        pathToLog = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project\log'
        writer = tf.summary.FileWriter(pathToLog)
        writer.add_graph(session.graph)
        len_y = len(y)
        
        # iterate over all epochs
        for epoch_i in range(self.number_of_epochs):
             print('Shuffle training data')
             np.random.seed(epoch_i)
             np.random.shuffle(X)
             np.random.seed(epoch_i)
             np.random.shuffle(y)
             print('#epoch: ', epoch_i) 
             # iterate over all batches
             for batch_i in range(num_batches):
                # get batches
                start = batch_i * batch_size
                end = min((batch_i + 1) * batch_size, len_y)
                
                feed_dict = {self.input_x: X[start:end],
                             self.input_y: y[start:end]}
                    
                _, loss, acc, pred, summary, \
                sentence_probability = session.run([self.train_op, self.loss, self.accuracy, 
                                                    self.predictions, self.merged, self.sentence_probability], 
                                                   feed_dict)
    
                perplexity = self.calculate_perplexity_new(sentence_probability, y[start:end], words_to_idx)
                print('Training: batch: ', batch_i , 'loss: ', np.sum(loss), \
                      'accuracy: ', acc, 'perplexity: ', np.nanmean(perplexity))
                writer.add_summary(summary, epoch_i)
                writer.flush()
        
        writer.close()

    def cleanOutput(self, X, word_to_idx):
        #create new dict for idx to word operation
        idx_to_word = {y:x for x,y in word_to_idx.items()}
        cleanX=[]
        
        for i in range(len(X)):
            print([idx_to_word[j] for j in X[i]])
            cleanScent=[]
            j = 1
            while True:
                cleanScent.append(idx_to_word[X[i][j]])
                if idx_to_word[X[i][j]]=='<eos>' or j==28: break
                j+=1
            cleanX.append(' '.join(cleanScent))
        return cleanX

    def generate_words_greedily(self, model, session, X, words_to_idx):
            """predict the next word of sentence, given the previous words
                load the trained model for this task 1.2"""
            
            Xorig_clean = self.cleanOutput(X, words_to_idx)
            
            for i in range(len(X)):#iterate over allscentences
                #set eos pointer to eos index
                p_eos = np.argwhere(np.array(X[i])==words_to_idx['<eos>'])[0][0] # 2 is eos but would be better using the dict
                while True:
                    #compute predictions
                    feed_dict = {self.input_x: np.array(X[i]).reshape((1,29)),
                                 self.input_y: np.array(X[i]).reshape((1,29))} # input_y is not needed
                                        
                    prediction, sentence_probability = session.run([self.predictions, self.sentence_probability], feed_dict)
                    
                    lastpred = prediction[0,p_eos-1]
                    X[i][p_eos]=lastpred
                    
                    p_eos += 1
                    if lastpred == words_to_idx['<eos>'] or p_eos==29: break
            
            #postprocess X
            Xclean = self.cleanOutput(X, words_to_idx)
            self.create_submission_file(Xorig_clean, task='originalX')
            self.create_submission_file(Xclean, task='continuation')


def main(): 
    """ main_train.py executes task 1.A, 1.B, 1.C and 1.2 """           
   
    # define the rnn with LSTM cell
    rnn_settings = {
        'sentence_length' : 30-1,        
        'batch_size' : 64, 
        'embedding_size' : 100, 
        'lstm_size' : 512,     # 1024 for Task C
        'lstm_size_down_projected': 512,
        'vocabulary_size' : 20000,
        'learning_rate' : 0.001, # default
        'number_of_epochs' : 2,
        'clip_gradient' : 5.0,
        'Training_mode': True,
        'Task': 'A'
        }

    training_mode = rnn_settings['Training_mode']
    task = rnn_settings['Task']
    
    pathMain = r'C:\Users\mauro\Desktop\CAS\_Natural_Language_Understanding\Project'
    pathData = os.path.join(pathMain, 'data')
    pathGraph = os.path.join(pathMain, 'graph')

    # set paths    
    pathToEmbedding = os.path.join(pathData,'wordembeddings-dim100.word2vec')
    trainFile = os.path.join(pathData, 'sentences.train')
    testFile =  os.path.join(pathData, 'sentences_test.new')
    contFile = os.path.join(pathData, 'sentences.continuation')
    
    # proprocess the train, validation and sentence continuation data
    train_X, train_Y, words_to_idx, \
    word_dict = preprocessing(pathToFile = trainFile, 
                              mode = 'training', 
                              words_to_idx = None, 
                              word_dict = None)
  
    eval_X, eval_Y, words_to_idx, \
    word_dict = preprocessing(pathToFile = testFile, 
                              mode = 'test', 
                              words_to_idx = words_to_idx, 
                              word_dict = word_dict)

    scent_cont_X, _, words_to_idx, \
    word_dict = preprocessing(pathToFile = contFile, 
                             mode = '1.2', 
                             words_to_idx = words_to_idx, 
                              word_dict = word_dict)  
    
    # create rnn graph
    rnn_train = RNN_class(rnn_settings)
    
    # Launch the graph
    with tf.Session() as session:

        if training_mode:    
            
            if task =='A':
                saver = tf.train.Saver()
                # Initialize the variables 
                session.run(tf.global_variables_initializer())
                # train the model
                rnn_train.train_rnn(rnn_train, session, train_X, train_Y, rnn_settings, words_to_idx)
                # export the trained meta-graph
                saver.save(session, os.path.join(pathGraph, 'modelA.ckpt'))
            
            if task =='B':
                saver = tf.train.Saver()
                # Initialize the variables 
                session.run(tf.global_variables_initializer())
                # load embeddings
                embedding_matrix= tf.get_variable(name="embedding", \
                                  initializer = tf.random_uniform([20000, 100], -0.1, 0.1))

                load_embedding(session = session, 
                               vocab = words_to_idx, 
                               emb = embedding_matrix, 
                               path = pathToEmbedding,
                               dim_embedding = rnn_settings['embedding_size'],
                               vocab_size = rnn_settings['vocabulary_size'])
                
                # train the model
                rnn_train.train_rnn(rnn_train, session, train_X, train_Y, rnn_settings, words_to_idx)
                # export the trained meta-graph
                saver.save(session, os.path.join(pathGraph, 'modelB.ckpt'))
            
            if task == 'C':
                saver = tf.train.Saver()
                # Initialize the variables 
                session.run(tf.global_variables_initializer())
                # load embeddings
                embedding_matrix= tf.get_variable(name="embedding", \
                                  initializer = tf.random_uniform([20000, 100], -0.1, 0.1))

                load_embedding(session = session, 
                               vocab = words_to_idx, 
                               emb = embedding_matrix, 
                               path = pathToEmbedding,
                               dim_embedding = rnn_settings['embedding_size'],
                               vocab_size = rnn_settings['vocabulary_size'])
                
                # train the model
                rnn_train.train_rnn(rnn_train, session, train_X, train_Y, rnn_settings, words_to_idx)
                # export the trained meta-graph
                saver.save(session, os.path.join(pathGraph, 'modelC.ckpt'))
            
        else: # test mode
            
            if task=='A': # eval task A            
                saver = tf.train.Saver()
                saver.restore(session, os.path.join(pathGraph, 'modelA.ckpt'))
                print('model restored for task A')
                # validate the model
                rnn_train.test_rnn(rnn_train, session, eval_X, eval_Y, task, rnn_settings, words_to_idx)
            
            if task=='B': # eval task B            
                saver = tf.train.Saver()
                saver.restore(session, os.path.join(pathGraph, 'modelB.ckpt'))
                print('model restored for task B')
                # validate the model
                rnn_train.test_rnn(rnn_train, session, eval_X, eval_Y, task, rnn_settings, words_to_idx)
            
            if task == 'C': # eval task B            
                saver = tf.train.Saver()
                saver.restore(session, os.path.join(pathGraph, 'modelC.ckpt'))
                print('model restored for task C')
                # validate the model
                rnn_train.test_rnn(rnn_train, session, eval_X, eval_Y, task, rnn_settings, words_to_idx)
            
            if task=='1.2':
                # predict the next word word of sentence
                saver = tf.train.Saver()
                saver.restore(session, os.path.join(pathGraph, 'modelB.ckpt'))
                print('model C restored for task 1.2')
                # generate scentences
                rnn_train.generate_words_greedily(rnn_train, session, scent_cont_X, words_to_idx)
            
        
if __name__ == '__main__':
    # reset the built graph
    tf.reset_default_graph()
    # run main method
    main()            


