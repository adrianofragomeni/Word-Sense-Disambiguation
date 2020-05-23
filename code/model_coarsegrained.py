import tensorflow as tf
import utils 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import ast
import creation_test_set as cts
import os
import candidates_synsets as cs
from operator import itemgetter 
import random
import tensorflow_hub as hub

###############################################################################
#### GLOBAL VARIABLES
###############################################################################

random.seed('abc')
vocab_wndomain={value:key for key,value in utils.load_pickle("../resources/Wndomain_vocab.pickle").items()}
vocab_lexname= {value:key for key,value in utils.load_pickle("../resources/lexnames_vocab.pickle").items()}
vocab_POS= {value:key for key,value in utils.load_pickle("../resources/POS_vocab.pickle").items()}

path_dev_json="../resources/Dev/semeval2007.json"
path_train_json="../resources/Train/semcor+omsti.json"
path_dev_labels="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt"

MAX_LENGTH=260
EPOCHS=30
BATCH_SIZE=16
ITERATIONS =1200
KEEP_PROB=0.9
DEV_ITERATIONS=15
HIDDEN_SIZE= 256
LEARNING_RATE= 0.001
LEN_LEXNAME=len(vocab_lexname)
LEN_WNDOMAIN=len(vocab_wndomain)
LEN_POS=len(vocab_POS)



def model():
###############################################################################
# This function defines the tensorflow graph of the WSD model.
#
# Input:
#   None
#    
# Output:
#   train: optimizer of the sum of the losses
#   train_wndomain: optimizer of the wndomain loss
#   train_lexname: optimizer of the lexname loss
#   train_POS: optimizer of the pos loss
#   loss: tensor of the total loss
#   loss_wndomain: tensor of the wn_domain loss
#   loss_lexname: tensor of the lexname loss
#   loss_POS: tensor of the Pos loss
#   output_wndomain: tensor of the wn_domain output
#   output_lexname: tenspr of the lexname output
#   input_: tensor of the input (string sentences)
#   input_w_ids: tensor of the input (ids sentences)
#   y_wndomain: tensor of the wn_domain labels
#   y_lexnames: tensor of the lexname labels
#   y_POS: tensor of the Pos labels
#   keep_prob: tensor of the dropout 
###############################################################################     
    
    # define input tensors
    input_= tf.placeholder(tf.string, shape=[None,None], name='inputs')
    input_w_ids=tf.placeholder(tf.int32, shape=[None,None], name='inputs_int')
    seq_length=tf.count_nonzero(input_w_ids, axis=-1, name='length',dtype=tf.int32)
    y_wndomain=tf.placeholder(tf.int32, shape=[None,None], name='wndomain_label')
    y_lexnames=tf.placeholder(tf.int32, shape=[None,None], name='lexnames_label')
    y_POS=tf.placeholder(tf.int32, shape=[None,None], name='POS_label')
    keep_prob=tf.placeholder(tf.float32, shape=[], name='keep_prob')
    elmo = hub.Module("../resources/elmo", trainable=False)

    # define Elmo embeddings node
    with tf.variable_scope("Elmo_embeddings"):
        
        embeddings = elmo(inputs={"tokens": input_,"sequence_len": seq_length},
                      signature="tokens",
                      as_dict=True)["elmo"]
        
    # define BiLSTM node
    with tf.variable_scope('BiLSTM'):

        cell_fw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE,initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,  input_keep_prob=keep_prob, 
                                            output_keep_prob= keep_prob, state_keep_prob=keep_prob)

        cell_bw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE,initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob, 
                                            output_keep_prob= keep_prob, state_keep_prob=keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                          cell_bw, 
                                                          embeddings,
                                                          sequence_length=seq_length,
                                                          dtype=tf.float32)

        # Concat the forward and backward outputs 
        outputs = tf.concat(outputs,2)


    def attention(i,out_bilstm,input_ids, W_att,c_mat):
    ###############################################################################
    # This function calculates and updates the weights matrix by adding the weights
    # of each sentence of the current batch 
    #
    # Input:
    #   i: batch number
    #   out_bilstm: output of the BiLSTM
    #   input_ids: input matrix with ids 
    #   W_att: parameter vector
    #   c_mat: weights matrix
    #   
    # Output:
    #   : updated batch number
    #   out_bilstm: output of the BiLSTM
    #   input_ids: input matrix with ids 
    #   W_att: parameter vector
    #   : updated weights matrix
    ###############################################################################     
        
        # take the output of the BiLSTM of the batch i
        mask=out_bilstm[i]
        # take the ids sentence of the batch i
        sentence_ids=input_ids[i]
        
        # mask to exclude the padding from the attention scores
        mask_attention=tf.not_equal(sentence_ids,0)
        h_masked = tf.boolean_mask(mask, mask_attention)

        u = tf.matmul(tf.tanh(h_masked), W_att)

        # apply softmax to the u vector     
        attention_score = tf.nn.softmax(u)
        attention_score=tf.math.l2_normalize(attention_score)
        
        # calculate attention score
        c = tf.reduce_sum(tf.multiply(h_masked, attention_score), 0)  
        
        # return the result of the first iteration
        if c_mat==None:
            return c
        
        c=tf.expand_dims(c,0)        
        return tf.add(i,1),out_bilstm,input_ids, W_att,tf.concat([c_mat,c],0)


    # define attention node
    with tf.variable_scope("attention"):
        
        output_shape=outputs.get_shape()[2].value
        batch_size=tf.shape(outputs)[0]

        # define parameter vector
        W= tf.get_variable("W_att", shape=[output_shape, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=0))

        # calculate the attention weights for all the sentences of the current batch
        c= tf.expand_dims(attention(0,outputs,input_w_ids,W,None), 0)
        i = tf.constant(1)
        cond = lambda i,out,inp_mask,W,c: tf.less(i, batch_size)
        _,_,_, _,attention_out= tf.while_loop(cond, attention,[i,outputs,input_w_ids, W,c],shape_invariants=[i.get_shape(),outputs.get_shape(),input_w_ids.get_shape(), W.get_shape(),tf.TensorShape([None, HIDDEN_SIZE*2])])

        attention_out = tf.expand_dims(attention_out, 1)
        c_final = tf.tile(attention_out, [1, MAX_LENGTH, 1])
        
        # concat output BiLSTM with attention weights
        concat_attention = tf.concat([c_final, outputs],2)
                
        
    # define output nodes
    with tf.variable_scope("dense_POS"):
        
        logits_POS = tf.layers.dense(inputs=concat_attention, units=LEN_POS)
        output_POS= tf.identity( logits_POS, name= 'POS_output')
    
    
    with tf.variable_scope("dense_lexname"):
        
        logits_lexname = tf.layers.dense(inputs=concat_attention, units=LEN_LEXNAME)
        output_lexname= tf.identity( logits_lexname, name= 'lexname_output')
        
    with tf.variable_scope("dense_wndomain"):
        
        logits_wndomain = tf.layers.dense(inputs=concat_attention, units=LEN_WNDOMAIN)
        output_wndomain= tf.identity( logits_wndomain, name= 'wndomain_output')
        
        
    # define loss nodes
    with tf.variable_scope('loss_POS'):
        
        loss_POS=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_POS,labels=y_POS)
        # mask for the POS loss
        mask_pos=tf.greater_equal(y_POS, 2)
        losses_pos_POS=tf.boolean_mask(loss_POS,mask_pos)
        
        loss_POS=tf.reduce_mean(losses_pos_POS, name="POS_loss")

        
    with tf.variable_scope('loss_wndomain'):
        
        loss_wndomain=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_wndomain,labels=y_wndomain)
        # mask for the wndomain loss
        mask_pos=tf.greater_equal(y_wndomain, 3)
        losses_pos_wndomain=tf.boolean_mask(loss_wndomain,mask_pos)
        
        loss_wndomain=tf.reduce_mean(losses_pos_wndomain, name="wndomain_loss")
    
    with tf.variable_scope('loss_lexname'):
        
        loss_lexname=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_lexname,labels=y_lexnames)
        # mask for the lexname loss
        mask_pos=tf.greater_equal(y_lexnames, 3)
        losses_pos_lexname=tf.boolean_mask(loss_lexname,mask_pos)
        
        loss_lexname=tf.reduce_mean(losses_pos_lexname, name="lexname_loss")
        
    with tf.variable_scope("total_loss"):
        # sum all the losses
        loss =  loss_wndomain + loss_lexname + loss_POS
        loss= tf.identity( loss, name= 'loss')

            
    with tf.variable_scope("train_total"):
        train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

                
    return (train, loss, loss_wndomain, loss_lexname,loss_POS, output_wndomain, output_lexname, input_,input_w_ids, y_wndomain, y_lexnames,y_POS, keep_prob)
        


def create_input_Dev():
###############################################################################
# This function creates the developtment input sentences, the labels and sentences
# with the candidate synsets
#
# Input:
#   None
#    
# Output:
#   : list with sentences, labels and candidate synsets
#   : list with ids of the candidate synsets
###############################################################################  

    # list of padded sentences (string sentences)    
    X_train=[sentence + [''] * (MAX_LENGTH - len(sentence)) if len(sentence)<=MAX_LENGTH else sentence[:MAX_LENGTH] for sentence in cts.create_test_sentence(path_dev_json)[0]]

    # list of padded sentences (ids sentences)
    X_train_int=[list(map(int,sentence.split())) for sentence in cts.create_test_sentence(path_dev_json)[1]]
    X_train_int=pad_sequences(X_train_int, truncating='pre', padding='post', maxlen=MAX_LENGTH)

    # lists of padded labels    
    Y_POS=[list(map(int,label.split())) for label in cts.create_dev_POS_labels(path_dev_json)]
    Y_POS=pad_sequences(Y_POS, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    
    Y_wndomain=[list(map(int,label.split())) for label in cts.create_dev_labels(path_dev_json,path_dev_labels,type_=1)]
    Y_wndomain=pad_sequences(Y_wndomain, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    
    Y_lex=[list(map(int,label.split())) for label in cts.create_dev_labels(path_dev_json,path_dev_labels,type_=2)]
    Y_lex=pad_sequences(Y_lex, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    
    # lists of padded candidate synsets sentences
    X_candidate_wndomain,ids_candidates_wndomain=cs.create_candidates(path_dev_json,type_=1)
    X_candidate_wndomain=pad_sequences(X_candidate_wndomain, truncating='pre', padding='post', maxlen=MAX_LENGTH)

    X_candidate_lexname,ids_candidates_lexname=cs.create_candidates(path_dev_json,type_=2)
    X_candidate_lexname=pad_sequences(X_candidate_lexname, truncating='pre', padding='post', maxlen=MAX_LENGTH)

    return [X_train,X_train_int, Y_wndomain, Y_lex, Y_POS,X_candidate_wndomain,X_candidate_lexname], [ids_candidates_wndomain,ids_candidates_lexname]



def create_input_Train():
 ###############################################################################
# This function creates the training input sentences, the labels and sentences
# with the candidate synsets
#
# Input:
#   None
#    
# Output:
#   : list with sentences, labels and candidate synsets
#   : list with ids of the candidate synsets
###############################################################################   
 
    # list of padded sentences (string sentences)    
    X_train=[sentence.split() + [''] * (MAX_LENGTH - len(sentence.split())) if len(sentence.split())<=MAX_LENGTH else sentence.split()[:MAX_LENGTH] for sentence in utils.load_txt("../resources/Train/semcor_omsti_string.txt")]

    # list of padded sentences (ids sentences)
    X_train_int=[list(map(int,sentence.split())) for sentence in utils.load_txt("../resources/Train/semcor_omsti.txt")]
    X_train_int=pad_sequences(X_train_int, truncating='pre', padding='post', maxlen=MAX_LENGTH)

    # lists of padded labels        
    Y_POS=[list(map(int,label.split())) for label in utils.load_txt("../resources/Train/POS.txt")]
    Y_POS=pad_sequences(Y_POS, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    
    Y_wndomain=[list(map(int,label.split())) for label in utils.load_txt("../resources/Train/Wndomain.txt")]
    Y_wndomain=pad_sequences(Y_wndomain, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    
    Y_lex=[list(map(int,label.split())) for label in utils.load_txt("../resources/Train/lexnames.txt")]
    Y_lex=pad_sequences(Y_lex, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    
    # lists of padded candidate synsets sentences
    X_candidate_wndomain,ids_candidates_wndomain=cs.create_candidates(path_train_json,type_=1)
    X_candidate_wndomain=pad_sequences(X_candidate_wndomain, truncating='pre', padding='post', maxlen=MAX_LENGTH)

    X_candidate_lexname,ids_candidates_lexname=cs.create_candidates(path_train_json,type_=2)
    X_candidate_lexname=pad_sequences(X_candidate_lexname, truncating='pre', padding='post', maxlen=MAX_LENGTH)

    return [X_train,X_train_int, Y_wndomain, Y_lex, Y_POS,X_candidate_wndomain,X_candidate_lexname], [ids_candidates_wndomain,ids_candidates_lexname]
            
        
def trainer():
###############################################################################
# This function trains the model and tests it on the development set
#
# Input:
#   None
#    
# Output:
#   None
###############################################################################     
    
    print("======="*10)
    print("\nCreate Inputs Training set")
    inputs_,ids_candidates=create_input_Train()
    
    print("======="*10)
    print("\nCreate Inputs Dev set")
    dev_,ids_candidates_dev=create_input_Dev()

    data_gen=batch_creation(*inputs_,BATCH_SIZE)
    data_gen_dev=batch_creation(*dev_,BATCH_SIZE)
    
    # return the tensor from the model to feed them with the inputs
    train, loss, loss_wndomain, loss_lexname, loss_POS,output_wndomain, output_lexname, input_,input_w_ids, y_wndomain, y_lexnames, y_POS, keep_prob=model()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())

        print("======="*10)        
        print("Start Training")
        print("======="*10)

        for epoch in range(EPOCHS):
 
            # define the initial value for the losses and accuracies (for train and development sets)
            epoch_loss = 0.
            epoch_loss_wndomain = 0.
            epoch_loss_lexname = 0.
            epoch_loss_POS=0.

            epoch_acc_wndomain= 0.
            epoch_acc_lexname= 0.
            
            dev_loss = 0.
            dev_loss_wndomain = 0.
            dev_loss_lexname = 0.
            dev_loss_POS=0.
            
            dev_acc_wndomain= 0.
            dev_acc_lexname= 0.

            # train the model           
            for _ in range(ITERATIONS): 

                batch_x,batch_labels,batch_candidates= next(data_gen)
                    
                _,loss_val,loss_val_wndomain,loss_val_lexname,loss_val_POS,preds_wndomain,preds_lexname= sess.run([train, loss, loss_wndomain, loss_lexname,loss_POS,output_wndomain,output_lexname],
                                                                                       feed_dict={input_: batch_x[0],input_w_ids: batch_x[1],y_wndomain: batch_labels[0], y_lexnames: batch_labels[1], y_POS: batch_labels[2],keep_prob: KEEP_PROB})

                # sum the losses and the accuracies every iteration                
                epoch_loss += loss_val
                epoch_loss_wndomain  += loss_val_wndomain
                epoch_loss_lexname += loss_val_lexname
                epoch_loss_POS += loss_val_POS

                epoch_acc_wndomain+=calculate_accuracy(batch_x[1],preds_wndomain,batch_labels[0],batch_candidates[0],ids_candidates[0])
                epoch_acc_lexname+=calculate_accuracy(batch_x[1],preds_lexname,batch_labels[1],batch_candidates[1],ids_candidates[1])
                
            # update the losses and the accuracies every epoch          
            epoch_loss /= ITERATIONS
            epoch_loss_wndomain  /= ITERATIONS
            epoch_loss_lexname  /= ITERATIONS
            epoch_loss_POS /= ITERATIONS

            
            epoch_acc_wndomain /= ITERATIONS
            epoch_acc_lexname /= ITERATIONS

               
            summary_train_loss = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=epoch_loss)])
            writer.add_summary(summary_train_loss, global_step=epoch)
            
            print("\nEpoch", epoch + 1,"\nTrain Loss: {:.4f}".format(epoch_loss),"\nTrain Loss Wndomain: {:.4f}".format(epoch_loss_wndomain),"\nTrain Loss Lexname: {:.4f}".format(epoch_loss_lexname),"\nTrain Loss POS: {:.4f}".format(epoch_loss_POS),
                  "\nAccuracy WNDomain: {:.4f}".format(epoch_acc_wndomain),"\nAccuracy Lexname: {:.4f}".format(epoch_acc_lexname))
        

            # test the model on the development test
            for _ in range(DEV_ITERATIONS):
                
                batch_x,batch_labels,batch_candidates= next(data_gen_dev)
                    
                loss_val,loss_val_wndomain,loss_val_lexname,loss_val_POS,preds_wndomain,preds_lexname= sess.run([loss, loss_wndomain, loss_lexname,loss_POS,output_wndomain,output_lexname], 
                                                                                                                                          feed_dict={input_: batch_x[0],input_w_ids: batch_x[1],y_wndomain: batch_labels[0],
                                                                                                                                                     y_lexnames: batch_labels[1],y_POS: batch_labels[2],keep_prob: 1.})

                # sum the losses and the accuracies every iteration           
                dev_loss += loss_val
                dev_loss_wndomain  += loss_val_wndomain
                dev_loss_lexname += loss_val_lexname
                dev_loss_POS += loss_val_POS

                dev_acc_wndomain+=calculate_accuracy(batch_x[1],preds_wndomain,batch_labels[0],batch_candidates[0],ids_candidates_dev[0])
                dev_acc_lexname+=calculate_accuracy(batch_x[1],preds_lexname,batch_labels[1],batch_candidates[1],ids_candidates_dev[1])


            # update the losses and the accuracies every epoch                
            dev_loss/= DEV_ITERATIONS
            dev_loss_wndomain  /= DEV_ITERATIONS
            dev_loss_lexname  /= DEV_ITERATIONS
            dev_loss_POS  /= DEV_ITERATIONS

            dev_acc_wndomain /= DEV_ITERATIONS
            dev_acc_lexname /= DEV_ITERATIONS

             
            summary_dev_loss = tf.Summary(value=[tf.Summary.Value(tag='dev_loss', simple_value=dev_loss)])
            writer.add_summary(summary_dev_loss, global_step=epoch)
            
            
            print("\nEpoch", epoch + 1,"\nDevelopment Loss: {:.4f}".format(dev_loss),"\nDevelopment Loss WnDomain: {:.4f}".format(dev_loss_wndomain),"\nDevelopment Loss Lexname: {:.4f}".format(dev_loss_lexname),"\nDevelopment Loss POS: {:.4f}".format(dev_loss_POS),
                  "\nAccuracy WNDomain: {:.4f}".format(dev_acc_wndomain),"\nAccuracy Lexname: {:.4f}".format(dev_acc_lexname))
    
            print("======="*10) 
            
        print("Saving Model!")
        print("======="*10)

        saver.save(sess, os.path.join('..','resources','model','model_coarsegrained','model'))

    
    
def batch_creation(X,X_w_id,Y_wndomain,Y_lexname, Y_POS,candidate_wndomain,candidate_lexname,batch_size):
###############################################################################
# This function is a batch generator, it creates infinite batches which are passed
# to model to train it
#   
# Input:
#   X: list of sentences (string sentences)
#   X_w_id: list of sentences (ids sentences)
#   Y_wndomain: labels of the Wordnet domain
#   Y_lexname: labels of the lexname
#   Y_POS: labels of the POS tagging
#   candidate_wndomain: list of candidate Wndomain for each words
#   candidate_lexname: list of candidate lexname for each words
#   batch_size: size of the batch
#
# Output:
#   :batch of sentences
#   :batch of the labels 
#   :batch of candidates
###############################################################################
    
    while True:
        # random permutation of the elements
        perm = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size

            yield ([list(itemgetter(*list(perm[start:end]))(X)),X_w_id[perm[start:end]]],[Y_wndomain[perm[start:end]], Y_lexname[perm[start:end]], Y_POS[perm[start:end]]], 
                   [list(itemgetter(*list(perm[start:end]))(candidate_wndomain)),list(itemgetter(*list(perm[start:end]))(candidate_lexname)) ])
 
    
    
def calculate_accuracy(batch_x,preds,batch_labels,batch_candidates,dictionary_ids):
###############################################################################
# This function calculates the accuracy only for the istances words which have labels
#   
# Input:
#   batch_x: current batch
#   preds: prediction of the current batch
#   batch_labels: labels of the current batch
#   batch_candidates: candidates output of the current batch
#   dictionary_ids: dictionary with the ids of the candidate outputs
#
# Output:
#   accuracy: accuracy obtained on the current batch
###############################################################################

    final_pred=[]
    converter_ids={value:key for key,value in dictionary_ids.items()}
    
    for pos_sentence in range(len(batch_x)):

        for pos_word in range(np.count_nonzero(batch_x[pos_sentence])):
            
            # condition to exclude the no-istance words form the accuracy calculation
            if batch_candidates[pos_sentence][pos_word]== 1:
                continue
            else:
                
                # try-except statement to exclude all the istance words without labels from the accuray calculation
                try:
                    list_ids=ast.literal_eval(converter_ids[batch_candidates[pos_sentence][pos_word]])
                    prediction=list_ids[np.argmax(preds[pos_sentence,pos_word,list_ids])]
                    label=batch_labels[pos_sentence][pos_word]
                except:
                    continue
                
                final_pred.append(prediction==label)
                
    accuracy=sum(final_pred)/len(final_pred)
    
    return accuracy