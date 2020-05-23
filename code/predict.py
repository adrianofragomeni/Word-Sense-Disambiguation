import creation_test_set as cts
import parsing
import os
import tensorflow as tf
import candidates_synsets as cs
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import ast
import utils

###############################################################################
#### GLOBAL VARIABLES
###############################################################################
MAX_LENGTH=260
BATCH_SIZE=64
vocab_fine= utils.load_pickle("../resources/Wordnet_sense_vocab.pickle")
vocab_wndomain=utils.load_pickle("../resources/Wndomain_vocab.pickle")
vocab_lexname= utils.load_pickle("../resources/lexnames_vocab.pickle")

input_path="../resources/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"
output_path="../resources/Test/pred_test.txt"
resources_path="../resources"

def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    
    path_json=os.path.join(resources_path,"Test",'test.json')
    path_model=os.path.join(resources_path,'model','model_finegrained')

    # parse the xml file
    parsing.parsing_datasets(input_path,path_json)
    # creates candidates and input file for the model
    input_,candidates, ids_candidates=create_batched_inputs(path_json)
    ids=take_id_istances(resources_path)
    final_prediction=[]

    with tf.Session() as sess:
        
        # load the model
        output_finegrained,_,_,inputs_, input_int, keep_prob= load_model_finegrained(sess,path_model)
        for batch_num in range(len(input_[0])):
            
            # do the prediction
            preds_fine= sess.run(output_finegrained,feed_dict={inputs_: input_[0][batch_num],input_int:input_[1][batch_num],keep_prob: 1.})
            final_prediction+=calculate_prediction(input_[1][batch_num],preds_fine,candidates[0][batch_num],ids_candidates[0],vocab_fine)

        combined=list(zip(ids,final_prediction))
        
        # save the prediction
        utils.save_txt(list(map(lambda word: word[0]+" "+ word[1],combined)),output_path)
        
    tf.reset_default_graph()
    pass


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    
    path_json=os.path.join(resources_path,"Test",'test.json')
    path_model=os.path.join(resources_path,'model','model_coarsegrained')

    # parse the xml file
    parsing.parsing_datasets(input_path,path_json)
    # creates candidates and input file for the model
    input_,candidates, ids_candidates=create_batched_inputs(path_json)
    ids=take_id_istances(resources_path)
    final_prediction=[]

    with tf.Session() as sess:

        # load the model
        output_wndomain,_,inputs_, input_int, keep_prob= load_model_coarsegrained(sess,path_model)
        for batch_num in range(len(input_[0])):
                
            # do the prediction
            preds_wndomain= sess.run(output_wndomain,feed_dict={inputs_: input_[0][batch_num],input_int:input_[1][batch_num],keep_prob: 1.})
            final_prediction+=calculate_prediction(input_[1][batch_num],preds_wndomain,candidates[1][batch_num],ids_candidates[1],vocab_wndomain)
        
        combined=list(zip(ids,final_prediction))

        # save the prediction    
        utils.save_txt(list(map(lambda word: word[0]+" "+ word[1],combined)),output_path)
        
    tf.reset_default_graph()
    pass


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    
    path_json=os.path.join(resources_path,"Test",'test.json')
    path_model=os.path.join(resources_path,'model','model_coarsegrained')

    # parse the xml file
    parsing.parsing_datasets(input_path,path_json)
    # creates candidates and input file for the model
    input_,candidates, ids_candidates=create_batched_inputs(path_json)
    ids=take_id_istances(resources_path)
    final_prediction=[]
    
    with tf.Session() as sess:

        # load the model
        _,output_lexname, inputs_, input_int, keep_prob= load_model_coarsegrained(sess,path_model)
        for batch_num in range(len(input_[0])):

            # do the prediction                
            preds_lexname= sess.run(output_lexname,feed_dict={inputs_: input_[0][batch_num],input_int:input_[1][batch_num],keep_prob: 1.})
            final_prediction+=calculate_prediction(input_[1][batch_num],preds_lexname,candidates[2][batch_num],ids_candidates[2],vocab_lexname)

        combined=list(zip(ids,final_prediction))
        
        # save the prediction    
        utils.save_txt(list(map(lambda word: word[0]+" "+ word[1],combined)),output_path)
        
    tf.reset_default_graph()
    pass



def load_model_finegrained(session,path_model):
###############################################################################
# This function loads the trained fine-grained model 
#
# Input:
#   session: session of tensorflow
#   path_model: the path of the model
#
# Output:
#   output_finegrained: tensor of the finegrained output
#   output_wndomain: tensor of the wndomain output
#   output_lexname: tensor of the lexname output
#   input_: tensor of the input sentences (string tensor)
#   input_w_ids: tensor of the input sentences (integer tensor)
#   keep_prob: tensor of the keep_prob
###############################################################################   
    
    # import the graph
    model= tf.train.import_meta_graph( os.path.join(path_model,"model.meta"))
    #restore the model
    model.restore(session, tf.train.latest_checkpoint(path_model))
    
    graph = tf.get_default_graph()
    
    # get some tensors from the graph
    input_ = graph.get_tensor_by_name('inputs:0')
    input_w_ids = graph.get_tensor_by_name('inputs_int:0')

    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    
    output_finegrained= graph.get_tensor_by_name('dense_finegrained/finegrained_output:0')
    output_wndomain= graph.get_tensor_by_name('dense_wndomain/wndomain_output:0')
    output_lexname=  graph.get_tensor_by_name('dense_lexname/lexname_output:0')

    return (output_finegrained, output_wndomain, output_lexname,input_, input_w_ids, keep_prob)

    

def load_model_coarsegrained(session,path_model):
###############################################################################
# This function loads the trained coarse-grained model 
#
# Input:
#   session: session of tensorflow
#   path_model: the path of the model
#
# Output:
#   output_wndomain: tensor of the wndomain output
#   output_lexname: tensor of the lexname output
#   input_: tensor of the input sentences (string tensor)
#   input_w_ids: tensor of the input sentences (integer tensor)
#   keep_prob: tensor of the keep_prob
###############################################################################   
    
    # import the graph
    model= tf.train.import_meta_graph( os.path.join(path_model,"model.meta"))
    #restore the model
    model.restore(session, tf.train.latest_checkpoint(path_model))
    
    graph = tf.get_default_graph()
    
    # get some tensors from the graph
    input_ = graph.get_tensor_by_name('inputs:0')
    input_w_ids = graph.get_tensor_by_name('inputs_int:0')

    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    
    output_wndomain= graph.get_tensor_by_name('dense_wndomain/wndomain_output:0')
    output_lexname=  graph.get_tensor_by_name('dense_lexname/lexname_output:0')

    return (output_wndomain, output_lexname,input_, input_w_ids, keep_prob)



def create_batched_inputs(path_json):
###############################################################################
# This function creates batches of sentences and batches of candidates for synsets,
# wndomain and lexname.
#
# Input:
#   path_json: path of the json file  
#
# Output:
#   : a list with batches of sentences (string and integer sentences)
#   : a list where there are the candidates for synsets,wndomain and lexname
#   : a list where there are the ids of the list of candidates 
###############################################################################       
    
    # list of padded sentences (string sentences) divided into batches
    X_test=check_length_sentences([sentence + [''] * (MAX_LENGTH - len(sentence)) for sentence in cts.create_test_sentence(path_json)[0]])
    X_test_batched=[X_test[pos:pos+BATCH_SIZE] for pos in range(0, len(X_test), BATCH_SIZE)]

    # list of padded sentences (ids sentences) divided into batches
    X_test_int=check_length_sentences([list(map(int,sentence.split())) for sentence in cts.create_test_sentence(path_json)[1]])
    X_test_int=pad_sequences(X_test_int, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    X_test_int_batched=[X_test_int[pos:pos+BATCH_SIZE] for pos in range(0, len(X_test_int), BATCH_SIZE)]

    # lists of padded candidate synsets sentences divided into batches
    X_candidate_synsets,ids_candidates_synsets=cs.create_candidates(path_json,type_=0)
    X_candidate_synsets=check_length_sentences(X_candidate_synsets)
    X_candidate_synsets=pad_sequences(X_candidate_synsets, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    X_candidate_synsets_batched=[X_candidate_synsets[pos:pos+BATCH_SIZE] for pos in range(0, len(X_candidate_synsets), BATCH_SIZE)]

    X_candidate_wndomain,ids_candidates_wndomain=cs.create_candidates(path_json,type_=1)
    X_candidate_wndomain=check_length_sentences(X_candidate_wndomain)
    X_candidate_wndomain=pad_sequences(X_candidate_wndomain, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    X_candidate_wndomain_batched=[X_candidate_wndomain[pos:pos+BATCH_SIZE] for pos in range(0, len(X_candidate_wndomain), BATCH_SIZE)]
    
    X_candidate_lexname,ids_candidates_lexname=cs.create_candidates(path_json,type_=2)
    X_candidate_lexname=check_length_sentences(X_candidate_lexname)
    X_candidate_lexname=pad_sequences(X_candidate_lexname, truncating='pre', padding='post', maxlen=MAX_LENGTH)
    X_candidate_lexname_batched=[X_candidate_lexname[pos:pos+BATCH_SIZE] for pos in range(0, len(X_candidate_lexname), BATCH_SIZE)]

    
    return [X_test_batched,X_test_int_batched], [X_candidate_synsets_batched,X_candidate_wndomain_batched,X_candidate_lexname_batched], [ids_candidates_synsets,ids_candidates_wndomain,ids_candidates_lexname]
    
    
    
def calculate_prediction(batch_x,preds,batch_candidates,dictionary_ids,vocabulary_output):
###############################################################################
# This function calculates the prediction of the model: given a word if the word is 
# an instance word a prediction is returned.
# When the lemma of the word is not in the training lemmas, the most predominant sense
# is returned as prediction
#
# Input:
#   batch_x:  batch of sentences 
#   preds: predictions associated to the current batch of sentences
#   batch_candidates: candidates ids associated to the current batch
#   dictionary_ids: dictionary of the ids associated to the list of candidates
#   vocabulary_output: ids associated to the labels of a task
#
# Output:
#   final_pred: final prediction of the target words
###############################################################################       
    
    final_pred=[]
    # creates dictionary for the conversion from ids to string
    converter_ids={value:key for key,value in dictionary_ids.items()}
    converter_output={value:key for key,value in vocabulary_output.items()}
    
    for pos_sentence in range(len(batch_x)):
        
        # for loop considering only the words and removing the padding
        for pos_word in range(np.count_nonzero(batch_x[pos_sentence])):
            
            # check if the word doesn't have any label and skip it
            if batch_candidates[pos_sentence][pos_word]== 1:
                continue
            # do the prediction for the other words
            else:
                # the try-except is used for the backoff strategy (except part)
                try:
                    # take the list of candidates ids associated to the current word
                    list_ids=ast.literal_eval(converter_ids[batch_candidates[pos_sentence][pos_word]])
                    # take the id whose value in the prediction list is the highest one
                    prediction=list_ids[np.argmax(preds[pos_sentence,pos_word,list_ids])]
                    prediction=converter_output[prediction]
                except:
                    #take the predominant sense of the lemma 
                    prediction=converter_ids[batch_candidates[pos_sentence][pos_word]]
                    
                final_pred.append(prediction)
    
    return final_pred



def take_id_istances(resources_path):
###############################################################################
# This function creats a list of words, which will have a prediction from the model.
# The ids are taken from the Json file.
#
# Input:
#   resources_path: the path of the resources folder
#
# Output:
#   : list of words' ids
###############################################################################    
    
    path=os.path.join(resources_path,"Test","test.json")
    ids=[]
    
    for sentence in utils.load_json(path):
        id_istances=list(map(lambda info: info[3],list(sentence.values())[0]))
        ids+=id_istances
    
    return list(filter(None,ids))



def check_length_sentences(list_):
###############################################################################
# Given a list of words, this function checks if the length of the list is 
# larger than the maximum length used in the training phase. If it is true 
# the sentence is split in subsentences
#
# Input:
#   list_: list of words
#
# Output:
#   new_list: new list of words, based on sublists if the original list is larger than the threshold
###############################################################################        
    
    new_list=[]
    
    for elem in list_:
        
        if len(elem)<=MAX_LENGTH:
            new_list.append(elem)
        else:
            new_list+=[elem[pos:pos+MAX_LENGTH] for pos in range(0,len(elem),MAX_LENGTH)]
    
    return new_list
    
    

