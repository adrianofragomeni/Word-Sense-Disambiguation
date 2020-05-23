import utils
from nltk.corpus import wordnet as wn
from functools import partial

###############################################################################
#### GLOBAL VARIABLES
###############################################################################
Wordnet_match=utils.reading_Wordnet_matching()
Wndomain_match=utils.reading_matching('../resources/babelnet2wndomains.tsv')
lexname_match=utils.reading_matching('../resources/babelnet2lexnames.tsv')
vocab_fine= utils.load_pickle("../resources/Wordnet_sense_vocab.pickle")
vocab_wndomain=utils.load_pickle("../resources/Wndomain_vocab.pickle")
vocab_lexname= utils.load_pickle("../resources/lexnames_vocab.pickle")
vocab_POS= utils.load_pickle("../resources/POS_vocab.pickle")
vocabulary= utils.load_pickle("../resources/vocabulary.pickle")
vocabulary_lemmas= utils.load_pickle("../resources/vocabulary_lemmas.pickle")



def create_labels_words(path_input,save_gt=False):
###############################################################################
# This function, given the gold file, creates a dictionary of labels for each 
# ambiguous words (babelnet id, Wndomain, Lexname) and  if save_gt is True, 
# this function saves the ground truth
#
# Input:
#   path: path of the gold file
#   save_gt: if True, the ground truth is saved, otherwise no
#
# Output:
#   dict_sensekey: dictionary of labels
############################################################################### 
    
    sense_keys=[sensekey.split() for sensekey in utils.load_txt(path_input)]
    dict_sensekey={}
    
    for list_info in sense_keys:
        
        # take the synset from the sense key
        synset =  wn.lemma_from_key(list_info[1]).synset()
        # take the wordnet id from the sense key
        wn_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
        bn_id=Wordnet_match[wn_id]
        
        try:
            dict_sensekey[list_info[0]]=[bn_id,Wndomain_match[bn_id],lexname_match[bn_id]] 
            
        # add the factotum label to all the words which don't have a wndomain label
        except:
            dict_sensekey[list_info[0]]=[bn_id,"factotum",lexname_match[bn_id]]
          
    # save the ground truth
    if save_gt:
        
        name_words=list(dict_sensekey.keys())
        gt_words=list(dict_sensekey.values())
        combined=list(zip(name_words,gt_words))
        
        utils.save_txt(list(map(lambda word: word[0]+" "+ word[1][0],combined)),"../resources/Test/fine_grained_gt.txt")
        utils.save_txt(list(map(lambda word: word[0]+" "+ word[1][1],combined)),"../resources/Test/wndomain_gt.txt")
        utils.save_txt(list(map(lambda word: word[0]+" "+ word[1][2],combined)),"../resources/Test/lexname_gt.txt")
        
        return None

    return dict_sensekey



def sentence_from_dictionaries(dictionary,training_sentence):
###############################################################################
# This funcion, given a dictionary with all the information about a sentence,
# returns a list with the words of a current sentence or a string of labels with
# <WORD> for the unlabelled words, with <UNSEEN> for the labelled word 
# whose lemma wasn't in the training set or the id for the labelled words
#
# Input:
#   dictionary: information for one sentence
#   training_sentence: if True a list of words is created, otherwise a string with ids of the labelled words
#   
# Output:
#   : list of words or a string with ids of the labelled words
###############################################################################

    # list with all the information of each words    
    info_words=list(dictionary.values())[0]
    # list with the information for the whole sentence
    info_sentence=list(zip(*info_words))
    
    if training_sentence:
        return [info_sentence[0][pos] for pos in range(len(info_sentence[0]))]
    
    else:
        
        labels=[]
        
        for pos in range(len(info_sentence[0])):
        
            # check if the current word does not have a label
            if info_sentence[3][pos]==None:
                labels.append("<WORD>")
            
            # check if the lemma of the current word is in the lemma vocabulary
            elif info_sentence[1][pos] not in vocabulary_lemmas.keys():
                labels.append("<UNSEEN>")
                
            else:
                labels.append(info_sentence[3][pos])
            
        return " ".join(labels)



def label_POS(dictionary):
###############################################################################
# This function creates a sentence with the POS labels for a single test sentence
#
# Input:
#   dictionary: information for one sentence
#
# Output:
#   : sentence of POS labels
############################################################################### 

    # list with all the information of each words        
    info_words=list(dictionary.values())[0]
    # list with the information for the whole sentence
    info_sentence=list(zip(*info_words))
    
    return " ".join([info_sentence[2][pos] if info_sentence[3][pos]!=None else "<UNSEEN>" for pos in range(len(info_sentence[0]))])



def create_dev_POS_labels(path_input_json):
###############################################################################
# This function returns a list with sentences with POS labels using 
# the POS vocabulary created from the training set to assign the id POS
#
#
# Input:
#   path_input_json: path of the json file
#
# Output:
#   data: list with POS labels 
############################################################################### 
    
    data=utils.load_json(path_input_json)
    data=[sentence.split() for sentence in list(map(label_POS,data))]
    # if a POS is not in the POS vocabulary, put 1 as unseen POS
    data=list(map(lambda sentence: " ".join([vocab_POS.get(word,"1") for word in sentence]) ,data))

    return data
    


def create_dev_labels(path_input_json,path_input_labels, type_):    
###############################################################################
# This function returns a list with sentences with a specific label, using the 
# vocabulary of the considered label created from the training set to assign the id labels
#
# Input:
#   path_input_json: path of the json file 
#   path_input_labels: path gold file
#   type_: it is a number used to choose the type of label
#
# Output:
#   : list with labels
###############################################################################    
    
    vocabularies={0:vocab_fine,1:vocab_wndomain,2:vocab_lexname}
    
    # create a list with sentences of the considered labels for all the test set
    data=utils.load_json(path_input_json)
    dictionary=create_labels_words(path_input_labels)
    data=list(map(partial(sentence_from_dictionaries,training_sentence=False),data))
    
    sentences=[]
    
    for sentence in data:
        
        single_sentence=[]
        
        for word in sentence.split():
            
            # insert the current word
            if type(dictionary.get(word, word))!= list:
                single_sentence.append(word)
                
            # insert the corrispondent label for the current word
            else:
                single_sentence.append(str(dictionary.get(word, word)[type_]))
                
        sentences.append(single_sentence)
    # if a word is in the training set but we don't have that synset, put 1 as unseen label
    return list(map(lambda sentence: ' '.join(str(vocabularies[type_].get(word, "1")) for word in sentence), sentences))
    


def create_test_sentence(path_input):
###############################################################################
# This function returns two lists: one with all the test sentences of strings and
# the other list with all the test sentences of integers, adding 1 as id for all the
# words which never occured in the training set
#
# Input:
#   path_input: path of the json file
#    
# Output:
#   : tuple with a list with sentences of strings and a list with sentences of integers
###############################################################################
    
    data=utils.load_json(path_input)
    data=[sentence for sentence in list(map(partial(sentence_from_dictionaries,training_sentence=True),data))]
    # exchange the words in the test set with the ids from the vocabulary created during the training, put one for the OOV
    data_w_ids=list(map(lambda sentence: " ".join([vocabulary.get(word, word) if word in vocabulary.keys() else "1" for word in sentence]) ,data))

    return (data, data_w_ids)
