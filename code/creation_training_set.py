import utils
from nltk.corpus import wordnet as wn
from functools import partial
import itertools
import os
from collections import Counter


###############################################################################
#### GLOBAL VARIABLES
###############################################################################
name_training_file={0:["Wordnet_sense.txt","Wordnet_sense_vocab.pickle"],1:["Wndomain.txt","Wndomain_vocab.pickle"],2:["lexnames.txt","lexnames_vocab.pickle"],3:["POS.txt","POS_vocab.pickle"]}
Wordnet_match=utils.reading_Wordnet_matching()
Wndomain_match=utils.reading_matching('../resources/babelnet2wndomains.tsv')
lexname_match=utils.reading_matching('../resources/babelnet2lexnames.tsv')
path_dataset="../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI/"
path_train="../resources/Train/"



def create_labels_words(path=path_dataset+"semcor+omsti.gold.key.txt"):
###############################################################################
# This function, given the gold file, creates a dictionary of labels (babelnet id, Wndomain, Lexname)
# for each ambiguous words
#
# Input:
#   path: path of the gold file
#
# Output:
#   dict_sensekey: dictionary of labels
############################################################################### 
    
    sense_keys=[sensekey.split() for sensekey in utils.load_txt(path)]
        
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
    
    return dict_sensekey



def label_POS(dictionary):
###############################################################################
# This function creates a sentence with the POS labels for a single training sentence
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

    
    
def create_training_POS_labels(path=os.path.join(path_train,"semcor+omsti.json"),type_=3):
###############################################################################
# This function creates a txt file with sentences with POS labels
# and a vocabulary with all the seen POS labels. 
#
# Input:
#   path: path of the json file
#   type_: it is a label used to select the type of the output
#
# Output:
#   None 
############################################################################### 

    # create a list with sentences of POS labels for all the training set     
    data=utils.load_json(path)
    data=[sentence.split() for sentence in list(map(label_POS,data))]

    # create the vocabulary of seen POS labels, adding the ids for the padding and the unseen labels 
    dictionary_POS={value:str(key) for key,value in dict(enumerate(set(itertools.chain.from_iterable(data))-{"<UNSEEN>"},2)).items()}
    dictionary_POS["<PAD>"]="0"
    dictionary_POS["<UNSEEN>"]="1"

    # exchange strings with their ids
    data=list(map(lambda sentence: " ".join([dictionary_POS.get(word,word) for word in sentence]) ,data))

    utils.save_txt(data,path_train+name_training_file[type_][0])
    utils.save_pickle(dictionary_POS,"../resources/"+name_training_file[type_][1])



def create_training_labels(type_,path=os.path.join(path_train,"semcor+omsti.json")):    
###############################################################################
# This function creates a txt file with sentences with a specific label
# and a vocabulary with all the seen labels. 
#
# Input:
#   type_: it is a laabel used to choose the type of label
#   path: path of the json file
#
# Output:
#   None 
###############################################################################    
    
    # create a list with sentences of the considered labels for all the training set
    dictionary=create_labels_words()
    data=utils.load_json(path)
    data=list(map(partial(sentence_from_dictionaries,training_sentence=False),data))
    
    sentences=[]
    labels=set()
    
    for sentence in data:
        
        single_sentence=[]
        
        for word in sentence.split():
            
            # insert the current word
            if type(dictionary.get(word, word))!= list:
                single_sentence.append(word)
            
            # insert the corrispondent label for the current word
            else:
                single_sentence.append(str(dictionary.get(word, word)[type_]))
                labels.add(str(dictionary.get(word, word)[type_]))
                
        sentences.append(single_sentence)
    
    # create the vocabulary of seen labels, adding the ids for the padding, the unseen labels and the unlabelled words
    vocabulary={value:key for key, value in dict(enumerate(labels,3)).items()}
    vocabulary["<PAD>"]="0"
    vocabulary["<UNSEEN>"]="1"
    vocabulary["<WORD>"]="2"
    
    # exchange strings with ids
    sentences=list(map(lambda sentence: ' '.join(str(vocabulary.get(word, word)) for word in sentence), sentences))
    
    utils.save_txt(sentences,path_train+name_training_file[type_][0])
    utils.save_pickle(vocabulary,"../resources/"+name_training_file[type_][1])
    
    
    
def sentence_from_dictionaries(dictionary,training_sentence):
###############################################################################
# This funcion, given a dictionary with all the information about a sentence,
# returns a list with the words of the current sentence or a string 
# with <WORD> for the unlabelled words and the id for the labelled words, this 
# will be used to create a training ground truth to train the model
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
        return " ".join(["<WORD>" if info_sentence[3][pos]==None else info_sentence[3][pos] for pos in range(len(info_sentence[0]))])
    
    

def create_training_sentence(path=os.path.join(path_train,"semcor+omsti.json")):
###############################################################################
# This function creates a txt file with all the training sentences of strings,
# a txt file with all the training sentences of integers and two vocabularies, 
# one of them with all the seen lemma and the other with all the seen words
# in the training set.
#
# Input:
#   path: path of the json file
#    
# Output:
#   None 
###############################################################################
    data=utils.load_json(path)
    
    # create the dictionaries of lemmas and words 
    dictionary_train=create_training_dictionary(data)
    dictionary_lemmas=create_dictionary_lemmas(data)
    
    data=[sentence for sentence in list(map(partial(sentence_from_dictionaries,training_sentence=True),data))]
    
    print(max([len(sentence) for sentence in data]))
    # exchange the words in the training set with their ids, put one for the OOV
    data_w_ids=list(map(lambda sentence: " ".join([dictionary_train.get(word,"1") for word in sentence]) ,data))

    utils.save_txt([" ".join(sentence) for sentence in data],path_train+"semcor_omsti_string.txt")
    utils.save_txt(data_w_ids,path_train+"semcor_omsti.txt")
    
    utils.save_pickle(dictionary_train,"../resources/vocabulary.pickle")
    utils.save_pickle(dictionary_lemmas,"../resources/vocabulary_lemmas.pickle")



def finder(dictionary,flag_None,flag_lemma):
###############################################################################
# This function creates a list of elements (lemmas or words) for labelled or
# unlabelled words
#
# Input:
#   dictionary: information for one sentence
#   flag_None: if True information of the labelled words are taken, otherwise of the unlabelled words
#   flag_lemma: if True lemmas are considered, otherwise words are considered
#
# Output:
#   : list of words or lemmas of the training set
###############################################################################
    
    # list with all the information of each words
    info_words=list(dictionary.values())[0]
    # list with the information for the whole sentence
    info_sentence=list(zip(*info_words))

    if flag_None:
        return [info_sentence[int(flag_lemma)][pos] for pos in range(len(info_sentence[0])) if info_sentence[3][pos]==None]
    
    else:
        return [info_sentence[int(flag_lemma)][pos] for pos in range(len(info_sentence[0])) if info_sentence[3][pos]!=None]



def create_training_dictionary(data,min_threshold=3):
###############################################################################
# This function creates a dictionary of words for the training set, all the words
# that occur less than a min_thresold are not considered in the vocabulary.
# The previous process is applied only to the words whitout the ground truth (unambiguous words)
#
# Input:
#   data: ljson file with training sentences
#   min_threshold: threshold to set whether a word can be considered an OOV or not
#    
# Output:
#   dictionary_train: dictionary of training words 
###############################################################################
    
    # separate the words with the label and without it
    data_nolabels=[sentence for sentence in list(map(partial(finder,flag_None=True,flag_lemma=False),data))]
    data_labels=[sentence for sentence in list(map(partial(finder,flag_None=False,flag_lemma=False),data))]
    
    # Count the number of occurences for each word without a label
    Count_data_nolabels=Counter(itertools.chain.from_iterable(data_nolabels))
    # remove the words which occur less than the threshold
    Count_data_nolabels=[word for word in Count_data_nolabels if Count_data_nolabels[word] >= min_threshold]
    
    # merge the list of labelled and unlabelled words
    total_words=Count_data_nolabels+list(itertools.chain.from_iterable(data_labels))
    
    # creation of the dictionary, adding the OOV and PAD ids
    dictionary_train={value:str(key) for key,value in dict(enumerate(set(total_words),2)).items()}
    dictionary_train["<PAD>"]="0"
    dictionary_train["<OOV>"]="1"

    return dictionary_train
    


def create_dictionary_lemmas(data):
###############################################################################
# This function creates a dictionary of lemmas for the training set, considering
# only those lemmas associated to the ambiguous words
#
# Input:
#   data: ljson file with training sentences
#    
# Output:
#   dictionary_lemmas: dictionary of training lemmas 
###############################################################################
    
    data_labels=[sentence for sentence in list(map(partial(finder,flag_None=False,flag_lemma=True),data))]
    dictionary_lemmas={value:str(key) for key,value in dict(enumerate(set(itertools.chain.from_iterable(data_labels)))).items()}
    
    return dictionary_lemmas
    
    