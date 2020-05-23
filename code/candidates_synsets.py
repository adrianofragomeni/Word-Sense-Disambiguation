import utils
import itertools
from nltk.corpus import wordnet as wn


###############################################################################
#### GLOBAL VARIABLES
###############################################################################
Wordnet_match=utils.reading_Wordnet_matching()
Wndomain_match=utils.reading_matching('../resources/babelnet2wndomains.tsv')
lexname_match=utils.reading_matching('../resources/babelnet2lexnames.tsv')
vocab_fine= utils.load_pickle("../resources/Wordnet_sense_vocab.pickle")
vocab_wndomain=utils.load_pickle("../resources/Wndomain_vocab.pickle")
vocab_lexname= utils.load_pickle("../resources/lexnames_vocab.pickle")
vocabulary_lemmas= utils.load_pickle("../resources/vocabulary_lemmas.pickle")
wordnet_pos = {"NOUN": wn.NOUN,"VERB": wn.VERB,"ADJ": wn.ADJ,"ADV": wn.ADV}



def create_candidates(path,type_):
###############################################################################
# This function returns a list of lists with all the candidate outputs
# (Babelnet_ids, Wn_domain or Lexname) associated to each word in each sentence.
# The instance words will have a list with all the candidate outputs, whereas the
# other words will have a label equal to one.
# 
# Input:
#   path: path of the json file
#   type_: type of the candidate
#
# Output:
#   data: list of lists with candidates
#   ids_candidates: dictionary with the ids of the candidate lists
############################################################################### 
    
    # define the type of candidate
    type_candidate={0:take_synsets,1:take_wndomain,2:take_lexname}
    
    data=utils.load_json(path)
    
    # creates the candidates for each sentence 
    data=list(map(type_candidate[type_],data))
    
    # remove all the None elements to create a dictionary with the ids of the list of the candidates
    merged =set(filter(None,list(itertools.chain(*data))))
    ids_candidates={value:key for key, value in dict(enumerate(merged,2)).items() if value !=1}
    
    data=list(map(lambda sentence: [ids_candidates.get(word, word) for word in sentence], data))

    return data, ids_candidates



def take_synsets(dictionary):
###############################################################################
# This function returns a sentence with 1 if the word is not an instance word
# or a list with the candidate synsets for an instance word (considering only those synsets
# already seen in the training)
# 
# Input:
#   dictionary: information for one sentence
#
# Output:
#   sentence: list of candidate synsets for the current sentence
############################################################################### 
    
    sentence=[]

    for word in list(dictionary.values())[0]:
        
        # append 1 if the word is not an instance word
        if word[3]==None:
            sentence.append(1)
        
        # save the most predominant sense of a lemma if the lemma is not in the trained lemma vocabulary (backoff strategy)
        elif word[1] not in vocabulary_lemmas.keys():
            lemma=word[1]
            s=wn.synsets(lemma)[0]
            sentence.append(str(Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]))
        
        # create a list with all the candidate synsets of the current word
        else:
            lemma=word[1]
            pos= wordnet_pos[word[2]]
            candidates=list(set([int(vocab_fine[Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]])
                                        for s in wn.synsets(lemma,pos) if Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()] in vocab_fine.keys()]))
            
            # apply the backoff strategy also for those words whose list of candidates is empty
            if len(candidates)==0:

                s=wn.synsets(lemma)[0]
                sentence.append(str(Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]))

            else:
    
                sentence.append(str(candidates))

    return sentence



def take_wndomain(dictionary):
###############################################################################
# This function returns a sentence with 1 if the word is not an instance word
# and a list with the candidate wndomains for an instance word (considering only those wndomains
# already seen in the training)
# 
# Input:
#   dictionary: information for one sentence
#
# Output:
#   sentence: list of candidate wndomain for one sentence
############################################################################### 
    
    sentence=[]

    for word in list(dictionary.values())[0]:

        # append 1 if the word is not an instance word
        if word[3]==None:
            sentence.append(1)
            
        # save the most predominant sense of a lemma if the lemma is not in the trained lemma vocabulary (backoff strategy)
        elif word[1] not in vocabulary_lemmas.keys():
            lemma=word[1]
            s=wn.synsets(lemma)[0]
            try:
                sentence.append(str(Wndomain_match[Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]]))
            except:
                sentence.append("factotum")
                
        # create a list with all the candidate wndomains of the current word
        else:
            list_synsets=[]
            lemma=word[1]
            pos= wordnet_pos[word[2]]
            
            for s in wn.synsets(lemma,pos):
                
                if Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()] in vocab_fine.keys():
                    
                    try:
                        list_synsets.append(int(vocab_wndomain[Wndomain_match[Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]]]))
                    except:
                        list_synsets.append(int(vocab_wndomain["factotum"]))
                        
            candidates=list(set(list_synsets))
            
            # apply the backoff strategy also for those words whose list of candidates is empty
            if len(candidates)==0:
                s=wn.synsets(lemma)[0]
                try:
                    sentence.append(str(Wndomain_match[Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]]))
                except:
                    sentence.append("factotum")

            else:
                sentence.append(str(candidates))

    return sentence



def take_lexname(dictionary):
###############################################################################
# This function returns a sentence with 1 if the word is not an instance word
# and a list with the candidate lexnames for an instance word (considering only those lexnames
# already seen in the training)
# 
# Input:
#   dictionary: information for one sentence
#
# Output:
#   sentence: list of candidate lexname for one sentence
############################################################################### 
    
    sentence=[]

    for word in list(dictionary.values())[0]:

        # append 1 if the word is not an instance word
        if word[3]==None:
            sentence.append(1)
            
        # save the most predominant sense of a lemma if the lwmma is not in the trained lemma vocabulary (backoff strategy)
        elif word[1] not in vocabulary_lemmas.keys():
            lemma=word[1]
            s=wn.synsets(lemma)[0]
            sentence.append(str(lexname_match[Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]]))
           
        # create a list with all the candidate lexnames of the current words
        else:
            lemma=word[1]
            pos= wordnet_pos[word[2]]
            candidates=list(set([int(vocab_lexname[lexname_match[Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]]])
                                        for s in wn.synsets(lemma,pos) if Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()] in vocab_fine.keys()]))
    
            # apply the backoff strategy also for those words whose list of candidates is empty
            if len(candidates)==0:
                s=wn.synsets(lemma)[0]
                sentence.append(str(lexname_match[Wordnet_match["wn:" + str(s.offset()).zfill(8) + s.pos()]]))
            else:
                sentence.append(str(candidates))
                
    return sentence