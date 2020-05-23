import json
import pickle


def save_json(obj,path_file):
###############################################################################
# This function saves a list as a json file
#
# Input:
#   obj: list which will be saved
#   path_file: path where the file will be saved
#    
# Output:
#   None 
###############################################################################    
    with open(path_file,'w') as output:
        json.dump(obj,output)



def load_json(path_file):
###############################################################################
# This function loads a json file
#
# Input:
#   path_file: path where the file is saved
#    
# Output:
#   file: the loaded json file
###############################################################################     
    with open(path_file,'r') as output:
        file = json.load(output) 
        
    return file



def load_txt(path_file):
###############################################################################
# This function loads a txt file
#
# Input:
#   path_file: path where the file is saved
#    
# Output:
#   file: the loaded txt file 
###############################################################################     
    with open(path_file, 'r') as output:
        file = output.read().splitlines()
        
    return file
    


def save_txt(obj,path_file):
###############################################################################
# This function saves an element as a txt file
#
# Input:
#   obj: element which will be saved
#   path_file: path where the file will be saved
#    
# Output:
#   None 
###############################################################################     
    with open(path_file,'w') as output:
        output.write("\n".join(element for element in obj))



def save_pickle(obj,path_file):
###############################################################################
# This function saves an element as a pickle file
#
# Input:
#   obj: element which will be saved
#   path_file: path where the file will be saved
#    
# Output:
#   None 
###############################################################################     
    with open(path_file,"bw") as output:
        pickle.dump(obj,output)
        
        
        
def load_pickle(path_file):
###############################################################################
# This function loads a pickle file
#
# Input:
#   path_file: path where the file is saved
#    
# Output:
#   file: the loaded pickle file 
###############################################################################     
    with open(path_file, 'rb') as output:
        file = pickle.load(output)
        
    return file
        
       
 
def reading_Wordnet_matching():
###############################################################################
# This function creats a dictionary which matches the Babelnet_id and the Wordnet_id
#
# Input:
#   : None
#    
# Output:
#   Wordnet_match: dictionary of Babelnet_id and Wordnet_id 
###############################################################################
    with open('../resources/babelnet2wordnet.tsv','r') as file:

        Wordnet_match = {}
        
        for line in file:
            matching=line.split()
            # take the first element as GT when there is more than one label
            if len(matching)>2:
                
                for wn_id in matching[1:]:
                    Wordnet_match[wn_id]=matching[0]
            else:
                Wordnet_match[matching[1]]=matching[0]
        
    return Wordnet_match
    


def reading_matching(path):
###############################################################################
# This function creats a dictionary with the matching between Babelnet_ids and 
# Wn_domain or Lexname domain.
#
# Input:
#   path: path of the matching files
#    
# Output:
#   match: dictionary of the matches
###############################################################################
    
    with open(path,'r') as file:

        match = {line.split()[0]:line.split()[1] for line in file}
        
    return match
    