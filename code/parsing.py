import lxml.etree
import utils

# function to remove capital letters
lower_letter= lambda w: w.lower() if w else  ''



def parsing_datasets(path_input, path_output):
###############################################################################
# This function creats a JSON file from a XML file, saving, for each word of the sentences,
# the word, the lemma, the POS and the id
#
# Input:
#   path_input: path of the XML file
#   path_output: path of the JSON file
#    
# Output:
#   None 
###############################################################################
    list_dictionaries=[]
    
    for event, elem in lxml.etree.iterparse(path_input,tag='sentence'):
        
        dictionary_info={}
        id_=elem.attrib['id']
        
        # save the information for a single word
        info_words=[(lower_letter(child_1.text), lower_letter(child_1.get('lemma')), child_1.get('pos') , child_1.get('id')) 
                    for child_1 in elem.getchildren()]
            
        dictionary_info[id_]=info_words
        list_dictionaries.append(dictionary_info)
        
        elem.clear()
    
    utils.save_json(list_dictionaries,path_output)

    

