import pandas as pd
import spacy
from collections import Counter
import gc
from tqdm import tqdm
import psutil

nlp = spacy.load('en_core_web_sm')

role_mappings = {
    "nsubj": "s",
    "dobj": "o",
    "pobj": "o"
}

def entity_grid(text):
    transitions = list()
    entities = list()
    sentences_counter = 0


    doc = nlp(text)
    sentences = [sent for sent in doc.sents]
    sentences_counter += len(sentences)

   
    for sent in sentences:
        dict_sentence = {}
        for token in sent:
          
            if token.pos_ in ["PROPN", "NOUN", "PRON"] and token.dep_ != "compound":
                if token.text not in dict_sentence:
                    token_role = role_mappings.get(token.dep_, "x")  
                    dict_sentence[token.text] = token_role
        entities.append(dict_sentence)


    for i in range(len(entities) - 1):
        for key, role_1 in entities[i].items():
            role_2 = entities[i + 1].get(key, "-")  
            transitions.append(f"{role_1}->{role_2}")

    
    count_transitions = Counter(transitions)
    weighted_transitions = {k: v / (sentences_counter - 1) for k, v in count_transitions.items()}

    return weighted_transitions


input_file = 'D:\\Project_2\\dataset\\consolidated_train.csv'
output_file = 'transitions_output.csv'

chunk_size = 1000

with open(output_file, 'w') as f:
    f.write("Essay_Index,Transition,Weight\n")  
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        
        chunk['Essay'] = chunk['Essay'].fillna('')
        chunk['Label'] = chunk['Label'].fillna('Unknown')


        required_columns = ['Essay', 'Label']
        for col in required_columns:
            if col not in chunk.columns:
                raise KeyError(f"The required column '{col}' is missing from the dataset.")

      
        for index, row in tqdm(chunk.iterrows(), total=chunk.shape[0], desc="Processing Essays", unit="essay"):
            essay_text = row["Essay"]

          
            weighted_transitions = entity_grid(essay_text)
            
          
            for transition, weight in weighted_transitions.items():
                f.write(f"{index},{transition},{weight}\n")

        
        gc.collect()

    
        print(f"Memory usage after processing chunk: {psutil.virtual_memory().percent}%")

print("Entity grid and transitions for the entire dataset have been successfully created and saved.")
