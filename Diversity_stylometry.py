import pandas as pd
import json
import spacy
from lexical_diversity import lex_div as ld
from tqdm.auto import tqdm


NLP = spacy.load("en_core_web_sm")
LATIN = ["i.e.", "e.g.", "etc.", "c.f.", "et", "al."]
FEATS = ["ttr", "root_ttr", "log_ttr", "maas_ttr", "msttr", "mattr", "hdd", "mtld", "mtld_ma_wrap", "mtld_ma_bid"]

def style_features_processing(text: str) -> tuple:
    doc = NLP(text)
    pos_tokens = []
    shape_tokens = []

    for word in doc:
        if word.is_punct or word.is_stop or word.text in LATIN:
            pos_target = word.text
            shape_target = word.text
        else:
            pos_target = word.pos_
            shape_target = word.shape_

        pos_tokens.append(pos_target)
        shape_tokens.append(shape_target)

    return " ".join(pos_tokens), " ".join(shape_tokens)

def lex_div_feats_extraction(text: str, features: list = FEATS) -> dict:
    preprocessed = preprocess(text) 
    result = {}
    
    for feature in features:
        result[feature] = getattr(ld, feature)(preprocessed)
    
    return result

def preprocess(text: str) -> list:
    doc = NLP(text)
    return [f"{w.lemma_}_{w.pos_}" for w in doc if not w.pos_ in ["PUNCT", "SYM", "SPACE"]]

def process_csv_with_stylistics(input_csv: str, output_csv: str):
    
    df = pd.read_csv(input_csv)
    
    
    processed_data = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        essay = row['Essay']  
        label = row['Label']  

        
        pos, shape = style_features_processing(essay)
        diversity_features = lex_div_feats_extraction(essay)
        processed_entry = {
            "essay": essay,
            "label": label,
            "pos": pos,
            "shape": shape,
            **diversity_features,
        }
        
        processed_data.append(processed_entry)

    
    processed_df = pd.DataFrame(processed_data)
    
    
    processed_df.to_csv(output_csv, index=False)

def main():
    input_csv_path = r"D:\Project_2\dataset\consolidated_train.csv"
    output_csv_path = r"D:\Project_2\dataset\processed_stylistic_data.csv"
    
    process_csv_with_stylistics(input_csv_path, output_csv_path)

if __name__ == "__main__":
    main()
