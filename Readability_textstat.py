import pandas as pd
import textstat
from tqdm.auto import tqdm 

def readability_scores(text: str) -> dict:
    scores = {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
    }
    return scores

def process_csv_with_readability(input_csv: str, output_csv: str):
    
    try:
        # First try with UTF-8
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try Windows-1252 encoding
        df = pd.read_csv(input_csv, encoding='cp1252')
    
    processed_data = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        essay = row['Essay']  
        label = row['Label']  
        
        
        readability = readability_scores(essay)

        
        processed_entry = {
            "essay": essay,
            "label": label,
            **readability,
        }
        
        processed_data.append(processed_entry)

    
    processed_df = pd.DataFrame(processed_data)
    
    processed_df.to_csv(output_csv, index=False)

def main():
    input_csv_path = r"D:\Project_2\dataset\consolidated_train.csv"
    output_csv_path = r"D:\Project_2\dataset\processed_readability_data.csv"
    
    process_csv_with_readability(input_csv_path, output_csv_path)

if __name__ == "__main__":
    main()
