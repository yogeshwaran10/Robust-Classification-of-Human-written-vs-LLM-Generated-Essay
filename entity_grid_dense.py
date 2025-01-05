import pandas as pd
import numpy as np
from tqdm import tqdm
import torch 


file_path = r"D:\Project_2\transitions_output.csv"
df = pd.read_csv(file_path)

unique_transitions = sorted(df['Transition'].unique()) 
print("Unique Transitions:", unique_transitions)

essay_data = {}

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing transitions"):
    essay_index = row['Essay_Index']
    transition = row['Transition']
    weight = row['Weight']
    if essay_index not in essay_data:
        essay_data[essay_index] = {transition: weight}
    else:
        essay_data[essay_index][transition] = weight

dense_matrix = []
essay_indices = sorted(essay_data.keys())  

for essay_index in tqdm(essay_indices, desc="Constructing dense matrix"):
    row = [essay_data[essay_index].get(transition, 0) for transition in unique_transitions]
    dense_matrix.append(row)


dense_matrix_tensor = torch.tensor(dense_matrix, device='cuda', dtype=torch.float32)


output_path = r"D:\Project_2\dense_matrix.npy"
np.save(output_path, dense_matrix_tensor.cpu().numpy()) 
print(f"Dense matrix saved to {output_path}")

csv_output_path = r"D:\Project_2\dense_matrix.csv"
dense_df = pd.DataFrame(dense_matrix_tensor.cpu().numpy(), columns=unique_transitions, index=essay_indices)
dense_df.to_csv(csv_output_path)
print(f"Dense matrix saved to {csv_output_path}")

print(f"Dense matrix shape: {dense_matrix_tensor.shape}")
