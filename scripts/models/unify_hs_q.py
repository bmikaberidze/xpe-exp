"""
python -m micm_nlp.models.scripts.peft.xpe.unify_hs_q
"""
import torch
import numpy as np
from micm_nlp.models.model import MODEL

HS_DIR = f'{MODEL.stor_path}/xlmr/FacebookAI|xlm-roberta-large/prompt_hidden_states'

TEMPLATE1 = 'hs_q_joshi5_{i}'
TEMPLATE2 = 'hs_q_joshi5_{i}_seed_6_11'

for i in range(25):
    name1 = TEMPLATE1.format(i=i)
    name2 = TEMPLATE2.format(i=i)
    path1 = f'{HS_DIR}/{name1}.pt'
    path2 = f'{HS_DIR}/{name2}.pt'
    
    # Load both
    data1 = torch.load(path1, map_location="cpu", weights_only=True)
    data2 = torch.load(path2, map_location="cpu", weights_only=True)
    
    hs1 = data1["prompt_hidden_states"]
    hs2 = data2["prompt_hidden_states"]
    
    # Merge: for each group, concatenate tensors
    merged_hs = {}
    all_keys = list(hs1.keys()) + [k for k in hs2.keys() if k not in hs1]
    for key in all_keys:
        parts = []
        if key in hs1:
            parts.append(hs1[key] if torch.is_tensor(hs1[key]) else torch.tensor(hs1[key]))
        if key in hs2:
            parts.append(hs2[key] if torch.is_tensor(hs2[key]) else torch.tensor(hs2[key]))
        merged_hs[key] = torch.cat(parts, dim=0)
    
    # Save merged with original name
    merged_path = f'{HS_DIR}/{name1}.pt'
    # But first rename original file1
    import os
    renamed_path1 = f'{HS_DIR}/{name1}_seed_1_6.pt'
    os.rename(path1, renamed_path1)
    
    torch.save({"prompt_hidden_states": merged_hs}, merged_path)
    print(f'[{i}] Merged {name1} + {name2} -> {merged_path}')