import numpy as np
def extract_grouped_hs_embeddings(hs_path) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract grouped hidden state embeddings from a file,
    returning merged embeddings and their labels.
    """
    import torch
    hs_model = torch.load(hs_path, map_location="cpu", weights_only=True)
    hs = hs_model["prompt_hidden_states"]
    print(f"[INFO] Hidden States: {len(hs)}")
    # exit()

    grouped_hs_embedds_list = []
    grouped_hs_labels_list = []
    for group_name, emb in hs.items():
        # print(group_name)
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()
        grouped_hs_embedds_list.append(emb)
        grouped_hs_labels_list.extend([group_name] * len(emb))

    grouped_hs_embedds = np.vstack(grouped_hs_embedds_list)
    grouped_hs_labels = np.array(grouped_hs_labels_list, dtype=object)
    return grouped_hs_embedds, grouped_hs_labels