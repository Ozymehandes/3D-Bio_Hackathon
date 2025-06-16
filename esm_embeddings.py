import torch
import esm


# All of ESM-2 pre-trained models by embedding size
ESM_MODELS_DICT = {320: esm.pretrained.esm2_t6_8M_UR50D,
                   480: esm.pretrained.esm2_t12_35M_UR50D,
                   640: esm.pretrained.esm2_t30_150M_UR50D,
                   1280: esm.pretrained.esm2_t33_650M_UR50D,
                   2560: esm.pretrained.esm2_t36_3B_UR50D,
                   5120: esm.pretrained.esm2_t48_15B_UR50D}


def get_esm_model(embedding_size=1280):
    """
    Retrieves a pre-trained ESM-2 model
    :param embedding_size: The ESM-2 model embedding size
    :return: esm_model, alphabet, batch_converter, device
    """

    if embedding_size not in ESM_MODELS_DICT:
        raise ValueError(f"ERROR: ESM does not have a trained model with embedding size of {embedding_size}.\n "
                         f"Please use one of the following embedding sized: {ESM_MODELS_DICT.keys()}")

    model, alphabet = ESM_MODELS_DICT[embedding_size]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"ESM model loaded to {device}")
    return model, alphabet, batch_converter, device


def get_esm_embeddings(pep_tuple_list, esm_model, alphabet, batch_converter, device, embedding_layer=33, sequence_embedding=True):
    """
    This function convert peptide sequence data into ESM sequence embeddings
    :param pep_tuple_list: peptide tuple list of format : [(name_1, seq_1), (name_2, seq_2), ...]
    :param esm_model: Pre-trained ESM-2 model
    :param alphabet: ESM-2 alphabet object
    :param batch_converter: ESM-2 batch_converter object
    :param device: GPU/CPU device
    :param embedding_layer: The desired embedding layer to get
    :param sequence_embedding: Whether to use a sequence embedding (default=True) or amino acid embedding
    :return: List of ESM-2 sequence/amino acids embeddings
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(pep_tuple_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations
    with torch.no_grad():
        results = esm_model(batch_tokens.to(device), repr_layers=[embedding_layer])
    token_representations = results["representations"][embedding_layer]

    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    representations = []
    for i, tokens_len in enumerate(batch_lens):
        embedding = token_representations[i, 1: tokens_len - 1]
        # Generate per-sequence representations via averaging
        if sequence_embedding:
            embedding = embedding.mean(dim=0)
        representations.append(embedding.cpu().numpy())

    return representations