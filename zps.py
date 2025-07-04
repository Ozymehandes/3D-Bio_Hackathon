import numpy as np
import ruptures as rpt


def get_protein_segments_old(emb_dict, max_bkps_per100aa=3):
    """
    Define the boundaries of protein segments using change point analysis.

    Parameters:
    emb_dict: a dictionary of UniProt protein IDs as keys and ProtT5 per-residue embeddings as values
    max_bkps_per100aa: an integer that represents the maximum number of boundaries per 100 amino acids in the protein

    Returns:
    protein_segments: a dictionary of Uniprot protein IDs as keys and a list of boundaries as values
    """

    protein_segments = {}

    # for each protein and embedding
    for protein, emb in emb_dict.items():
        n_boundaries = max(int(emb.shape[0]*max_bkps_per100aa/100),1)

        # try to get the breakpoints from change point analysis
        try:
            alg = rpt.Window(width=30, model='rbf', jump=1).fit(emb)
            # n_bkps is the maximum number of breakpoints allowed in this protein
            boundaries = alg.predict(n_bkps=n_boundaries)
            # convert boundaries to segments
            my_segments = [[0,boundaries[0]]]
            my_segments.extend([[boundaries[ii], boundaries[ii+1]] for ii in range(len(boundaries)-1)])
        # this algorithm can fail if the maximum number of breakpoints is unreasonable
        except:
            print("Failed", protein, "number of boundaries", n_boundaries)
            my_segments = "Failed"

        protein_segments[protein] = my_segments

    return protein_segments

def get_protein_segments(emb_dict, max_bkps_per100aa=3):
    """
    Define the boundaries of protein segments using change point analysis.

    Parameters:
    emb_dict: a dictionary of UniProt protein IDs as keys and ProtT5 per-residue embeddings as values
    max_bkps_per100aa: an integer that represents the maximum number of boundaries per 100 amino acids in the protein

    Returns:
    protein_segments: a dictionary of Uniprot protein IDs as keys and a list of boundaries as values
    """

    protein_segments = {}

    # for each protein and embedding
    for protein, emb in emb_dict.items():
        n_boundaries = max(int(emb.shape[0]*max_bkps_per100aa/100),1)

        # try to get the breakpoints from change point analysis
        try:
            alg = rpt.Window(width=30, model='rbf', jump=1).fit(emb)
            # n_bkps is the maximum number of breakpoints allowed in this protein
            boundaries = alg.predict(n_bkps=n_boundaries)
            # convert boundaries to segments
            my_segments = [[0,boundaries[0]]]
            my_segments.extend([[boundaries[ii], boundaries[ii+1]] for ii in range(len(boundaries)-1)])
        # this algorithm can fail if the maximum number of breakpoints is unreasonable
        except:
            print("Failed", protein, "number of boundaries", n_boundaries)
            my_segments = "Failed"

        protein_segments[protein] = my_segments

    return protein_segments

def get_protein_segment_embeddings(emb_dict, protein_segment_boundaries):
    """
    Calculates the segment embedding for each segment defined by the change point analysis

    Parameters:
    emb_dict: a dictionary of UniProt protein IDs as keys and ProtT5 per-residue embeddings as values
    protein_segment_boundaries: a dictionary of Uniprot protein IDs as keys and a list of boundaries as values

    Returns:
    protein_segment_embeddings: a dictionary of "ID start-stop" as keys and segment embeddings as values
        IDs are UniProt protein IDs, start and stop corresponds to the positions of the segment in the 
        original protein sequence, and segment embeddings are vectors of size 1x1024
    """

    protein_segment_embeddings = {}

    # for each protein with an embedding
    for protein, emb in emb_dict.items():

        # if the segmentation did not fail
        if protein_segment_boundaries[protein] != "Failed":

            # generate a "segment embedding" for each segment of the protein
            for segment in protein_segment_boundaries[protein]:
                seg_emb = np.mean(emb[segment[0]:segment[1],:], axis=0)
                key = protein+" "+str(segment[0])+"-"+str(segment[1])

                protein_segment_embeddings[key] = seg_emb

    return protein_segment_embeddings


