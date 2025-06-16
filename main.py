import numpy as np
from neural_net import get_net_scores, train_net, prepare_loader, SimpleDenseNet
from pep_utils import load_peptide_data, get_peptide_distances, pep_train_test_split
from plot import plot_boxplot, plot_roc_curve, plot_esm_amino_acids_embeddings, plot_2dim_reduction
from esm_embeddings import get_esm_embeddings, get_esm_model
from clustering import kmeans_clustering, tsne_dim_reduction

def run(chosen_embedding_size=1280, chosen_embedding_layer=33, chosen_test_size=0.25):
    # Load all the peptide data
    print("Loading peptide data")
    positive_pep, negative_pep, doubt_lables = load_peptide_data()

    # Load the pre-trained ESM-2 model with the desired embedding size
    print("Loading ESM-2 model")
    model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)

    print("Getting ESM-2 amino acid embeddings for the first negative and positive peptides")
    # Get amino acids embedding of the first negative and first positive peptides
    positive_aa_embeddings = get_esm_embeddings(positive_pep[0:1], model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=False)[0]
    negative_aa_embeddings = get_esm_embeddings(negative_pep[0:1], model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=False)[0]
    print("Plotting heatmaps of the amino acid embedding")
    plot_esm_amino_acids_embeddings(positive_aa_embeddings, out_file_path="positive_heatmap.png")
    plot_esm_amino_acids_embeddings(negative_aa_embeddings, out_file_path="negative_heatmap.png")


    # Get the ESM-2 sequence embeddings from all the negative and positive peptides
    print("Getting the ESM-2 embeddings for all the peptide data")
    positive_esm_emb = get_esm_embeddings(positive_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=True)
    negative_esm_emb = get_esm_embeddings(negative_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=True)

    # Clustering and Dimensionality reduction
    print("Clustering using K-means clustering and reduce to dim=2 using TSNE")
    all_esm_embeddings =  negative_esm_emb + positive_esm_emb
    all_labels = [0] * len(negative_esm_emb) + [1] * len(positive_esm_emb)

    k_means_labels = kmeans_clustering(all_esm_embeddings, k=2)
    coords_2d = tsne_dim_reduction(all_esm_embeddings, dim=2)

    print("Plotting 2D dimensionality reduction by true labels and by K-means clustering")
    plot_2dim_reduction(coords_2d, [["N", "P"][i] for i in all_labels], out_file_path="2d_true_labels.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path="2d_k_means.png")


    # Split the data into train and test sets
    print("Splitting to train and test sets")
    positive_train, negative_train, is_doubt_train, positive_test, negative_test, is_doubt_test = pep_train_test_split(
        positive_esm_emb, negative_esm_emb, doubt_lables, test_size=chosen_test_size)

    print("Calculating euclidean distances of ESM-2 embeddings")
    # Calculate a score based on the log fold difference between distance to negative mean and positive mean
    positive_score, negative_score = simple_score(positive_train, negative_train, positive_test, negative_test)

    # Plot the results in a boxplot and in a ROC curve
    print("Plotting Baseline results")
    plot_boxplot({"Positive Test": positive_score, "Negative Test": negative_score}, out_file_path="baseline_boxplot.png")
    plot_roc_curve([0] * len(negative_score) + [1] * len(positive_score), np.concatenate([negative_score, positive_score]), out_file_path="baseline_roc_curve.png")

    # Train a simple neural network
    print("Training a neural network on the training set")
    # TODO: Select parameters for the network
    batch_size = 64
    epochs = 50
    lr = 1e-3
    hidden_dim = 128
    dropout = 0.2
    # Prepare a Dataloader and create model
    net_dataloader = prepare_loader(positive_train, negative_train, batch_size=batch_size)
    network = SimpleDenseNet(esm_emb_dim=chosen_embedding_size, hidden_dim=hidden_dim, dropout=dropout)
    trained_network = train_net(network, net_dataloader, num_epochs=epochs, lr=lr)

    print("Getting the trained network scores for the test set")
    positive_score = get_net_scores(trained_net=trained_network, esm_seq_embeddings=positive_test)
    negative_score = get_net_scores(trained_net=trained_network, esm_seq_embeddings=negative_test)

    print("Plotting Network results")
    plot_boxplot({"Positive Test": positive_score, "Negative Test": negative_score}, out_file_path="network_boxplot.png")
    plot_roc_curve([0] * len(negative_score) + [1] * len(positive_score), np.concatenate([negative_score, positive_score]), out_file_path="network_roc_curve.png")