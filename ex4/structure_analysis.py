"""
Compute a ROC curve that discriminates positive vs. negative peptides
using the distance between peptide centres of mass (COM) and a
reference/native COM.
"""

import pathlib
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
import os
from plot import plot_roc_curve, plot_boxplot
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist


def load_structure(path):
    """Read a PDB or mmCIF file and return a Bio.PDB.Structure."""
    path = pathlib.Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, path)


# You don't need this function for the exercise, but might find it useful for the Hackathon
def get_peptide_interface_atoms(structure, pep_chain_id, prot_chain_id):
    """Get only the peptide atoms that are in the interface"""
    pep_atoms = np.array([atom for residue in structure[0][pep_chain_id] for atom in residue])
    prot_atoms = np.array([atom for residue in structure[0][prot_chain_id] for atom in residue])

    # Compute distances of all pep to all prot atoms
    dist = cdist([a.coord for a in pep_atoms], [a.coord for a in prot_atoms], metric="euclidean")
    mask = dist.min(axis=1) < 5
    # get only pep atoms with distance < 5A of some prot atom
    return [atom for atom, keep in zip(pep_atoms, mask) if keep]


def get_peptide_atoms(structure, chain_id):
    """Generator of atoms that belong to the chosen peptide chain."""
    chain = structure[0][chain_id]  # first model assumed
    atoms = [atom for residue in chain for atom in residue]
    return atoms


def get_mean_plddt(atoms):
    return np.mean([atom.get_bfactor() for atom in atoms])


def centre_of_mass(atoms):
    """Return 3-vector COM (in Å) given an iterable of Bio.PDB atoms."""
    coords = np.array([a.coord for a in atoms], dtype=np.float64)
    return coords.mean(axis=0)


def com_distance(com_1, com_2):
    """Distance (Å) between COM of model and of ref."""
    return np.linalg.norm(com_1 - com_2)


if __name__ == "__main__":
    ref_pdb_path = "structures/6cit.pdb"
    ref_peptide_chain_id = "D"
    positives_dir = "structures/af_positives"
    negatives_dir = "structures/af_negatives"
    af_peptide_chain_id = "B"
    # Get reference peptide center of mass
    ref_struct = load_structure(ref_pdb_path)
    ref_pep_atoms = get_peptide_atoms(ref_struct, ref_peptide_chain_id)
    ref_com = centre_of_mass(ref_pep_atoms)

    # Get positives and negatives distances
    dists, plddts, labels = [], [], []
    for cls, folder in ((1, positives_dir), (0, negatives_dir)):
        pdb_models = [os.path.join(folder, pdb) for pdb in os.listdir(folder) if pdb.endswith(".pdb") or pdb.endswith(".cif")]
        for model in tqdm(pdb_models):
            # Get peptide atoms
            model_struct = load_structure(model)
            model_pep_atoms = get_peptide_atoms(model_struct, af_peptide_chain_id)

            # Get peptide plddt
            pep_plddt = get_mean_plddt(model_pep_atoms)
            plddts.append(pep_plddt)

            # Get peptide center of mass distance from the reference center of mass
            model_com = centre_of_mass(model_pep_atoms)
            dists.append(com_distance(model_com, ref_com))

            labels.append(cls)

    dists, plddts, labels = np.array(dists), np.array(plddts), np.array(labels)
    dists = -dists  # we want larger score -> positive

    # Plot ROC and boxplot
    plot_roc_curve(labels, dists, out_file_path="com_roc_curve.png")
    plot_roc_curve(labels, plddts, out_file_path="plddt_roc_curve.png")

    plot_boxplot({"Positive Test": dists[labels == 1], "Negative Test": dists[labels == 0]}, out_file_path="com_boxplot.png")
    plot_boxplot({"Positive Test": plddts[labels == 1], "Negative Test": plddts[labels == 0]}, out_file_path="plddt_boxplot.png")