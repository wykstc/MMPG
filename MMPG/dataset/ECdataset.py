import numpy as np
import warnings
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import *
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import os
import os.path as osp
from utils import set_seed
set_seed(42)

class ECdataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split='train'
                 ):
        self.split = split
        self.root = root
        self.amino_acids = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'UNK': 20, 'X': 21, 'SEC': 22, 'PYL': 23, 'XAA': 24,
            'UNK25': 25
        }
        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(root, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        num_classes = len(ec_annotations)

        super(ECdataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])
        labellist = []
        for i in self.data["id"]:
            label = np.zeros((num_classes,)).astype(np.float32)
            if len(self.labels[i]) > 0:
                label[self.labels[i]] = 1.0
            labellist.append(label)
        self.data.y = torch.tensor(labellist)

        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]


    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name, self.split)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _normalize(self, tensor, dim=-1):
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_positions(self, structure):
        """Extracts atom coordinates from a PDB structure."""
        amino_types = []
        pos_n = []
        pos_ca = []
        pos_c = []
        pos_cb = []
        pos_g = []
        pos_d = []
        pos_e = []
        pos_z = []
        pos_h = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    # 跳过非标准氨基酸
                    if residue.get_resname() not in self.amino_acids:
                        continue

                    amino_types.append(self.amino_acids[residue.get_resname()])
                    atoms = {atom.get_name(): atom for atom in residue}

                    # 获取各原子的坐标
                    pos_n.append(self.get_atom_coord(atoms, 'N'))
                    pos_ca.append(self.get_atom_coord(atoms, 'CA'))
                    pos_c.append(self.get_atom_coord(atoms, 'C'))
                    pos_cb.append(self.get_atom_coord(atoms, 'CB'))

                    # 侧链原子
                    pos_g.append(self.get_sidechain_coord(atoms, ['CG', 'SG', 'OG', 'CG1', 'OG1']))
                    pos_d.append(self.get_sidechain_coord(atoms, ['CD', 'SD', 'CD1', 'OD1', 'ND1']))
                    pos_e.append(self.get_sidechain_coord(atoms, ['CE', 'NE', 'OE1']))
                    pos_z.append(self.get_sidechain_coord(atoms, ['CZ', 'NZ']))
                    pos_h.append(self.get_sidechain_coord(atoms, ['NH1']))

        # 转换为张量
        return (torch.tensor(amino_types),
                *[torch.tensor(pos, dtype=torch.float32) for pos in
                  [pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h]])

    def get_atom_coord(self, atoms, atom_name):
        """Returns the coordinate of a given atom or NaN if missing."""
        if atom_name in atoms:
            return atoms[atom_name].get_coord()
        return [np.nan, np.nan, np.nan]

    def get_sidechain_coord(self, atoms, atom_names):
        """Returns the coordinate of the first available atom in a list of possible side-chain atoms."""
        for name in atom_names:
            if name in atoms:
                return atoms[name].get_coord()
        return [np.nan, np.nan, np.nan]

    def compute_dihedrals(self, v1, v2, v3):
        """Computes torsion (dihedral) angles given three sequential vectors."""
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion

    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        """Computes side-chain embedding based on torsion angles between backbone and side-chain atoms."""
        v1 = pos_ca - pos_n
        v2 = pos_cb - pos_ca
        v3 = pos_g - pos_cb
        v4 = pos_d - pos_g
        v5 = pos_e - pos_d
        v6 = pos_z - pos_e
        v7 = pos_h - pos_z

        angles = [
            self.compute_dihedrals(v1, v2, v3),
            self.compute_dihedrals(v2, v3, v4),
            self.compute_dihedrals(v3, v4, v5),
            self.compute_dihedrals(v4, v5, v6),
        ]

        angles = torch.stack(angles, dim=1)
        return torch.cat((torch.sin(angles), torch.cos(angles)), 1)

    def bb_embs(self, X):
        """Computes backbone embedding from torsion angles along the backbone."""
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_dihedrals(u0, u1, u2)
        angle = F.pad(angle, [1, 2])
        angle = torch.reshape(angle, [-1, 3])
        return torch.cat([torch.cos(angle), torch.sin(angle)], 1)

    def protein_to_graph(self, pdb_file):
        """Parses a PDB file and converts the protein structure into graph-compatible data."""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        # Extract atomic coordinates
        amino_types, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_positions(
            structure)

        data = Data()

        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0
        data.side_chain_embs = side_chain_embs

        bb_coords = torch.stack((pos_n, pos_ca, pos_c), dim=1)
        bb_embs = self.bb_embs(bb_coords)
        bb_embs[torch.isnan(bb_embs)] = 0
        data.bb_embs = bb_embs

        data.x = amino_types.unsqueeze(1)
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c

        return data

    def get_pdb_chain_id(self,file_name):
        return file_name.split('_')[0]
    #
    def process(self):
        percent = 95
        test_set = set()
        if self.split == "test":
            with open(osp.join(self.root, "/EnzymeCommission/nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        self.level_idx = 1
        with open(osp.join(self.root, '/EnzymeCommission/nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        print("Reading the data")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_list = []

            for protein_id in tqdm(self.labels.keys()):
                if self.split == "test" and protein_id not in test_set:
                    continue

                try:
                    pdb_dir = osp.join(self.root, self.split)
                    pdb_file = None
                    for filename in os.listdir(pdb_dir):
                        if filename.startswith(protein_id) and filename.endswith('.pdb'):
                            pdb_file = osp.join(pdb_dir, filename)
                            break
                    if pdb_file is not None:
                        cur_protein = self.protein_to_graph(pdb_file)
                        cur_protein.id = protein_id

                        label = np.zeros(len(ec_annotations), dtype=np.float32)
                        indices = self.labels[protein_id].astype(int)
                        label[indices] = 1.0
                        cur_protein.y = torch.tensor(label)

                        if cur_protein.x is not None:
                            data_list.append(cur_protein)
                except Exception as e:
                    print(f"Error processing {protein_id}: {str(e)}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    print('Done!')

if __name__ == "__main__":
    for split in ['train', 'valid', 'test']:
        print('#### Now processing {} data ####'.format(split))
        ECdataset(root='/EnzymeCommission', split=split)
