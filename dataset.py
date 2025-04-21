import os
import torch
import requests
import numpy as np
from Bio.PDB import PDBParser
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric import data as DATA
from tqdm import tqdm


class TestbedDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset,
        pro,
        poc,
        y,
        smile_graph,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.pro = pro
        self.poc = poc
        self.y = y
        self.smile_graph = smile_graph

        # download raw PDBs if missing
        if not os.path.isdir(self.raw_dir):
            self.download()

        # load or process
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # we save each PDB as <pdbid>.pdb in raw_dir
        return [f"{name}.pdb" for name in self.smile_graph]

    @property
    def processed_file_names(self):
        return [f"{self.dataset}.pt"]

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        for pdbid in tqdm(self.smile_graph, desc="Downloading PDBs"):
            outpath = os.path.join(self.raw_dir, f"{pdbid}.pdb")
            if os.path.exists(outpath):
                continue
            url = f"https://files.rcsb.org/download/{pdbid}.pdb"
            r = requests.get(url)
            r.raise_for_status()
            with open(outpath, "w") as f:
                f.write(r.text)

    def process(self):
        parser = PDBParser(QUIET=True)
        data_list = []

        for name in tqdm(self.smile_graph, desc="Processing graphs"):
            # --- ligand graph as before ---
            c_size, feats, edge_idx = self.smile_graph[name]
            ligand = DATA.Data(
                x=torch.Tensor(feats),
                edge_index=torch.LongTensor(edge_idx).t().contiguous(),
                y=torch.FloatTensor([self.y[name]]),
            )
            ligand.c_size = torch.LongTensor([c_size])
            ligand.protein = torch.LongTensor([self.pro[name]])
            ligand.pocket = torch.LongTensor([self.poc[name]])
            ligand.pdbid = name

            # --- build structure graph from downloaded PDB ---
            pdb_path = os.path.join(self.raw_dir, f"{name}.pdb")
            coords, struct_eidx = self._build_structure_graph(parser, pdb_path)

            ligand.prot_str_x = torch.Tensor(coords)  # [N_res,3]
            ligand.prot_str_edge_index = torch.LongTensor(struct_eidx)  # [2, E]

            data_list.append(ligand)

        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _build_structure_graph(self, parser, pdb_file, cutoff=8.0):
        struct = parser.get_structure("X", pdb_file)[0]
        coords = []
        for chain in struct:
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_coord())
        coords = np.vstack(coords)  # (N,3)

        # fully connect within cutoff
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        src, dst = np.where((D <= cutoff) & (D > 0))
        edge_index = np.vstack((src, dst))
        return coords, edge_index
