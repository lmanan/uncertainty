from torch.utils.data import Dataset
import logging
from pathlib import Path
from typing import List, Optional, Callable
import numpy as np
import pickle
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CSVDatasetInfer(Dataset):
    def __init__(
        self,
        tracker_csv_dir_name: str,
        embedding_csv_dir_name: str,
        candidate_graph_dir_name: str,
        gt_csv_dir_name: str,
        run: str | None = None,
        filter_fn: Optional[Callable[[Path], bool]] = None,
    ):
        self.tracker_csv_dir = Path(tracker_csv_dir_name)
        self.embedding_csv_dir = Path(embedding_csv_dir_name)
        self.gt_csv_dir = Path(gt_csv_dir_name)
        self.candidate_graph_dir = Path(candidate_graph_dir_name)
        self.run = run
        self.tracker_csv_files: List[Path] = sorted(
            self.tracker_csv_dir.glob(f"{run}/result_tta/*/jsons/ilp_original_ids.csv")
        )
        self.embedding_csv_files: List[Path] = sorted(
            self.embedding_csv_dir.glob(f"{run}/embeddings/*.csv")
        )
        self.gt_csv_files: List[Path] = sorted(self.gt_csv_dir.glob(f"{run}/*.csv"))
        self.candidate_graph_pkl_files: List[Path] = sorted(
            self.candidate_graph_dir.glob(f"{run}/*.pkl")
        )

        if filter_fn:
            self.tracker_csv_files = [f for f in self.tracker_csv_files if filter_fn(f)]
            self.embedding_csv_files = [
                f for f in self.embedding_csv_files if filter_fn(f)
            ]
            self.gt_csv_files = [f for f in self.gt_csv_files if filter_fn(f)]
            self.candidate_graph_pkl_files = [
                f for f in self.candidate_graph_pkl_files if filter_fn(f)
            ]

        assert len(self.tracker_csv_files) == len(self.embedding_csv_files), (
            "Mismatch between tracker and embedding file counts"
        )

        logger.info(f"Found {len(self.tracker_csv_files)} tracker files.")
        logger.info(f"Found {len(self.embedding_csv_files)} embedding files.")
        logger.info(f"Found {len(self.gt_csv_files)} ground truth files.")
        logger.info(
            f"Found {len(self.candidate_graph_pkl_files)} candidate graph files."
        )

        # === Preload files into memory for speed ===
        self.tracker_edges = [
            np.loadtxt(f, delimiter=" ") for f in self.tracker_csv_files
        ]
        self.embeddings = [
            np.loadtxt(f, delimiter=" ") for f in self.embedding_csv_files
        ]
        self.gt_data = [np.loadtxt(f, delimiter=" ") for f in self.gt_csv_files]
        self.graphs = [
            pickle.load(open(f, "rb")) for f in self.candidate_graph_pkl_files
        ]

    def __len__(self):
        return len(self.tracker_csv_files)

    def __getitem__(self, idx):
        logger.info(f"Processing {self.tracker_csv_files[idx]}")
        edges = self.tracker_edges[idx]
        embedding = self.embeddings[idx]
        gt_data = self.gt_data[idx]
        mapping_id_embedding = {
            f"{int(row[1])}_{int(row[0])}": row[2:] for row in embedding
        }
        pairs = []
        for edge in edges:
            id_, t, id_tp1, tp1 = map(int, edge)
            x = np.concatenate(
                [
                    mapping_id_embedding[f"{t}_{id_}"],
                    mapping_id_embedding[f"{tp1}_{id_tp1}"],
                ]
            )
            y_ILP = np.array([1.0], dtype=np.float32)
            y_GT = np.array([0.0])
            for row in gt_data:
                id_gt, t_gt, id_tp1_gt, tp1_gt = map(int, row)
                if id_ == id_gt and t == t_gt and id_tp1 == id_tp1_gt and tp1 == tp1_gt:
                    y_GT = np.array([1.0])
                    break
            pair = np.concatenate([x, np.atleast_1d(y_ILP), np.atleast_1d(y_GT)])
            pairs.append(pair)
        return {
            "pairs": torch.from_numpy(np.stack(pairs)).float(),
            "edges": torch.from_numpy(np.array(edges)).float(),
            "file": str(self.tracker_csv_files[idx]),
        }
