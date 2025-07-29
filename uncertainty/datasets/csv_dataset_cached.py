from torch.utils.data import IterableDataset
from pathlib import Path
from typing import Optional, List, Callable
import torch
import logging
import numpy as np
import pickle

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CSVDataset(IterableDataset):
    def __init__(
        self,
        tracker_csv_dir_name: str,
        embedding_csv_dir_name: str,
        candidate_graph_dir_name: str,
        gt_csv_dir_name: str,
        run: str | None = None,
        filter_fn: Optional[Callable[[Path], bool]] = None,
        length: int | None = None,
    ):
        self.tracker_dir = Path(tracker_csv_dir_name)
        self.embedding_dir = Path(embedding_csv_dir_name)
        self.gt_dir = Path(gt_csv_dir_name)
        self.candidate_graph_dir = Path(candidate_graph_dir_name)
        if run is not None:
            self.tracker_csv_files: List[Path] = sorted(
                self.tracker_dir.glob(f"{run}/result_tta/*/jsons/ilp_original_ids.csv")
            )
            self.embedding_csv_files: List[Path] = sorted(
                self.embedding_dir.glob(f"{run}/embeddings/*.csv")
            )
            self.gt_csv_files: List[Path] = sorted(self.gt_dir.glob(f"{run}/*.csv"))
            self.candidate_graph_pkl_files: List[Path] = sorted(
                self.candidate_graph_dir.glob(f"{run}/*.pkl")
            )
        else:
            self.tracker_csv_files: List[Path] = sorted(
                self.tracker_dir.glob(
                    "[0][0-9]/result_tta/*/jsons/ilp_original_ids.csv"
                )
            )
            self.embedding_csv_files: List[Path] = sorted(
                self.embedding_dir.glob("[0][0-9]/embeddings/*.csv")
            )
            self.gt_csv_files: List[Path] = sorted(self.gt_dir.glob("[0][0-9]/*.csv"))
            self.candidate_graph_pkl_files: List[Path] = sorted(
                self.candidate_graph_dir.glob("[0][0-9]/*.pkl")
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

        self.length = length

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

    def __iter__(self):
        if self.length is None:
            while True:
                yield self.create_sample()
        else:
            for _ in range(self.length):
                yield self.create_sample()

    def create_sample(self):
        index = np.random.randint(len(self.tracker_edges))
        edges = self.tracker_edges[index]
        index_edge = np.random.randint(len(edges))
        id_, t, id_tp1, tp1 = map(int, edges[index_edge])

        embedding = self.embeddings[index]
        mapping_id_embedding = {
            f"{int(row[1])}_{int(row[0])}": row[2:] for row in embedding
        }

        try:
            x_positive = np.concatenate(
                [
                    mapping_id_embedding[f"{t}_{id_}"],
                    mapping_id_embedding[f"{tp1}_{id_tp1}"],
                ]
            )
        except KeyError:
            logger.warning("Missing embedding — resampling")
            return self.create_sample()

        y_ILP_positive = np.array([1.0], dtype=np.float32)

        gt_data = self.gt_data[index]
        y_GT_positive = 0.0
        for row in gt_data:
            id_gt, t_gt, id_tp1_gt, tp1_gt = map(int, row)
            if id_ == id_gt and t == t_gt and id_tp1 == id_tp1_gt and tp1 == tp1_gt:
                y_GT_positive = 1.0
                break

        G = self.graphs[index]
        node = f"{t}_{id_}"
        if node in G:
            out_edges = list(G.out_edges(node))
            if len(out_edges) > 1:
                not_found_negative = True
                while not_found_negative:
                    out_edge = out_edges[np.random.randint(len(out_edges))]
                    if out_edge[1] != f"{tp1}_{id_tp1}":
                        not_found_negative = False

                t_negative, id_negative = map(int, out_edge[1].split("_"))
                try:
                    x_negative = np.concatenate(
                        [
                            mapping_id_embedding[f"{t}_{id_}"],
                            mapping_id_embedding[f"{t_negative}_{id_negative}"],
                        ]
                    )
                except KeyError:
                    return self.create_sample()

                y_ILP_negative = np.array([0.0], dtype=np.float32)
                y_GT_negative = 0.0
                for row in gt_data:
                    id_gt, t_gt, id_tp1_gt, tp1_gt = map(int, row)
                    if (
                        id_ == id_gt
                        and t == t_gt
                        and id_negative == id_tp1_gt
                        and t_negative == tp1_gt
                    ):
                        y_GT_negative = 1.0
                    break
            else:
                x_negative = x_positive
                y_ILP_negative = y_ILP_positive
                y_GT_negative = y_GT_positive
        else:
            return self.create_sample()

        pair_positive = np.concatenate(
            [x_positive, np.atleast_1d(y_ILP_positive), np.atleast_1d(y_GT_positive)]
        )
        pair_negative = np.concatenate(
            [x_negative, np.atleast_1d(y_ILP_negative), np.atleast_1d(y_GT_negative)]
        )

        stacked = np.stack([pair_positive, pair_negative], axis=0)
        return torch.from_numpy(stacked).float()
