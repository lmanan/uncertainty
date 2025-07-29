from uncertainty.datasets import CSVDatasetInfer
import torch
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from uncertainty.models import Model
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def gaussian_log_prob(z: torch.Tensor) -> torch.Tensor:
    log_p = -0.5 * (z**2 + np.log(2 * np.pi))
    return log_p.flatten(1).sum(-1)


def create_dataset(cfg, ignore_keys):
    def exclude_if_contains(path: Path) -> bool:
        return not any(key in str(path) for key in ignore_keys)

    return CSVDatasetInfer(
        tracker_csv_dir_name=cfg["tracker_csv_dir_name"],
        embedding_csv_dir_name=cfg["embedding_csv_dir_name"],
        candidate_graph_dir_name=cfg["candidate_graph_dir_name"],
        gt_csv_dir_name=cfg["gt_csv_dir_name"],
        run=cfg["run"],
        filter_fn=exclude_if_contains,
    )


def main(args):
    # Experiment setup
    exp_cfg = args["experiment"]
    model_cfg = args["model"]
    dataset_cfg = args["dataset"]

    device = "cuda" if torch.cuda.is_available() else "mps"
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = exp_cfg["num_workers"] if torch.cuda.is_available() else 0
    dataset = create_dataset(dataset_cfg, dataset_cfg["ignore_substrings"])

    loader = DataLoader(
        dataset,
        batch_size=exp_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    model_checkpoint = exp_cfg["model_checkpoint"]
    model = Model(**model_cfg).to(device)
    state = torch.load(model_checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    all_bpd = 0
    n_dims = 129
    count = 0
    all_edges_seq_nll = []

    with torch.no_grad():
        for sequence_idx, batch_mapping in enumerate(loader):
            x = batch_mapping["pairs"]  # B N 130
            edges = batch_mapping["edges"]  # B N 4
            y_GT = x[..., -1].to(device)
            x = x[..., :-1]  # B N 129
            y_ILP = x[..., -1].to(device)
            B, N, D = x.shape
            x = x.view(B * N, D, 1)
            x = x.to(device)
            edges = edges.view(B * N, 4).to(device)
            y = None
            z, output, logdets = model(x, y)
            prior_log_p = gaussian_log_prob(z)
            nll = -prior_log_p / n_dims - logdets
            nll = nll.view(-1, 1)  # reshape to (N, 1)
            edges_with_nll = torch.cat([edges, nll], dim=1)  # result: (N, 5)
            seq_col = torch.full(
                (edges_with_nll.size(0), 1),
                sequence_idx,
                dtype=edges_with_nll.dtype,
                device=edges_with_nll.device,
            )
            edges_seq_nll = torch.cat([seq_col, edges_with_nll], dim=1)  # shape: (N, 6)
            edges_seq_nll = torch.cat(
                [edges_seq_nll, y_ILP.view(-1, 1), y_GT.view(-1, 1)], dim=1
            )
            all_edges_seq_nll.append(
                edges_seq_nll.cpu()
            )  # move to CPU to avoid GPU memory blow-up
            bpd = nll / np.log(2)
            batch_bpd = bpd.sum().item()
            all_bpd += batch_bpd
            count += z.size(0)
            assert z.size(0) == B * N, f"Expected {B * N} samples, got {z.size(0)}"
            running_mean_bpd = all_bpd / count
            logger.info(f"Running mean BPD: {running_mean_bpd:.4f}")
        final_bpd = all_bpd / count
        logger.info(f"BPD: {final_bpd:.4f}")
        final_edges_seq_nll = torch.cat(
            all_edges_seq_nll, dim=0
        )  # shape: (total_edges, 6)
        sorted_indices = torch.argsort(final_edges_seq_nll[:, 5], descending=True)
        sorted_edges_seq_nll = final_edges_seq_nll[sorted_indices]
        columns = ["seq_idx", "id@t", "t", "id@tp1", "tp1", "nll", "y_ILP", "y_GT"]
        df_sorted = pd.DataFrame(sorted_edges_seq_nll.numpy(), columns=columns)
        df_sorted.to_csv("edges_sorted_by_nll.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer TARFlow with YAML config")
    parser.add_argument("--yaml_config_file_name", type=str, required=True)
    args_cli = parser.parse_args()

    with open(args_cli.yaml_config_file_name, "r") as f:
        args_yaml = yaml.safe_load(f)

    main(args_yaml)
