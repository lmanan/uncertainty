from uncertainty.datasets import CSVDataset
from uncertainty.models import Model
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import logging
import yaml
import argparse
import uncertainty.utils as utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_loss(x: torch.Tensor, y: torch.Tensor | None, model: Model ):
    z, outputs, logdets = model(x, y)
    loss = model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)


def main(args):
    # Exclude test sequences
    test_substrings = [
        "150309-04",
        "150310-11",
        "151029_E1-1",
        "151029_E1-5",
        "151101_E3-12",
        "160112-06",
    ]

    def exclude_if_contains(path: Path) -> bool:
        return not any(bad in str(path) for bad in test_substrings)

    # Training hyperparameters
    experiment_cfg = args["experiment"]
    batch_size = experiment_cfg["batch_size"]
    num_iterations = experiment_cfg["num_iterations"]
    lr = float(experiment_cfg["lr"])
    num_workers = experiment_cfg["num_workers"] if torch.cuda.is_available() else 0
    device = "cuda" if torch.cuda.is_available() else "mps"

    # Dataset
    dataset_cfg = args["dataset"]
    dataset = CSVDataset(
        tracker_csv_dir_name=dataset_cfg["tracker_csv_dir_name"],
        embedding_csv_dir_name=dataset_cfg["embedding_csv_dir_name"],
        gt_csv_dir_name=dataset_cfg["gt_csv_dir_name"],
        run=dataset_cfg["run"],
        filter_fn=exclude_if_contains,
        candidate_graph_dir_name=dataset_cfg["candidate_graph_dir_name"],
    )

    # Model config
    model_cfg = args["model"]
    model = Model(
        in_channels=model_cfg["in_channels"],
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        layers_per_block=model_cfg["layers_per_block"],
        nvp=model_cfg["nvp"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    # Noise config
    noise_cfg = args["noise"]
    std = noise_cfg["std"]
    use_noise = bool(noise_cfg["use_noise"])

    # Data loader
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    optimizer = torch.optim.AdamW(
        model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4
    )

    # TARFlowLogger
    tarflow_logger = utils.TARFlowLogger(
        keys=["iteration", "loss", "loss/mse(z)", "loss/log(|det|)"],
        title="train",
    )

    # Iteration loop
    for i, x in enumerate(loader):
        metrics = utils.Metrics()
        if i >= num_iterations:
            break
        x = x[..., :-1]  # ignore the GT dimension for now!
        x = x.view(x.shape[0] * x.shape[1], 1, -1)
        x = x.to(device)
        if use_noise:
            eps = std + torch.randn_like(x)
            x = x + eps
        y = None  # class conditional input, not used in this example
        optimizer.zero_grad()
        loss, (z, outputs, logdets) = compute_loss(x, y, model)
        tarflow_logger.add("iteration", i)
        tarflow_logger.add("loss", loss.item())
        tarflow_logger.add("loss/mse(z)", 0.5 * (z**2).mean().item())
        tarflow_logger.add("loss/log(|det|)", logdets.mean().item())
        # TODO: nvp
        loss.backward()
        optimizer.step()
        metrics.update(
            {
                "loss": loss,
                "loss/mse(z)": 0.5 * (z**2).mean(),
                "loss/log(|det|)": logdets.mean(),
            }
        )
        logger.info(
            f"At iteration {i}, loss is {loss.item()}, loss/mse(z) is {0.5 * (z**2).mean().item()}, loss/log(|det|) is {logdets.mean().item()}"
        )
        tarflow_logger.write(reset=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with YAML config")
    parser.add_argument(
        "--yaml_config_file_name",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    args_cli = parser.parse_args()

    with open(args_cli.yaml_config_file_name, "r") as f:
        args_yaml = yaml.safe_load(f)

    main(args_yaml)
