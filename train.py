# train_tarflow.py

import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from uncertainty.datasets import CSVDataset
from uncertainty.models import Model
import uncertainty.utils as utils

torch.backends.cudnn.benchmark = True


def setup_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("TARFlow")


def create_dataset(cfg, ignore_keys):
    def exclude_if_contains(path: Path) -> bool:
        return not any(key in str(path) for key in ignore_keys)

    return CSVDataset(
        tracker_csv_dir_name=cfg["tracker_csv_dir_name"],
        embedding_csv_dir_name=cfg["embedding_csv_dir_name"],
        gt_csv_dir_name=cfg["gt_csv_dir_name"],
        run=cfg["run"],
        filter_fn=exclude_if_contains,
        candidate_graph_dir_name=cfg["candidate_graph_dir_name"],
        length=cfg.get("length", None),
    )


def maybe_add_noise(x, std, use_noise):
    return x + std * torch.randn_like(x) if use_noise else x


def compute_loss(x, y, model):
    z, outputs, logdets = model(x, y)
    loss = model.get_loss(z, logdets)
    return loss, (z, outputs, logdets)


def main(args):
    logger = setup_logger()
    today = datetime.today().strftime("%Y-%m-%d")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.get("seed", 42))

    # Experiment setup
    exp_cfg = args["experiment"]
    model_cfg = args["model"]
    dataset_cfg = args["dataset"]
    val_dataset_cfg = args["val_dataset"]
    noise_cfg = args["noise"]

    device = "cuda" if torch.cuda.is_available() else "mps"
    num_workers = exp_cfg["num_workers"] if torch.cuda.is_available() else 0

    dataset = create_dataset(dataset_cfg, dataset_cfg["ignore_substrings"])
    val_dataset = create_dataset(val_dataset_cfg, val_dataset_cfg["ignore_substrings"])

    loader = DataLoader(
        dataset,
        batch_size=exp_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=exp_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
    )

    model = Model(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(exp_cfg["lr"]),
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: (1 - it / exp_cfg["num_iterations"]) ** 0.9
    )

    experiment_dir = Path(f"experiment-{today}")
    model_dir = experiment_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(args, f)

    tarflow_logger = utils.TARFlowLogger(
        keys=[
            "iteration",
            "train_loss",
            "train_loss/mse(z)",
            "train_loss/log(|det|)",
            "val_loss",
            "val_loss/mse(z)",
            "val_loss/log(|det|)",
            "lr",
        ],
        title=experiment_dir / "training",
    )

    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    min_val_loss = float("inf")
    running_train_loss = 0.0
    patience = exp_cfg.get("early_stopping_patience", 10)
    epochs_since_improvement = 0

    for iteration, x in enumerate(loader):
        if iteration >= exp_cfg["num_iterations"]:
            break

        x = x[..., :-1]
        B, N, D = x.shape
        x = x.view(B * N, D, 1)
        x = x.to(device)
        x = maybe_add_noise(x, noise_cfg["std"], noise_cfg["use_noise"])
        y = None

        model.train()
        optimizer.zero_grad()
        loss, (z, _, logdets) = compute_loss(x, y, model)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_mse = 0.5 * (z**2).mean().item()
        loss_logdet = logdets.mean().item()

        tarflow_logger.add("iteration", iteration)
        tarflow_logger.add("train_loss", loss.item())
        tarflow_logger.add("train_loss/mse(z)", loss_mse)
        tarflow_logger.add("train_loss/log(|det|)", loss_logdet)
        tarflow_logger.add("lr", scheduler.get_last_lr()[0])
        logger.info(
            f"At iteration {iteration}, loss is {loss.item()}, loss/mse(z) is {loss_mse}, loss/log(|det|) is {loss_logdet}"
        )
        running_train_loss += loss.item()

        if (iteration + 1) % exp_cfg["log_loss_every"] == 0:
            avg_train_loss = running_train_loss / exp_cfg["log_loss_every"]
            logger.info(f"[Iter {iteration}] Avg. Train Loss: {avg_train_loss:.4f}")
            running_train_loss = 0.0

            # Evaluate
            model.eval()
            val_loss_sum, val_batches = 0.0, 0

            with torch.no_grad():
                for val_x in val_loader:
                    val_x = val_x[..., :-1]
                    B, N, D = val_x.shape
                    val_x = val_x.view(B * N, D, 1)
                    val_x = val_x.to(device)
                    val_x = maybe_add_noise(
                        val_x, noise_cfg["std"], noise_cfg["use_noise"]
                    )
                    val_y = None
                    val_loss, (val_z, _, val_logdets) = compute_loss(
                        val_x, val_y, model
                    )

                    tarflow_logger.add(
                        "val_loss/mse(z)", 0.5 * (val_z**2).mean().item()
                    )
                    tarflow_logger.add("val_loss/log(|det|)", val_logdets.mean().item())

                    val_loss_sum += val_loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / val_batches
            tarflow_logger.add("val_loss", avg_val_loss)
            logger.info(f"[Iter {iteration}] Avg. Val Loss: {avg_val_loss:.4f}")

            # Save checkpoints
            checkpoint = {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "loss_average": avg_train_loss,
                "logger_data": tarflow_logger.data,
            }

            # Save checkpoint
            torch.save(checkpoint, model_dir / "last.pth")

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                epochs_since_improvement = 0
                torch.save(checkpoint, model_dir / "best.pth")
                logger.info(f"New best model saved at iteration {iteration}")
            else:
                epochs_since_improvement += 1
                logger.info(
                    f"No improvement. Patience: {epochs_since_improvement}/{patience}"
                )

            if epochs_since_improvement >= patience:
                logger.info("Early stopping triggered — stopping training.")
                break

            torch.save(checkpoint, model_dir / "last.pth")
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(checkpoint, model_dir / "best.pth")
                logger.info(f"New best model saved at iteration {iteration}")
        else:
            tarflow_logger.add("val_loss", "")

        tarflow_logger.write(reset=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TARFlow with YAML config")
    parser.add_argument("--yaml_config_file_name", type=str, required=True)
    args_cli = parser.parse_args()

    with open(args_cli.yaml_config_file_name, "r") as f:
        args_yaml = yaml.safe_load(f)

    main(args_yaml)
