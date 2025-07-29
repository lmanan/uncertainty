from uncertainty.datasets import CSVDataset
from uncertainty.models import Model
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import logging
import yaml
import argparse
import uncertainty.utils as utils
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
today_str = datetime.today().strftime("%Y-%m-%d")

torch.backends.cudnn.benchmark = True


def compute_loss(x: torch.Tensor, y: torch.Tensor | None, model: Model):
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
    log_loss_every = experiment_cfg["log_loss_every"]
    val_dataset_length = experiment_cfg["val_dataset_length"]

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

    val_dataset = CSVDataset(
        tracker_csv_dir_name=dataset_cfg["tracker_csv_dir_name"],
        embedding_csv_dir_name=dataset_cfg["embedding_csv_dir_name"],
        gt_csv_dir_name=dataset_cfg["gt_csv_dir_name"],
        run=dataset_cfg["run"],
        filter_fn=exclude_if_contains,
        candidate_graph_dir_name=dataset_cfg["candidate_graph_dir_name"],
        length=val_dataset_length,
    )

    # Model config
    model_cfg = args["model"]
    #model = Model(
    #    in_channels=model_cfg["in_channels"],
    #    img_size=model_cfg["img_size"],
    #    patch_size=model_cfg["patch_size"],
    #    channels=model_cfg["channels"],
    #    num_blocks=model_cfg["num_blocks"],
    #    layers_per_block=model_cfg["layers_per_block"],
    #    nvp=model_cfg["nvp"],
    #    num_classes=model_cfg["num_classes"],
    #).to(device)


    model= Model(
        num_tokens=model_cfg["num_tokens"],
        token_size=model_cfg["token_size"],
        projection_dims=model_cfg["projection_dims"],
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: (1 - it / num_iterations) ** 0.9
    )

    experiment_dir = Path(f"experiment-{today_str}")
    model_dir = experiment_dir / "models_autoencoder"
    model_dir.mkdir(parents=True, exist_ok=True)

    # TARFlowLogger
    tarflow_logger = utils.TARFlowLogger(
        keys=[
            "iteration",
            "train_loss",
            "train_loss/mse(z)",
            "train_loss/log(|det|)",
            "val_loss",
            "val_loss/mse(z)",
            "val_loss/log(|det|)",
        ],
        title=experiment_dir / "training",
    )

    loss_average = 0.0
    min_loss = float("inf")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")

    # Iteration loop
    for iteration, x in enumerate(loader):
        if iteration >= num_iterations:
            break
        x = x[..., :-1]  # ignore the GT dimension for now! B 2 129
        x = x.view(x.shape[0] * x.shape[1], x.shape[-1], 1) # B 129 1
        x = x.to(device)
        if use_noise:
            eps = std + torch.randn_like(x)
            x = x + eps
        y = None  # class conditional input, not used in this example
        optimizer.zero_grad()
        loss, (z, outputs, logdets) = compute_loss(x, y, model)
        loss.backward()
        optimizer.step()
        scheduler.step()

        tarflow_logger.add("iteration", iteration)
        tarflow_logger.add("train_loss", loss.item())
        tarflow_logger.add("train_loss/mse(z)", 0.5 * (z**2).mean().item())
        tarflow_logger.add("train_loss/log(|det|)", logdets.mean().item())
        logger.info(
            f"At iteration {iteration}, loss is {loss.item()}, loss/mse(z) is {0.5 * (z**2).mean().item()}, loss/log(|det|) is {logdets.mean().item()}"
        )
        loss_average += loss.item()

        if (iteration + 1) % log_loss_every == 0:
            loss_average /= log_loss_every
            logger.info(f"[Iter {iteration}],  Avg. Train Loss: {loss_average:.4f}")

            # Evaluate on validation set
            model.eval()
            val_loss_average = 0.0
            count = 0
            with torch.no_grad():
                for val_x in val_loader:
                    val_x = val_x[..., :-1]  # ignore the GT dimension for now!
                    val_x = val_x.view(val_x.shape[0] * val_x.shape[1], 1, -1)
                    val_x = val_x.to(device)
                    if use_noise:
                        eps = std + torch.randn_like(val_x)
                        val_x = val_x + eps
                    val_y = None  # class conditional input, not used in this example
                    val_loss, (val_z, val_outputs, val_logdets) = compute_loss(
                        val_x, val_y, model
                    )
                    val_loss_average += val_loss.item()
                    count += 1
            val_loss_average /= count
            logger.info(f"[Iter {iteration}], Avg. Val Loss: {val_loss_average:.4f}")
            tarflow_logger.add("val_loss", val_loss_average)

            # Save checkpoints
            checkpoint_data = {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "loss_average": loss_average,
                "logger_data": tarflow_logger.data,
            }

            torch.save(checkpoint_data, model_dir / "last.pth")
            if val_loss_average < min_loss:
                min_loss = val_loss_average
                torch.save(checkpoint_data, model_dir / "best.pth")
                logger.info(f"New best model saved at iteration {iteration}")

            loss_average = 0.0
            model.train()
        else:
            tarflow_logger.add("val_loss", "")

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
