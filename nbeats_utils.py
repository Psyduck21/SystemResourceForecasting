import torch
import json
import numpy as np
from nbeats_pytorch.model import NBeatsNet

def load_nbeats_model(save_dir="saved_model", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(f"{save_dir}/config.json", "r") as f:
        config = json.load(f)

    # Initialize model
    model = NBeatsNet(
        stack_types=config["stack_types"],
        nb_blocks_per_stack=config["nb_blocks_per_stack"],
        forecast_length=config["forecast_length"],
        backcast_length=config["backcast_length"],
        hidden_layer_units=config["hidden_units"],
        device=device,
    )

    # Load weights
    model.load_state_dict(torch.load(f"{save_dir}/model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load scaling info
    with open(f"{save_dir}/scaling.json", "r") as f:
        scaling_info = json.load(f)

    return model, scaling_info, config


def forecast_future(model, scaling_info, input_series, device=None):
    """
    Forecast the next 5 steps using the N-BEATS model.
    
    Args:
        model: Loaded N-BEATS model.
        scaling_info: Dict containing 'series_min' and 'series_max' for normalization.
        input_series: 1D numpy array of historical series values.
        device: Torch device (CPU or CUDA).

    Returns:
        future_forecast_denorm: 1D numpy array of 5 denormalized forecasted values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series_min, series_max = scaling_info["series_min"], scaling_info["series_max"]
    series_norm = (input_series - series_min) / (series_max - series_min)

    current_seq = torch.tensor(series_norm[-model.backcast_length:], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        if current_seq.dim() == 3:
            current_seq = current_seq.squeeze(-1)
        _, forecast_step = model(current_seq)
        future_forecast = forecast_step.cpu().numpy().reshape(-1)  # length = 5

    # Denormalize
    future_forecast_denorm = future_forecast * (series_max - series_min) + series_min
    return future_forecast_denorm
