import json
import os
from pyexpat import model
import sys
import time
import uuid
import logging
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import psutil
import torch
from codecarbon import EmissionsTracker
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.model import SimpleCNN, SimpleDNN
from utils.llm_utils import load_tiny_llm_model, predict_with_tiny_llm
from utils.vlm_utils import load_tiny_vlm_model, predict_with_tiny_vlm

torch.set_grad_enabled(False)
logging.getLogger("codecarbon").setLevel(logging.ERROR)

LOG_DIR = ROOT / "logs"
CHECKPOINT_DIR = ROOT / "checkpoints"
CONFIG_DIR = ROOT / "config"

LOG_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

MODEL_METRICS_PATH = LOG_DIR / "model_metrics.json"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()


def get_hostname():
    return os.getenv("COMPUTERNAME", os.getenv("HOSTNAME", "Unknown_Device"))


def get_device_id():
    id_file = CONFIG_DIR / "device_id.txt"
    if id_file.exists():
        return id_file.read_text().strip()
    new_id = str(uuid.uuid4())
    id_file.write_text(new_id)
    return new_id


DEVICE_UUID = get_device_id()
DEVICE_SHORT = DEVICE_UUID.split("-")[0]


def get_automated_dataset_path():
    raw_dir = LOG_DIR / "raw_devices"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"fingerprint_dataset_{DEVICE_SHORT}_automated.csv"


def load_model_metrics():
    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def append_rows(rows, file_path):
    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if file_path.exists():
        try:
            existing_df = pd.read_csv(file_path, on_bad_lines="skip")
            for col in new_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            for col in existing_df.columns:
                if col not in new_df.columns:
                    new_df[col] = None
            new_df = new_df[existing_df.columns]
            full_df = pd.concat([existing_df, new_df], ignore_index=True)
            full_df.to_csv(file_path, index=False)
            return
        except Exception:
            pass

    new_df.to_csv(file_path, index=False)


def get_existing_count(file_path, model_name):
    if not file_path.exists():
        return 0

    try:
        df = pd.read_csv(file_path, on_bad_lines="skip")
        if "model_type" not in df.columns:
            return 0
        return len(df[df["model_type"].astype(str).str.strip() == model_name])
    except Exception:
        return 0


def tensor_to_canvas_array(image_tensor):
    img = image_tensor.squeeze(0).cpu().numpy() * 255.0
    img = img.clip(0, 255).astype("uint8")
    return img


def run_cnn(model, image_tensor):
    preprocess = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = preprocess(image_tensor).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.inference_mode():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
    exec_time = time.time() - start_time
    return pred, exec_time


def run_dnn(model, image_tensor):
    preprocess = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = preprocess(image_tensor).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.inference_mode():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
    exec_time = time.time() - start_time
    return pred, exec_time


def build_row(model_name, prediction, exec_time, tracker, emissions_value, model_metrics, true_label, sample_index, model_param):
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only"

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "pc_name": get_hostname(),
        "collection_mode": "automated",
        "sample_index": sample_index,
        "true_label": true_label,
        "model_type": model_name,
        "parameters": model_metrics.get("parameters"),
        "prediction": prediction,
        "execution_time_sec": round(exec_time, 4),
        "cpu_energy_kwh": getattr(tracker.final_emissions_data, "cpu_energy", 0),
        "gpu_energy_kwh": getattr(tracker.final_emissions_data, "gpu_energy", 0),
        "ram_energy_kwh": getattr(tracker.final_emissions_data, "ram_energy", 0),
        "total_energy_kwh": getattr(tracker.final_emissions_data, "energy_consumed", 0),
        "total_emissions_kg": emissions_value,
        "cpu_usage_pct": cpu_usage,
        "gpu_model": gpu_name,
        "ram_usage_pct": ram_usage,
        "cpu_clock_mhz": cpu_freq,
        "memory_footprint_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "model_accuracy": model_metrics.get("accuracy"),
        "model_precision_weighted": model_metrics.get("precision_weighted"),
        "model_recall_weighted": model_metrics.get("recall_weighted"),
        "model_f1_weighted": model_metrics.get("f1_weighted"),
        "model_flops": model_metrics.get("flops"),
        "model_param" : model_param
    }


def collect_for_model(model_name, num_samples=1000, flush_every=50):
    print(f"\nCollecting {num_samples} samples for {model_name} on {get_hostname()}")

    all_metrics = load_model_metrics()
    model_metrics = all_metrics.get(model_name, {})
    output_path = get_automated_dataset_path()

    base_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    rows = []

    if model_name == "CNN":
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_cnn.pth", map_location=DEVICE))
        model.eval()

    elif model_name == "DNN":
        model = SimpleDNN().to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_dnn.pth", map_location=DEVICE))
        model.eval()

    elif model_name == "Tiny LLM":
        model, llm_helper_dataset = load_tiny_llm_model(str(CHECKPOINT_DIR / "mnist_tiny_llm.pth"))

    elif model_name == "Tiny VLM":
        model, vlm_text_processor, cached_text_features = load_tiny_vlm_model(str(CHECKPOINT_DIR / "mnist_tiny_vlm.pth"))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    limit = min(num_samples, len(base_dataset))
    existing_count = get_existing_count(output_path, model_name)

    if existing_count >= limit:
        print(f"{model_name}: already complete ({existing_count}/{limit}) -> skipping")
        return

    print(f"{model_name}: resuming from {existing_count}/{limit}")

    progress_bar = tqdm(
        range(existing_count, limit),
        desc=f"{model_name}",
        unit="sample",
        dynamic_ncols=True
    )

    for i in progress_bar:
        image_tensor, true_label = base_dataset[i]

        tracker = EmissionsTracker(save_to_file=False, log_level="error")
        tracker.start()
        model_param = 0

        if model_name == "CNN":
            pred, exec_time = run_cnn(model, image_tensor)

        elif model_name == "DNN":
            pred, exec_time = run_dnn(model, image_tensor)

        elif model_name == "Tiny LLM":
            canvas_array = tensor_to_canvas_array(image_tensor)
            start_time = time.time()
            pred, confidence = predict_with_tiny_llm(canvas_array, model, llm_helper_dataset)
            exec_time = time.time() - start_time

        elif model_name == "Tiny VLM":
            canvas_array = tensor_to_canvas_array(image_tensor)
            start_time = time.time()
            pred, confidence = predict_with_tiny_vlm(canvas_array, model, cached_text_features)
            exec_time = time.time() - start_time

        emissions_value = tracker.stop()
        model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

        row = build_row(
            model_name=model_name,
            prediction=pred,
            exec_time=exec_time,
            tracker=tracker,
            emissions_value=emissions_value,
            model_metrics=model_metrics,
            true_label=int(true_label),
            sample_index=i,
            model_param = model_param
        )

        rows.append(row)

        progress_bar.set_postfix({
            "pred": pred,
            "true": int(true_label),
            "time_s": round(exec_time, 3),
            "done": i + 1
        })

        if (i + 1) % flush_every == 0:
            append_rows(rows, output_path)
            rows = []

    if rows:
        append_rows(rows, output_path)

    print(f"{model_name}: finished {limit}/{limit} -> {output_path}")


def main():
    collect_for_model("CNN", num_samples=1000)
    collect_for_model("DNN", num_samples=1000)
    collect_for_model("Tiny LLM", num_samples=1000)
    collect_for_model("Tiny VLM", num_samples=1000)
    print("\nDone.")


if __name__ == "__main__":
    main()