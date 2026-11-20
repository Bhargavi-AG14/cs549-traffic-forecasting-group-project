from pathlib import Path
import numpy as np

from starter_preprocessing import prepare_data

# Load and preprocess the data
data = prepare_data("upload")

# Create output folder
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)


def save_split(name):
    split = data[name]
    features = split["features"]

    output = {
        "X": split["X"],
        "valid_mask": split["valid_mask"],
        "timestamps": split["timestamps"],
        "sample_id": split["sample_id"],
        "sensor_ids": split["sensor_ids"],
        "last_speed": features["last_speed"],
        "mean_last_3": features["mean_last_3"],
        "mean_last_6": features["mean_last_6"],
        "std_last_6": features["std_last_6"],
        "trend_last_15min": features["trend_last_15min"],
        "missing_history": features["missing_history"],
        "time_features": features["time"],
        "sensor_coordinates": features["sensor_coordinates"],
    }

    if "Y_mph" in split:
        output["Y_mph"] = split["Y_mph"]
        output["Y_scaled"] = split["Y_scaled"]

    np.savez_compressed(output_dir / f"{name}_processed.npz", **output)
    print(f"Saved {name}_processed.npz")


save_split("train")
save_split("validation")
save_split("test")