from pathlib import Path
import pandas as pd
import os


def create_dataset_csv(dataset_dir: str, output_csv: str = "dataset.csv"):
    components = [
        "front_left_door",
        "front_right_door",
        "rear_left_door",
        "rear_right_door",
        "hood",
    ]

    data = []

    dataset_dir = Path(dataset_dir)

    classnames = os.listdir(dataset_dir)

    for classname in classnames:
        if not os.path.isdir(f"{dataset_dir}/{classname}"):
            continue

        labels = [int(i) for i in classname]
        list_img = os.listdir(f"{dataset_dir}/{classname}")
        for img in list_img:
            obj = {
                "filename": f"{classname}/{img}",
            }
            for i in range(5):
                obj[f"{components[i]}"] = labels[i]
            data.append(obj)

    df = pd.DataFrame(data)

    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")
    print(f"Total images: {len(df)}")

    print("\nFirst 5 rows of the CSV:")
    print(df.head())


if __name__ == "__main__":
    dataset_dir = "./dataset"
    output_csv = "./dataset/metadata.csv"

    create_dataset_csv(dataset_dir, output_csv)
