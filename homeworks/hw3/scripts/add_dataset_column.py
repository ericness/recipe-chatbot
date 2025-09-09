import numpy as np
import pandas as pd


def add_dataset_column():
    """
    Add a 'dataset' column to labeled_traces.csv with random splits:
    - 20% train
    - 40% dev
    - 40% test
    """

    # Read the CSV file
    input_file = "homeworks/hw3/data/labeled_traces.csv"
    output_file = "homeworks/hw3/data/labeled_traces_split.csv"

    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Original dataset has {len(df)} rows")

    # Shuffle the dataframe to ensure random assignment
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split sizes
    total_rows = len(df_shuffled)
    train_size = int(total_rows * 0.20)
    dev_size = int(total_rows * 0.40)
    test_size = total_rows - train_size - dev_size  # Remaining rows go to test

    print(f"Split sizes:")
    print(f"  Train: {train_size} rows ({train_size / total_rows * 100:.1f}%)")
    print(f"  Dev: {dev_size} rows ({dev_size / total_rows * 100:.1f}%)")
    print(f"  Test: {test_size} rows ({test_size / total_rows * 100:.1f}%)")

    # Create the dataset column
    dataset_labels = ["train"] * train_size + ["dev"] * dev_size + ["test"] * test_size
    df_shuffled["dataset"] = dataset_labels

    # Verify the splits
    print(f"\nActual distribution:")
    print(df_shuffled["dataset"].value_counts().sort_index())
    print(f"\nPercentages:")
    print(df_shuffled["dataset"].value_counts(normalize=True).sort_index() * 100)

    # Save to new file
    df_shuffled.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")

    return df_shuffled


if __name__ == "__main__":
    result_df = add_dataset_column()
