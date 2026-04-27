# Helper Functions for Debugging

from IPython.display import display

def print_dataset_info(dfs):
    for name, df in dfs.items():
        print(f"Dataset: {name}")
        print(f"Shape: {df.shape}")
        display(df.head())
        print("\n")
