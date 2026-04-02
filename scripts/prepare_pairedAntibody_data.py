# scripts/prepare_pairedAntibody_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_paired_data(heavy_chains, light_chains, output_path):
    """
    Concatenates paired heavy and light chains with '|' separator.
    Example: EVQLVES...|DIQMTQS...
    """
    assert len(heavy_chains) == len(light_chains), \
        "Heavy and light chains must be paired"

    # Single character concatenation — clean and simple
    sequences = [f"{h}|{l}" for h, l in zip(heavy_chains, light_chains)]

    train_seqs, val_seqs = train_test_split(
        sequences, test_size=0.1, random_state=42)

    df = pd.concat([
        pd.DataFrame({'Split': 'train',      'sequence': train_seqs}),
        pd.DataFrame({'Split': 'validation', 'sequence': val_seqs})
    ])
    df.to_csv(output_path, index=False)

    print(f"Saved {len(train_seqs)} train, {len(val_seqs)} val sequences")
    print(f"Example: {sequences[0][:30]}...|...{sequences[0][-20:]}")





    