import pandas as pd

DATA_DIR = "data_oct_2022/"

MAIN_FILE = "rsm021UA.csv"
OTHER_FILES = [
    "rsm021_141.csv",
    "rsm021_151.csv",
    "rsm021_161.csv",
    "rsm021_251.csv"
]

OUTPUT_FILE = "rsm021_combined.csv"

print("Loading Data...")
main_df = pd.read_csv(DATA_DIR + MAIN_FILE)

for f in OTHER_FILES:
    other_df = pd.read_csv(DATA_DIR + f)
    device_id = other_df["device_id"][0]
    main_df = main_df[main_df["device_id"] != device_id]

    main_df = pd.concat([main_df, other_df])

main_df.to_csv(DATA_DIR + OUTPUT_FILE)
