import pandas as pd
import os
import shutil

# Load coordinate files
cbr = pd.read_csv("metadata_code/cbr.csv")  # Contains index, axis-0, axis-1
meta = pd.read_excel("metadata_code/41467_2023_43458_MOESM4_ESM.xlsx", sheet_name=1)  # Contains cell_id and Cluster
image_dir = "images/100"  # Image path
output_base = "images/images_4fold"  # Output path

# Merge cell ID and annotation
meta["index"] = meta.index + 1  # Ensure cell_id matches index
print(meta.head())
df = pd.merge(cbr.drop_duplicates("index"), meta[["index", "Cluster"]], on="index")

# Get image space midpoint and divide into quadrants
x_mid = df["axis-1"].median()
y_mid = df["axis-0"].median()

def get_quadrant(row):
    if row["axis-1"] < x_mid and row["axis-0"] < y_mid:
        return "Q1"  # Top-left
    elif row["axis-1"] >= x_mid and row["axis-0"] < y_mid:
        return "Q2"  # Top-right
    elif row["axis-1"] < x_mid and row["axis-0"] >= y_mid:
        return "Q3"  # Bottom-left
    else:
        return "Q4"  # Bottom-right

df["quadrant"] = df.apply(get_quadrant, axis=1)

# Copy images to corresponding quadrant subfolders
for _, row in df.iterrows():
    cluster = row["Cluster"].replace("&", "and").replace(" ", "_")
    quadrant = row["quadrant"]
    src_path = os.path.join(image_dir, cluster, f"cell_{row['index']}_100.png")
    dst_folder = os.path.join(output_base, quadrant, cluster)
    os.makedirs(dst_folder, exist_ok=True)
    dst_path = os.path.join(dst_folder, f"cell_{row['index']}_100.png")
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)