import pandas as pd

# 1. Unir a nivel de secuencia (sequences_occlusion.csv)
df_seq_mars = pd.concat([
    pd.read_csv(f"D:/open_datasets/Mars/metadata/sequences_occlusion_shard_{i}_3.csv") 
    for i in range(3)
], ignore_index=True)
df_seq_mars.to_csv("D:/open_datasets/Mars/metadata/sequences_occlusion.csv", index=False)

# 2. Unir a nivel de frame (sequence_frames_occlusion.csv)
df_frames_mars = pd.concat([
    pd.read_csv(f"D:/open_datasets/Mars/metadata/sequence_frames_occlusion_shard_{i}_3.csv") 
    for i in range(3)
], ignore_index=True)
df_frames_mars.to_csv("D:/open_datasets/Mars/metadata/sequence_frames_occlusion.csv", index=False)

print("¡Archivos de MARS consolidados con éxito!")
