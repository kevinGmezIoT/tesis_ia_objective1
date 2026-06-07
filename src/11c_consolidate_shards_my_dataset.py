import pandas as pd

# 1. Unir el archivo de oclusión a nivel de secuencia (gait_cycles)
df_seq = pd.concat([
    pd.read_csv(f"dataset/output/metadata/gait_cycles_occlusion_shard_{i}_3.csv") 
    for i in range(3)
], ignore_index=True)
df_seq.to_csv("dataset/output/metadata/gait_cycles_occlusion.csv", index=False)

# 2. Unir el archivo de oclusión a nivel de frame (gait_cycle_frames)
df_frames = pd.concat([
    pd.read_csv(f"dataset/output/metadata/gait_cycle_frames_occlusion_shard_{i}_3.csv") 
    for i in range(3)
], ignore_index=True)
df_frames.to_csv("dataset/output/metadata/gait_cycle_frames_occlusion.csv", index=False)

print("¡Archivos consolidados con éxito!")
