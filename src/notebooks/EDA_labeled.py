# %% [markdown]
# # Exploratory Data Analysis (EDA) - Labeled Datasets
# Este notebook verifica y visualiza los atributos detectados para cada dataset.
# - Para Datasets RGB (Propio, MARS, iLIDS-VID): Calidad de silueta, iluminación, oclusión y atributos semánticos (ropa/color).
# - Para Datasets de Siluetas (CASIA-B, Gait3D): Solo calidad de silueta (SQQ/MCA).

# %%
import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)

# %% [markdown]
# ## 1. Configuración de Datasets
# Define las rutas a los metadatos generados para cada dataset. Modifica las rutas según donde tengas tus archivos.

# %%
DATASETS = {
    "Tu_Corpus": {
        "type": "RGB",
        "base_dir": "D:/tesis_dev/result1/dataset/output",
        "sequences_csv": "D:/tesis_dev/result1/dataset/output/metadata/gait_cycles.csv",
        "frames_csv": "D:/tesis_dev/result1/dataset/output/metadata/gait_cycle_frames.csv",
        "quality_csv": "D:/tesis_dev/result1/dataset/output/metadata/gait_cycles_quality.csv",
        "occlusion_csv": "D:/tesis_dev/result1/dataset/output/metadata/gait_cycles_occlusion.csv",
        "illumination_csv": "D:/tesis_dev/result1/dataset/output/metadata/sequences_illumination.csv",
        "visual_attr_csv": "D:/tesis_dev/result1/dataset/output/metadata/sequence_visual_attributes.csv"
    },
    "CASIA-B": {
        "type": "SILHOUETTE",
        "base_dir": "D:/open_datasets/CASIA-B",
        "sequences_csv": "D:/open_datasets/CASIA-B/metadata/gait_cycles.csv",
        "frames_csv": "D:/open_datasets/CASIA-B/metadata/gait_cycle_frames.csv",
        "quality_csv": "D:/open_datasets/CASIA-B/metadata/gait_cycles_quality.csv",
        "occlusion_csv": None,
        "illumination_csv": None,
        "visual_attr_csv": None
    },
    "Gait3D": {
        "type": "SILHOUETTE",
        "base_dir": "D:/open_datasets/Gait3D",
        "sequences_csv": "D:/open_datasets/Gait3D/metadata/gait_cycles.csv",
        "frames_csv": "D:/open_datasets/Gait3D/metadata/gait_cycle_frames.csv",
        "quality_csv": "D:/open_datasets/Gait3D/metadata/gait_cycles_quality.csv",
        "occlusion_csv": None,
        "illumination_csv": None,
        "visual_attr_csv": None
    },
    "MARS": {
        "type": "RGB",
        "base_dir": "D:/open_datasets/MARS/raw",
        "sequences_csv": "D:/open_datasets/MARS/metadata/sequences.csv",
        "frames_csv": "D:/open_datasets/MARS/metadata/sequence_frames.csv",
        "quality_csv": "D:/open_datasets/MARS/metadata/sequences_quality.csv",
        "occlusion_csv": "D:/open_datasets/MARS/metadata/sequences_occlusion.csv",
        "illumination_csv": "D:/open_datasets/MARS/metadata/sequences_illumination.csv",
        "visual_attr_csv": "D:/open_datasets/MARS/metadata/sequence_visual_attributes.csv"
    },
    "iLIDS-VID": {
        "type": "RGB",
        "base_dir": "D:/open_datasets/i-LIDS-VID",
        "sequences_csv": "D:/open_datasets/i-LIDS-VID/metadata/sequences.csv",
        "frames_csv": "D:/open_datasets/i-LIDS-VID/metadata/sequence_frames.csv",
        "quality_csv": "D:/open_datasets/i-LIDS-VID/metadata/sequences_quality.csv",
        "occlusion_csv": "D:/open_datasets/i-LIDS-VID/metadata/sequences_occlusion.csv",
        "illumination_csv": "D:/open_datasets/i-LIDS-VID/metadata/sequences_illumination.csv",
        "visual_attr_csv": "D:/open_datasets/i-LIDS-VID/metadata/sequence_visual_attributes.csv"
    }
}

# ==========================================
# SELECCIONA EL DATASET A ANALIZAR AQUÍ:
# ==========================================
CURRENT_DATASET_NAME = "Tu_Corpus" 
# ==========================================

config = DATASETS[CURRENT_DATASET_NAME]
print(f"Analizando dataset: {CURRENT_DATASET_NAME} ({config['type']})")

# %% [markdown]
# ## 2. Carga y Consolidación de Metadatos
# Unificaremos todos los CSVs disponibles en un único DataFrame en base a `sequence_id`.

# %%
def load_and_merge_metadata(cfg):
    if not os.path.exists(cfg["sequences_csv"]):
        print(f"ERROR: No se encontró {cfg['sequences_csv']}")
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.read_csv(cfg["sequences_csv"])
    
    # Merge Quality
    if cfg["quality_csv"] and os.path.exists(cfg["quality_csv"]):
        df_q = pd.read_csv(cfg["quality_csv"])
        df = df.merge(df_q, on="sequence_id", how="left", suffixes=("", "_dup"))
        
    # Merge Occlusion (solo RGB)
    if cfg["occlusion_csv"] and os.path.exists(cfg["occlusion_csv"]):
        df_o = pd.read_csv(cfg["occlusion_csv"])
        df = df.merge(df_o, on="sequence_id", how="left", suffixes=("", "_dup"))
        
    # Merge Illumination (solo RGB)
    if cfg["illumination_csv"] and os.path.exists(cfg["illumination_csv"]):
        df_i = pd.read_csv(cfg["illumination_csv"])
        df = df.merge(df_i, on="sequence_id", how="left", suffixes=("", "_dup"))
        
    # Merge Visual Attributes (Clothing/Color) (solo RGB)
    if cfg["visual_attr_csv"] and os.path.exists(cfg["visual_attr_csv"]):
        df_v = pd.read_csv(cfg["visual_attr_csv"])
        df = df.merge(df_v, on="sequence_id", how="left", suffixes=("", "_dup"))
        
    # Limpiar duplicados si hubieron cruces
    df = df.loc[:, ~df.columns.str.endswith('_dup')]
    
    # Cargar frames df
    df_frames = pd.DataFrame()
    if cfg["frames_csv"] and os.path.exists(cfg["frames_csv"]):
        df_frames = pd.read_csv(cfg["frames_csv"])
        
    return df, df_frames

df, frames_df = load_and_merge_metadata(config)
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])
df.head()

# %% [markdown]
# ## 3. Análisis Descriptivo de Atributos

# %%
def hist_plot(series: pd.Series, title: str, bins: int = 40):
    series = series.dropna()
    if series.empty:
        print(f"Sin datos para {title}")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(series.values, bins=bins, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

def bar_plot_counts(series: pd.Series, title: str):
    vc = series.value_counts(dropna=False)
    if vc.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(vc.index.astype(str), vc.values, color="coral", edgecolor="black")
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Conteo")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

# %%
# Calidad de Silueta (Aplica a RGB y SILHOUETTES)
if "sil_quality_mean" in df.columns:
    hist_plot(df["sil_quality_mean"], "Distribución de Calidad SQQ (sil_quality_mean)", bins=50)
if "mca_ratio_mean" in df.columns:
    hist_plot(df["mca_ratio_mean"], "Distribución de Maximal Connected Area (mca_ratio_mean)", bins=50)

# Iluminación (Solo RGB)
if "illum_y_mean" in df.columns:
    hist_plot(df["illum_y_mean"], "Distribución de Brillo (illum_y_mean)", bins=50)
if "illum_level" in df.columns:
    bar_plot_counts(df["illum_level"], "Conteo por Nivel de Iluminación")

# Oclusión (Solo RGB)
if "occlusion_pct" in df.columns:
    hist_plot(df["occlusion_pct"], "Distribución de Oclusión (%)", bins=50)
if "occlusion_level" in df.columns:
    bar_plot_counts(df["occlusion_level"], "Conteo por Nivel de Oclusión")

# Ropa y Mochila (Solo RGB)
for col in ["upper_garment_type", "lower_garment_type", "backpack_flag"]:
    if col in df.columns:
        bar_plot_counts(df[col], f"Conteo de Atributos - {col}")

# Colores de Ropa (Solo RGB)
for col in ["upper_color_label", "lower_color_label"]:
    if col in df.columns:
        bar_plot_counts(df[col], f"Conteo de Colores - {col}")

# %% [markdown]
# ## 4. Inspección Visual Aleatoria
# Vamos a extraer algunas muestras aleatorias de ciclos para ver sus imágenes y entender si las etiquetas predichas coinciden con la realidad visual.

# %%
def _safe_join(base_dir: str, path_str: str) -> str:
    if pd.isna(path_str) or not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(Path(base_dir) / p)

def visualize_cycle(sequence_row, frames_subset, base_dir, dataset_type):
    """Muestra frames equiespaciados de una secuencia junto con su metadata."""
    if frames_subset.empty:
        return
        
    # Muestrear 5 frames equiespaciados
    n_frames = len(frames_subset)
    indices = np.linspace(0, n_frames - 1, min(5, n_frames), dtype=int)
    sampled = frames_subset.iloc[indices]
    
    # Determinar si mostrar RGB, Silueta o Ambos
    show_rgb = (dataset_type == "RGB")
    show_sil = True # Todos tienen siluetas
    
    nrows = (1 if show_rgb else 0) + (1 if show_sil else 0)
    ncols = len(sampled)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0) # Forzar 2D array
        
    seq_id = sequence_row["sequence_id"]
    
    # Preparar título con atributos detectados
    title_parts = [f"Seq: {seq_id}"]
    if "sil_quality_mean" in sequence_row:
        title_parts.append(f"SQQ: {sequence_row['sil_quality_mean']:.2f}")
    if "illum_level" in sequence_row:
        title_parts.append(f"Illum: {sequence_row.get('illum_level', 'NA')}")
    if "occlusion_level" in sequence_row:
        title_parts.append(f"Occ: {sequence_row.get('occlusion_level', 'NA')}")
    if "upper_garment_type" in sequence_row:
        title_parts.append(f"Up: {sequence_row.get('upper_garment_type', 'NA')} ({sequence_row.get('upper_color_label', 'NA')})")
    if "lower_garment_type" in sequence_row:
        title_parts.append(f"Low: {sequence_row.get('lower_garment_type', 'NA')} ({sequence_row.get('lower_color_label', 'NA')})")
    if "backpack_flag" in sequence_row:
        title_parts.append(f"Bag: {sequence_row.get('backpack_flag', 'NA')}")
        
    plt.suptitle(" | ".join(title_parts), fontsize=11, y=1.05)
    
    for c, (_, frame_row) in enumerate(sampled.iterrows()):
        r = 0
        if show_rgb:
            rgb_path = _safe_join(base_dir, frame_row.get("rgb_source_path", ""))
            if os.path.exists(rgb_path):
                img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
                axes[r, c].imshow(img)
            else:
                axes[r, c].text(0.5, 0.5, "NO RGB", ha="center")
            axes[r, c].axis("off")
            axes[r, c].set_title(f"Frame {frame_row['frame_number']}")
            r += 1
            
        if show_sil:
            sil_path = _safe_join(base_dir, frame_row.get("silhouette_source_path", ""))
            if os.path.exists(sil_path):
                sil = cv2.imread(sil_path, cv2.IMREAD_GRAYSCALE)
                axes[r, c].imshow(sil, cmap="gray")
            else:
                axes[r, c].text(0.5, 0.5, "NO SIL", ha="center")
            axes[r, c].axis("off")
            if not show_rgb: # Si solo hay siluetas, el título va aquí
                axes[r, c].set_title(f"Frame {frame_row['frame_number']}")
                
    plt.tight_layout()
    plt.show()

# %%
if not df.empty and not frames_df.empty:
    N_SAMPLES = 5
    # Tomamos muestras aleatorias que tengan datos válidos
    sample_df = df.dropna(subset=["sequence_id"]).sample(min(N_SAMPLES, len(df)))
    
    for _, row in sample_df.iterrows():
        seq_frames = frames_df[frames_df["sequence_id"] == row["sequence_id"]]
        visualize_cycle(row, seq_frames, config["base_dir"], config["type"])
else:
    print("No se pudo cargar metadata suficiente para graficar (df o frames_df vacíos).")

# %%
