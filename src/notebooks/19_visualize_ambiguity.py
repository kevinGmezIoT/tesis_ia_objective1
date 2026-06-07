# %% [markdown]
# # Visualización de Resultados de Ambigüedad Visual-Semántica
# 
# Este notebook genera las visualizaciones cualitativas para el artículo, explorando
# cómo se distribuyen los cuadrantes de ambigüedad calculados al cruzar la similitud
# semántica (Gower) y la similitud visual (OSNet).

# %%
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.manifold import TSNE

# Configuración de estética base
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

TEXTS = {
    'en': {
        'quadrants': {
            'baja_ambiguedad': 'Low Ambiguity',
            'semejanza_semantica_controlable': 'Controllable Semantic Similarity',
            'ambiguedad_visual_no_explicada': 'Unexplained Visual Ambiguity',
            'ambiguedad_confirmada': 'Confirmed Ambiguity'
        },
        'bar_title': 'Proportion of Ambiguity Quadrants',
        'bar_xlabel': 'Number of Pairs',
        'scatter_title': 'Attribute Similarity (Gower) vs Visual Similarity (OSNet)',
        'scatter_xlabel': 'Semantic Similarity (Gower) + Jitter',
        'scatter_ylabel': 'OSNet Cosine Visual Similarity',
        'scatter_legend': 'Quadrant',
        'box_titles': ['Max Low Illumination', 'Max Heavy Occlusion', 'Max Degraded Silhouette'],
        'box_ylabel': 'Proportion (worst case in pair)',
        'tsne_title': 'Interactive Sequence t-SNE (OSNet)',
        'tsne_identity': 'Identity'
    },
    'es': {
        'quadrants': {
            'baja_ambiguedad': 'Baja Ambigüedad',
            'semejanza_semantica_controlable': 'Semejanza Semántica Controlable',
            'ambiguedad_visual_no_explicada': 'Ambigüedad Visual No Explicada',
            'ambiguedad_confirmada': 'Ambigüedad Confirmada'
        },
        'bar_title': 'Proporción de Cuadrantes de Ambigüedad',
        'bar_xlabel': 'Número de Pares',
        'scatter_title': 'Similitud de Atributos (Gower) vs Similitud Visual (OSNet)',
        'scatter_xlabel': 'Similitud Semántica (Gower) + Jitter',
        'scatter_ylabel': 'Similitud Visual Coseno (OSNet)',
        'scatter_legend': 'Cuadrante',
        'box_titles': ['Máxima Iluminación Baja', 'Máxima Oclusión Severa', 'Máxima Silueta Degradada'],
        'box_ylabel': 'Proporción (peor caso en el par)',
        'tsne_title': 't-SNE Interactivo de Secuencias (OSNet)',
        'tsne_identity': 'Identidad'
    }
}

out_dirs = {
    'en': "D:/tesis_dev/result1/src/paper/english/figures",
    'es': "D:/tesis_dev/result1/src/paper/spanish/figures"
}

for d in out_dirs.values():
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## 1. Carga de Datos

# %%
pairs_path = 'D:/tesis_dev/result1/dataset/output/metadata/gower_osnet_cross_analysis.csv'
summary_path = 'D:/tesis_dev/result1/dataset/output/metadata/gower_osnet_ambiguity_summary.csv'

pairs_df = pd.read_csv(pairs_path)
summary_df = pd.read_csv(summary_path)

p90_gower = summary_df['p90_gower'].iloc[0]
p90_osnet = summary_df['p90_osnet'].iloc[0]

valid_df = pairs_df.dropna(subset=['gower_similarity', 'osnet_similarity']).copy()

quadrant_order = [
    'baja_ambiguedad', 
    'semejanza_semantica_controlable', 
    'ambiguedad_visual_no_explicada', 
    'ambiguedad_confirmada'
]
quadrant_palette = {
    'baja_ambiguedad': '#cccccc',
    'semejanza_semantica_controlable': '#4c72b0',
    'ambiguedad_visual_no_explicada': '#dd8452',
    'ambiguedad_confirmada': '#c44e52'
}

np.random.seed(42)
valid_df['gower_jitter'] = valid_df['gower_similarity'] + np.random.uniform(-0.02, 0.02, size=len(valid_df))

osnet_seq_df = pd.read_csv('D:/tesis_dev/result1/dataset/output/embeddings/sequence_embeddings_osnet.csv')
osnet_seq_df['vector'] = osnet_seq_df['embedding_json'].apply(json.loads).apply(np.array)
X = np.vstack(osnet_seq_df['vector'].values)
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X)

osnet_seq_df['tsne_x'] = X_tsne[:, 0]
osnet_seq_df['tsne_y'] = X_tsne[:, 1]
osnet_seq_df['person_global_id_str'] = osnet_seq_df['person_global_id'].astype(str)


for lang in ['en', 'es']:
    # 2. Bar Plot
    counts = valid_df['ambiguity_quadrant'].value_counts().reindex(quadrant_order)
    
    plt.figure(figsize=(10, 5))
    translated_labels = [TEXTS[lang]['quadrants'][q] for q in counts.index]
    ax = sns.barplot(
        y=translated_labels, 
        x=counts.values, 
        palette=list(quadrant_palette.values())
    )
    plt.title(TEXTS[lang]['bar_title'])
    plt.xlabel(TEXTS[lang]['bar_xlabel'])
    plt.ylabel("")
    
    total = counts.sum()
    for p in ax.patches:
        width = p.get_width()
        pct = 100 * width / total
        plt.text(width + total*0.01, p.get_y() + p.get_height()/2., 
                 f"{int(width)} ({pct:.1f}%)", va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dirs[lang], 'ambiguity_quadrants_bar.png'), dpi=300)
    plt.close()

    # 3. Scatter Plot
    plt.figure(figsize=(10, 8))
    
    df_plot = valid_df.copy()
    df_plot['ambiguity_quadrant_translated'] = df_plot['ambiguity_quadrant'].map(TEXTS[lang]['quadrants'])
    translated_order = [TEXTS[lang]['quadrants'][q] for q in quadrant_order]
    
    sns.scatterplot(
        data=df_plot,
        x='gower_jitter',
        y='osnet_similarity',
        hue='ambiguity_quadrant_translated',
        palette=dict(zip(translated_order, quadrant_palette.values())),
        hue_order=translated_order,
        alpha=0.7,
        s=40,
        edgecolor=None
    )
    
    plt.axvline(x=max(p90_gower, 0.80), color='black', linestyle='--', label=f'Gower Umbral ({max(p90_gower, 0.80):.2f})')
    plt.axhline(y=p90_osnet, color='gray', linestyle='--', label=f'OSNet P90 ({p90_osnet:.4f})')
    
    plt.title(TEXTS[lang]['scatter_title'])
    plt.xlabel(TEXTS[lang]['scatter_xlabel'])
    plt.ylabel(TEXTS[lang]['scatter_ylabel'])
    plt.legend(title=TEXTS[lang]['scatter_legend'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dirs[lang], 'gower_vs_osnet_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ('pair_max_low_illum_ratio', TEXTS[lang]['box_titles'][0]),
        ('pair_max_heavy_occlusion_ratio', TEXTS[lang]['box_titles'][1]),
        ('pair_max_degraded_sil_ratio', TEXTS[lang]['box_titles'][2])
    ]
    
    for i, (col, title) in enumerate(metrics):
        sns.boxplot(
            data=df_plot, 
            x='ambiguity_quadrant_translated', 
            y=col, 
            ax=axes[i],
            palette=dict(zip(translated_order, quadrant_palette.values())),
            order=translated_order
        )
        axes[i].set_title(title)
        axes[i].set_ylabel(TEXTS[lang]['box_ylabel'])
        axes[i].set_xlabel("")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dirs[lang], 'context_boxplots.png'), dpi=300)
    plt.close()

    # 5. TSNE HTML
    fig = px.scatter(
        osnet_seq_df, 
        x='tsne_x', 
        y='tsne_y', 
        color='person_global_id_str', 
        hover_data=['sequence_id', 'cam_id', 'person_global_id'],
        title=TEXTS[lang]['tsne_title'],
        labels={"person_global_id_str": TEXTS[lang]['tsne_identity']}
    )
    
    html_out = os.path.join(out_dirs[lang], 'tsne_sequences_interactive.html')
    fig.write_html(html_out)
    print(f"[{lang}] Done.")
