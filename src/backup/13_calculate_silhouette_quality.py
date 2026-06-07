import pandas as pd
import numpy as np
import os
import argparse
import cv2
from glob import glob
from tqdm import tqdm

def analyze_silhouette_image(image_path):
    """
    Calcula fill_norm y border_frac para una imagen de silueta binaria.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    # Asegurar que sea binaria (0 o 255)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 1. Calcular Área de la silueta
    area = np.sum(binary == 255)
    if area == 0:
        return 0.0, 1.0 # 0 calidad

    # 2. Calcular Convex Hull para fill_norm
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 1.0
    
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    
    # fill_norm: Relación entre área real y área del casco convexo
    # Una silueta humana sana suele tener una relación específica
    fill_norm = area / hull_area if hull_area > 0 else 0
    
    # 3. Calcular border_frac (Ruido en los bordes)
    # Medimos el perímetro total vs el perímetro suavizado para detectar "picos"
    perimeter = cv2.arcLength(cnt, True)
    # Suavizar contorno (epsilon proporcional al perímetro)
    epsilon = 0.005 * perimeter 
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx_perimeter = cv2.arcLength(approx, True)
    
    # border_frac: Diferencia entre el perímetro real y el suavizado
    # Un valor alto indica bordes muy irregulares/ruidosos
    border_frac = (perimeter - approx_perimeter) / perimeter if perimeter > 0 else 1.0
    border_frac = np.clip(border_frac, 0, 1)

    return fill_norm, border_frac

def calculate_sqi_from_images(input_csv, silhouettes_dir, output_report, output_table, limit=None, sample_frames=5):
    """
    Analiza las siluetas en disco y calcula el SQI.
    """
    if not os.path.exists(os.path.dirname(output_report)):
        os.makedirs(os.path.dirname(output_report), exist_ok=True)
    if not os.path.exists(os.path.dirname(output_table)):
        os.makedirs(os.path.dirname(output_table), exist_ok=True)

    print(f"Cargando dataset desde {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if limit and limit > 0:
        print(f"Limitando análisis a las primeras {limit} secuencias...")
        df = df.head(limit)
    
    results = []
    
    print(f"Analizando calidad de siluetas (muestreo de {sample_frames} frames por secuencia)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Intentar obtener el directorio desde la columna 'silhouette_dir'
        rel_path = row.get('silhouette_dir')
        
        if pd.isna(rel_path) or not rel_path:
            seq_id = row.get('sequence_id', row.get('id', str(idx)))
            seq_dir = os.path.join(silhouettes_dir, str(seq_id))
        else:
            seq_dir = os.path.join(silhouettes_dir, rel_path)
        
        # 2. Localizar imágenes
        img_paths = glob(os.path.join(seq_dir, "*.png")) or glob(os.path.join(seq_dir, "*.jpg"))
        
        if not img_paths:
            seq_id = row.get('sequence_id', str(idx))
            img_paths = glob(os.path.join(silhouettes_dir, f"*{seq_id}*.png"))
            
        if not img_paths:
            results.append({'fill_norm': 0.0, 'border_frac': 1.0, 'sqi': 0.0})
            continue
            
        # 3. Analizar una muestra de la secuencia
        actual_sample_size = min(sample_frames, len(img_paths))
        sample_indices = np.linspace(0, len(img_paths)-1, actual_sample_size, dtype=int)
        seq_samples = [img_paths[i] for i in sample_indices]
        
        f_norms, b_fracs = [], []
        for img_p in seq_samples:
            fn, bf = analyze_silhouette_image(img_p)
            if fn is not None:
                f_norms.append(fn)
                b_fracs.append(bf)
        
        if f_norms:
            avg_fn = np.mean(f_norms)
            avg_bf = np.mean(b_fracs)
            sqi = (1.0 - abs(0.7 - avg_fn) / 0.7) * (1.0 - avg_bf)
            results.append({'fill_norm': avg_fn, 'border_frac': avg_bf, 'sqi': np.clip(sqi, 0, 1)})
        else:
            results.append({'fill_norm': 0, 'border_frac': 1, 'sqi': 0})

    # (Resto del código de agrupación y guardado se mantiene igual...)
    analysis_df = pd.DataFrame(results)
    df['sqi'] = analysis_df['sqi'].values
    # ...
    df['fill_norm_calc'] = analysis_df['fill_norm']
    df['border_frac_calc'] = analysis_df['border_frac']

    # Manejo de etiquetas visuales (VI3/VI4)
    has_visual_attrs = all(col in df.columns for col in ['VI3', 'VI4'])
    
    # Agrupación Factorial
    vi4_order = ['Blanco dominante', 'Gris dominante', 'Negro dominante', 'Cálido dominante', 'Frío dominante', 'Mixto']
    vi3_order = ['Blanco dominante', 'Gris dominante', 'Negro dominante', 'Cálido dominante', 'Frío dominante', 'Mixto']
    table_index = pd.MultiIndex.from_product([vi4_order, vi3_order], names=['VI4', 'VI3'])
    results_table = pd.DataFrame(index=table_index)
    results_table['SQI_Mean'] = np.nan

    if has_visual_attrs:
        df['VI3'] = df['VI3'].astype(str).str.strip().str.capitalize()
        df['VI4'] = df['VI4'].astype(str).str.strip().str.capitalize()
        grouped = df.groupby(['VI4', 'VI3'])['sqi'].mean()
        for (v4, v3), val in grouped.items():
            try: results_table.loc[(v4, v3), 'SQI_Mean'] = val
            except: pass
        results_table.to_csv(output_table)
    
    # Guardar reporte
    sqi_global = df['sqi'].mean()
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE CALIDAD DE SILUETA DESDE IMÁGENES\n")
        f.write("============================================\n")
        f.write(f"SQI GLOBAL: {sqi_global:.4f}\n")
        f.write("Metodología: Análisis de fill_norm (convex-hull) y border_frac (boundary noise)\n")
        f.write(f"Dataset analizado: {input_csv}\n")
        f.write(f"Directorio siluetas: {silhouettes_dir}\n")

    print(f"\nProceso finalizado. SQI Global: {sqi_global:.4f}")
    print(f"Reporte: {output_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SQI desde imágenes de silueta.')
    parser.add_argument('--input', type=str, required=True, help='CSV del dataset')
    parser.add_argument('--silhouettes', type=str, required=True, help='Directorio base de imágenes')
    parser.add_argument('--report', type=str, default='outputs/reports/sqi_from_images_report.txt')
    parser.add_argument('--table', type=str, default='outputs/reports/sqi_factorial_table.csv')
    parser.add_argument('--limit', type=int, default=None, help='Limitar a N secuencias')
    parser.add_argument('--sample_frames', type=int, default=5, help='Frames por secuencia')
    
    args = parser.parse_args()
    calculate_sqi_from_images(args.input, args.silhouettes, args.report, args.table, args.limit, args.sample_frames)
