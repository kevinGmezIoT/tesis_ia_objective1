import os
import base64
import argparse
import json
import time
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from openai import OpenAI
from typing import Tuple, List, Optional, Dict

# Configuración de categorías
CATEGORIES_VI1 = ["Polo", "camisa", "chompa", "saco", "vestido superior", "N/A"]
CATEGORIES_VI2 = ["Pantalón", "falda", "short", "vestido inferior", "N/A"]
CATEGORIES_COLOR = ["Blanco dominante", "Gris dominante", "Negro dominante", "Cálido dominante", "Frío dominante", "Mixto"]

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_clothing_labels(image_path: str, client: OpenAI) -> Tuple[str, str]:
    """Usa OpenAI Multimodal para obtener etiquetas de prendas."""
    base64_image = encode_image(image_path)
    
    prompt = f"""
    Analiza la imagen de la persona y clasifica su vestimenta superior e inferior según estas categorías:
    VI1 (Superior): {', '.join(CATEGORIES_VI1)}
    VI2 (Inferior): {', '.join(CATEGORIES_VI2)}
    
    Regla importante: Si eliges 'vestido superior', la inferior DEBE ser 'vestido inferior', y viceversa.
    Responde ÚNICAMENTE en formato JSON con las llaves "VI1" y "VI2".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        vi1 = data.get("VI1", "Polo")
        vi2 = data.get("VI2", "Pantalón")
        
        # Aplicar regla de consistencia por si el modelo falla
        if vi1 == "vestido superior" or vi2 == "vestido inferior":
            vi1 = "vestido superior"
            vi2 = "vestido inferior"
            
        return vi1, vi2
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Desconocido", "Desconocido"

def classify_color_distribution(img_rgb: np.ndarray, mask: np.ndarray) -> str:
    """Clasifica la distribución de color usando OpenCV."""
    if mask.sum() == 0:
        return "Mixto"
    
    # Extraer píxeles de interés
    pixels = img_rgb[mask > 0]
    
    # Convertir a HSV para mejor análisis de color
    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    
    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]
    
    # Definir umbrales para acromáticos (Blanco, Gris, Negro)
    is_black = (v < 50)
    is_white = (v > 200) & (s < 30)
    is_gray = (s < 30) & (~is_black) & (~is_white)
    
    total = len(pixels)
    p_black = is_black.sum() / total
    p_white = is_white.sum() / total
    p_gray = is_gray.sum() / total
    
    if p_white > 0.6: return "Blanco dominante"
    if p_black > 0.6: return "Negro dominante"
    if p_gray > 0.6: return "Gris dominante"
    
    # Análisis de colores cromáticos
    is_chromatic = ~(is_black | is_white | is_gray)
    if is_chromatic.sum() / total < 0.2:
        # Si casi no hay color, elegir el acromático predominante
        vals = [p_white, p_black, p_gray]
        idx = np.argmax(vals)
        return ["Blanco dominante", "Negro dominante", "Gris dominante"][idx]
    
    h_chromatic = h[is_chromatic]
    # Cálidos: Rojo, Naranja, Amarillo (0-30, 150-180)
    # Fríos: Verde, Cian, Azul, Violeta (30-150)
    is_warm = (h_chromatic < 30) | (h_chromatic > 150)
    is_cold = (h_chromatic >= 30) & (h_chromatic <= 150)
    
    p_warm = is_warm.sum() / total
    p_cold = is_cold.sum() / total
    
    if p_warm > 0.5: return "Cálido dominante"
    if p_cold > 0.5: return "Frío dominante"
    
    return "Mixto"

def main():
    parser = argparse.ArgumentParser(description="Etiqueta secuencias con prendas y colores.")
    parser.add_argument("--base_dir", type=str, default=".", help="Directorio base de las imágenes")
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--output_csv", type=str, required=True, help="Ruta al CSV de salida")
    parser.add_argument("--limit", type=int, default=None, help="Límite de secuencias a procesar")
    parser.add_argument("--sleep", type=float, default=0.5, help="Tiempo de espera entre peticiones a OpenAI")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API Key (alternativa a variable de entorno)")
    
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Debe proporcionar una OpenAI API Key via --api_key o variable de entorno OPENAI_API_KEY")
    
    client = OpenAI(api_key=api_key)
    
    # Cargar CSV de salida si existe para retomar progreso
    if os.path.exists(args.output_csv):
        print(f"Cargando progreso existente de {args.output_csv}...")
        df_out = pd.read_csv(args.output_csv)
        # Asegurar que las columnas existan
        for col in ["VI1", "VI2", "VI3", "VI4"]:
            if col not in df_out.columns:
                df_out[col] = None
    else:
        df_out = pd.read_csv(args.input_csv)
        if args.limit:
            df_out = df_out.head(args.limit)
        for col in ["VI1", "VI2", "VI3", "VI4"]:
            df_out[col] = None
    
    # Identificar secuencias pendientes
    # Se consideran pendientes si alguna de las VIs es nula o Error
    mask_pending = df_out["VI1"].isna() | (df_out["VI1"] == "Error")
    df_pending = df_out[mask_pending]
    
    if len(df_pending) == 0:
        print("No hay secuencias pendientes para procesar.")
        return

    print(f"Procesando {len(df_pending)} secuencias pendientes...")
    
    for idx, row in tqdm(df_pending.iterrows(), total=len(df_pending)):
        seq_id = row['sequence_id']
        rgb_dir = os.path.join(args.base_dir, row['rgb_dir'])
        sil_dir = os.path.join(args.base_dir, row['silhouette_dir'])
        
        # Obtener primera imagen
        # Asumiendo que frame_numbers_json contiene los números de frame
        try:
            frames = json.loads(row['frame_numbers_json'])
            first_frame = frames[10]
            rgb_path = os.path.join(rgb_dir, f"{first_frame:06d}.jpg")
            sil_path = os.path.join(sil_dir, f"{first_frame:06d}.png")
            
            if not os.path.exists(rgb_path):
                # Intentar buscar el primer archivo en el directorio si el nombre no coincide
                rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
                if not rgb_files: raise FileNotFoundError(f"No hay JPG en {rgb_dir}")
                rgb_path = os.path.join(rgb_dir, rgb_files[10])
                sil_files = sorted([f for f in os.listdir(sil_dir) if f.endswith(".png")])
                sil_path = os.path.join(sil_dir, sil_files[10])

            # 1. Obtener prendas con OpenAI
            vi1, vi2 = get_clothing_labels(rgb_path, client)
            
            # 2. Análisis de color con OpenCV
            img_bgr = cv2.imread(rgb_path)
            if img_bgr is None: raise FileNotFoundError(rgb_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(sil_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise FileNotFoundError(sil_path)
            
            # VI3: Color del individuo
            vi3 = classify_color_distribution(img_rgb, mask)
            
            # VI4: Color del fondo
            bg_mask = cv2.bitwise_not(mask)
            vi4 = classify_color_distribution(img_rgb, bg_mask)
            
            # Actualizar dataframe y guardar inmediatamente
            df_out.at[idx, "VI1"] = vi1
            df_out.at[idx, "VI2"] = vi2
            df_out.at[idx, "VI3"] = vi3
            df_out.at[idx, "VI4"] = vi4
            
            df_out.to_csv(args.output_csv, index=False)
            
            if args.sleep > 0:
                time.sleep(args.sleep)
            
        except Exception as e:
            print(f"Error procesando {seq_id}: {e}")
            df_out.at[idx, "VI1"] = "Error"
            df_out.at[idx, "VI2"] = "Error"
            df_out.at[idx, "VI3"] = "Error"
            df_out.at[idx, "VI4"] = "Error"
            df_out.to_csv(args.output_csv, index=False)
            
    print(f"Proceso completado. Resultados guardados en {args.output_csv}")

if __name__ == "__main__":
    main()
