import pandas as pd
import numpy as np
import os
import argparse

def calculate_similarity(input_csv, output_report, output_table):
    """
    Calcula el Coeficiente de Similitud Inter-Clase (ICAS) Global y por Celda.
    Cada celda en la tabla final representará el coeficiente de similitud promedio
    de las identidades con ese perfil visual frente a todo el dataset.
    """
    if not os.path.exists(os.path.dirname(output_report)):
        os.makedirs(os.path.dirname(output_report), exist_ok=True)
    if not os.path.exists(os.path.dirname(output_table)):
        os.makedirs(os.path.dirname(output_table), exist_ok=True)

    if not os.path.exists(input_csv):
        print(f"Error: No se encontró el archivo {input_csv}")
        return

    print(f"Cargando datos desde {input_csv}...")
    df = pd.read_csv(input_csv)

    # Definir los atributos de interés
    attr_cols = ['VI1', 'VI2', 'VI3', 'VI4']
    
    # 1. Normalizar y limpiar datos
    for col in attr_cols:
        df[col] = df[col].astype(str).str.strip().str.capitalize()
        # Mapeo específico para coincidir con la imagen (Capitalized)
        df[col] = df[col].replace({
            'Nan': 'N/a', 'N/a': 'N/a', 'None': 'N/a', '': 'N/a',
            'Pantalón': 'Pantalón', 'Camisa': 'Camisa', 'Chompa': 'Chompa', 
            'Polo': 'Polo', 'Saco': 'Saco', 'Vestido superior': 'Vestido superior',
            'Vestido inferior': 'Vestido inferior'
        })

    # 2. Agrupar por persona (perfil modal)
    print("Calculando perfiles visuales por identidad...")
    identities = df.groupby('person_global_id')[attr_cols].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/a'
    ).reset_index()
    num_ids = len(identities)
    print(f"Total de identidades: {num_ids}")

    if num_ids < 2:
        print("Error: Se necesitan al menos 2 identidades para calcular similitud inter-clase.")
        return

    # 3. Calcular ICAS por identidad (Similitud promedio de cada persona vs el resto)
    print("Calculando matriz de similitudes (optimizado)...")
    attr_values = identities[attr_cols].values # Matrix N x 4
    
    # Usar broadcasting para comparar todos contra todos
    # (N, 1, 4) == (1, N, 4) -> (N, N, 4)
    matches_matrix = (attr_values[:, np.newaxis, :] == attr_values[np.newaxis, :, :]).sum(axis=2)
    
    # Similitud por persona: sumamos matches con otros (excluyendo el diagonal que es 4)
    # y dividimos por (4 atributos * (N-1) personas)
    id_similarities = (matches_matrix.sum(axis=1) - 4) / (4.0 * (num_ids - 1))
    identities['similarity_coeff'] = id_similarities

    # Coeficiente Global
    icas_global = id_similarities.mean()
    
    # Tasa de Colisión (Personas con similitud 1.0 exacta con alguien más)
    # Una colisión ocurre si hay algún match == 4 fuera de la diagonal
    collisions_count = (matches_matrix == 4).sum() - num_ids # Restar diagonal
    total_pairs = num_ids * (num_ids - 1)
    collision_rate = (collisions_count / total_pairs) * 100

    # 4. Generar Tabla Factorial de Coeficientes
    print("Generando tabla de coeficientes por celda...")
    
    vi4_order = ['Blanco dominante', 'Gris dominante', 'Negro dominante', 'Cálido dominante', 'Frío dominante', 'Mixto']
    vi3_order = ['Blanco dominante', 'Gris dominante', 'Negro dominante', 'Cálido dominante', 'Frío dominante', 'Mixto']
    vi2_order = ['Pantalón', 'Falda', 'Short', 'Vestido inferior']
    vi1_order = ['Polo', 'Camisa', 'Chompa', 'Saco', 'Vestido superior']

    rows_index = pd.MultiIndex.from_product([vi4_order, vi3_order], names=['VI4', 'VI3'])
    cols_index = pd.MultiIndex.from_product([vi2_order, vi1_order], names=['VI2', 'VI1'])
    
    # Tabla de coeficientes (promedio de similitud de las personas en esa celda)
    coeff_table = pd.DataFrame(0.0, index=rows_index, columns=cols_index)
    # Tabla de conteo (para el promedio)
    count_table = pd.DataFrame(0, index=rows_index, columns=cols_index)
    
    for _, row in identities.iterrows():
        v1, v2, v3, v4, coeff = row['VI1'], row['VI2'], row['VI3'], row['VI4'], row['similarity_coeff']
        try:
            coeff_table.loc[(v4, v3), (v2, v1)] += coeff
            count_table.loc[(v4, v3), (v2, v1)] += 1
        except KeyError:
            pass

    # Calcular promedio real (evitar división por cero)
    final_table = coeff_table / count_table.replace(0, np.nan)
    final_table = final_table.fillna(0.0)

    # 4.1 Verificación Matemática: ICAS Global desde la Tabla
    # El ICAS Global es el promedio ponderado de los coeficientes de cada celda
    weighted_sum = (final_table.values * count_table.values).sum()
    total_identities = count_table.values.sum()
    icas_from_table = weighted_sum / total_identities if total_identities > 0 else 0

    # 5. Guardar resultados
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("REPORTE TÉCNICO: COEFICIENTES DE SIMILITUD INTER-CLASE (FACTORIAL)\n")
        f.write("==================================================================\n")
        f.write(f"Dataset: {os.path.basename(input_csv)}\n")
        f.write(f"Identidades procesadas (N): {num_ids}\n\n")
        f.write("1. MÉTRICAS GLOBALES\n")
        f.write("--------------------\n")
        f.write(f"S-GLOBAL (ICAS): {icas_global:.4f}\n")
        f.write(f"TASA DE COLISIÓN PROMEDIO: {collision_rate:.4f}%\n\n")
        f.write("2. SUSTENTACIÓN PARA LA TESIS (Derivación):\n")
        f.write("-------------------------------------------\n")
        f.write("El ICAS global puede derivarse de la tabla factorial mediante un promedio ponderado:\n")
        f.write("ICAS_global = [ Sumatoria( Coef_celda_k * Poblacion_celda_k ) ] / N_total\n")
        f.write(f"Verificación del cálculo desde la tabla: {icas_from_table:.4f}\n")
        f.write(f"Diferencia: {abs(icas_global - icas_from_table):.6f} (Cercana a 0 indica validez)\n\n")
        f.write("3. INTERPRETACIÓN DE LA TABLA (CSV):\n")
        f.write("------------------------------------\n")
        f.write("Cada celda representa el grado de ambigüedad visual de ese perfil específico.\n")
        f.write("Un coeficiente de 1.0 en una celda significa que todas las personas con esa ropa\n")
        f.write("son idénticas al resto de personas que comparten ese perfil (colisión local total).\n")

    final_table.to_csv(output_table)
    
    print("\n--- RESULTADOS ---")
    print(f"S-Global (Calculado): {icas_global:.4f}")
    print(f"S-Global (Desde tabla): {icas_from_table:.4f}")
    print(f"Colisiones: {collision_rate:.4f}%")
    print(f"Reporte técnico guardado en: {output_report}")
    print(f"Tabla de Coeficientes guardada en: {output_table}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcula similitud inter-clase con derivación factorial.')
    parser.add_argument('--input', type=str, default='outputs/metadata/sequences_labeled_full.csv')
    parser.add_argument('--report', type=str, default='outputs/reports/dataset_similarity.txt')
    parser.add_argument('--table', type=str, default='outputs/reports/similarity_coefficient_table.csv')
    
    args = parser.parse_args()
    calculate_similarity(args.input, args.report, args.table)
