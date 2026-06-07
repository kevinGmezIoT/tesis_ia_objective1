import json
import sys
import os

def py_to_ipynb(py_path):
    if not os.path.exists(py_path):
        print(f"Error: No se encontró {py_path}")
        return

    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    # Dividir el archivo por la marca de celda
    chunks = content.split('# %%')
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        lines = chunk.strip('\n').split('\n')
        
        # Verificar si es una celda markdown
        if lines and lines[0].strip() == '[markdown]':
            cell_type = 'markdown'
            # Quitar el '[markdown]' y los '#' del inicio
            source = [line.lstrip('# ') + '\n' for line in lines[1:]]
        else:
            cell_type = 'code'
            source = [line + '\n' for line in lines]

        if source:
            # Quitar el último salto de línea del último elemento
            source[-1] = source[-1].rstrip('\n')
            
            cells.append({
                "cell_type": cell_type,
                "metadata": {},
                "source": source
            })
            if cell_type == 'code':
                cells[-1]["outputs"] = []
                cells[-1]["execution_count"] = None

    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = py_path.replace('.py', '.ipynb')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
        
    print(f"¡Convertido exitosamente! Guardado como: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python py_to_ipynb.py <archivo.py>")
    else:
        py_to_ipynb(sys.argv[1])
