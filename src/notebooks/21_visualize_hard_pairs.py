#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_middle_frame(df_frames, pid, base_dir):
    """Load the middle RGB frame associated with an identity."""
    frames = df_frames[df_frames["person_global_id"] == pid]

    if frames.empty:
        return None

    mid = len(frames) // 2
    rel_path = frames.iloc[mid]["rgb_source_path"]
    img_path = os.path.join(base_dir, rel_path)

    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path)

    if img is None:
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_attributes(df_attrs, pid):
    """Return the first attribute record associated with an identity."""
    rows = df_attrs[df_attrs["person_global_id"] == pid]
    return None if rows.empty else rows.iloc[0]


GARMENTS_ES = {
    'jacket': 'Chaqueta', 'pants': 'Pantalón', 'hoodie': 'Polera', 
    'tshirt': 'Camiseta', 'shirt': 'Camisa', 'shorts': 'Pantalón corto',
    'skirt': 'Falda', 'dress': 'Vestido', 'coat': 'Abrigo'
}
COLORS_ES = {
    'blue': 'Azul', 'black': 'Negro', 'gray': 'Gris', 'red': 'Rojo', 
    'white': 'Blanco', 'green': 'Verde', 'yellow': 'Amarillo',
    'purple': 'Morado', 'pink': 'Rosa', 'orange': 'Naranja', 'brown': 'Marrón'
}
YESNO_ES = {'yes': 'Sí', 'no': 'No'}

TEXTS = {
    'en': {
        'unknown': 'Unknown',
        'sim_scores': 'Similarity scores',
        'identity': 'Identity',
        'upper': 'Upper',
        'lower': 'Lower',
        'backpack': 'Backpack'
    },
    'es': {
        'unknown': 'Desconocido',
        'sim_scores': 'Puntajes de similitud',
        'identity': 'Identidad',
        'upper': 'Superior',
        'lower': 'Inferior',
        'backpack': 'Mochila'
    }
}

def translate_attr(val, mapping, lang):
    if lang == 'en' or not isinstance(val, str):
        return str(val).title()
    val_lower = val.lower()
    return mapping.get(val_lower, val.title())

def format_attributes(attr, lang='en'):
    """Format semantic attributes for compact visualization."""
    if attr is None:
        return {
            "Upper garment": TEXTS[lang]['unknown'],
            "Lower garment": TEXTS[lang]['unknown'],
            "Backpack": TEXTS[lang]['unknown'],
        }

    return {
        "Upper garment": (
            f"{translate_attr(attr['upper_garment_type'], GARMENTS_ES, lang)} "
            f"({translate_attr(attr['upper_color_label'], COLORS_ES, lang)})"
        ),
        "Lower garment": (
            f"{translate_attr(attr['lower_garment_type'], GARMENTS_ES, lang)} "
            f"({translate_attr(attr['lower_color_label'], COLORS_ES, lang)})"
        ),
        "Backpack": translate_attr(attr["backpack_flag"], YESNO_ES, lang),
    }


def add_identity_image(ax, img, pid):
    """Render an identity crop with a compact title."""
    if img is not None:
        ax.imshow(img)

    ax.set_title(
        f"ID {pid:04d}",
        fontsize=9,
        fontweight="semibold",
        pad=4,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.70")


def add_pair_summary(ax, pid_x, pid_y, attr_x, attr_y, gower_sim, osnet_sim, lang='en'):
    """Render metrics and semantic attributes in an academic-style panel."""
    ax.axis("off")

    attrs_x = format_attributes(attr_x, lang)
    attrs_y = format_attributes(attr_y, lang)

    metrics_text = (
        rf"$S_{{\mathrm{{Gower}}}} = {gower_sim:.2f}$"
        "\n"
        rf"$S_{{\mathrm{{OSNet}}}} = {osnet_sim:.2f}$"
    )

    ax.text(
        0.02,
        0.93,
        TEXTS[lang]['sim_scores'],
        fontsize=9,
        fontweight="semibold",
        va="top",
    )

    ax.text(
        0.02,
        0.80,
        metrics_text,
        fontsize=9,
        va="top",
        linespacing=1.45,
    )

    table_data = [
        [TEXTS[lang]['identity'], f"ID {pid_x:04d}", f"ID {pid_y:04d}"],
        [TEXTS[lang]['upper'], attrs_x["Upper garment"], attrs_y["Upper garment"]],
        [TEXTS[lang]['lower'], attrs_x["Lower garment"], attrs_y["Lower garment"]],
        [TEXTS[lang]['backpack'], attrs_x["Backpack"], attrs_y["Backpack"]],
    ]

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        colWidths=[0.24, 0.38, 0.38],
        bbox=[0.00, 0.05, 0.98, 0.52],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.45)
        cell.PAD = 0.05

        if row == 0:
            cell.set_text_props(fontweight="semibold")
            cell.set_facecolor("0.94")

        if col == 0:
            cell.set_text_props(fontweight="semibold")


def main():
    pairs_csv = "D:/tesis_dev/result1/dataset/output/metadata/gower_osnet_cross_analysis.csv"
    frames_csv = "D:/tesis_dev/result1/dataset/output/metadata/gait_cycle_frames.csv"
    attrs_csv = "D:/tesis_dev/result1/dataset/output/metadata/identity_visual_attributes_pair_ready.csv"

    base_dir = "D:/tesis_dev/result1/dataset/output"
    
    out_dirs = {
        'en': "D:/tesis_dev/result1/src/paper/english/figures",
        'es': "D:/tesis_dev/result1/src/paper/spanish/figures"
    }

    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    df_pairs = pd.read_csv(pairs_csv)
    df_frames = pd.read_csv(frames_csv)
    df_attrs = pd.read_csv(attrs_csv)

    top_pairs = (
        df_pairs[df_pairs["ambiguity_quadrant"] == "ambiguedad_confirmada"]
        .sort_values(by="osnet_similarity", ascending=False)
        .head(3)
    )

    # Academic-oriented defaults.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )

    for lang in ['en', 'es']:
        fig = plt.figure(figsize=(7.1, 8.2))

        grid = GridSpec(
            nrows=3,
            ncols=3,
            figure=fig,
            width_ratios=[1.0, 1.0, 1.65],
            height_ratios=[1.0, 1.0, 1.0],
            left=0.055,
            right=0.985,
            top=0.975,
            bottom=0.045,
            wspace=0.16,
            hspace=0.20,
        )

        row_labels = ["(a)", "(b)", "(c)"]

        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            pid_x = int(str(row["identity_x"]).replace("ID", ""))
            pid_y = int(str(row["identity_y"]).replace("ID", ""))

            img_x = load_middle_frame(df_frames, pid_x, base_dir)
            img_y = load_middle_frame(df_frames, pid_y, base_dir)

            attr_x = get_attributes(df_attrs, pid_x)
            attr_y = get_attributes(df_attrs, pid_y)

            ax_x = fig.add_subplot(grid[idx, 0])
            ax_y = fig.add_subplot(grid[idx, 1])
            ax_info = fig.add_subplot(grid[idx, 2])

            add_identity_image(ax_x, img_x, pid_x)
            add_identity_image(ax_y, img_y, pid_y)

            add_pair_summary(
                ax=ax_info,
                pid_x=pid_x,
                pid_y=pid_y,
                attr_x=attr_x,
                attr_y=attr_y,
                gower_sim=row["gower_similarity"],
                osnet_sim=row["osnet_similarity"],
                lang=lang
            )

            # Row label for referencing the examples in the manuscript.
            import matplotlib.transforms as transforms
            trans = transforms.blended_transform_factory(fig.transFigure, ax_x.transAxes)
            ax_x.text(
                0.015,
                1.03,
                row_labels[idx],
                transform=trans,
                fontsize=10,
                fontweight="semibold",
                va="bottom",
            )

        png_path = os.path.join(out_dirs[lang], "hard_pairs_examples_academic.png")
        pdf_path = os.path.join(out_dirs[lang], "hard_pairs_examples_academic.pdf")

        fig.savefig(
            png_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.03,
        )

        plt.close(fig)

        print(f"[{lang}] Saved PNG to: {png_path}")
        print(f"[{lang}] Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()