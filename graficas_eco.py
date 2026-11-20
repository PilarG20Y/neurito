"""Genera gráficas de barras con media y desviación estándar desde resultados_detalle.csv."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generar_graficas(
    csv_path: str = "evidencia_eco/resultados_detalle.csv",
    salida_dir: str = "evidencia_eco",
):
    ruta_csv = Path(csv_path)
    salida = Path(salida_dir)
    salida.mkdir(parents=True, exist_ok=True)

    filas = []
    with ruta_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filas.append(row)

    if not filas:
        raise ValueError(f"No hay datos en {ruta_csv}")

    escenarios = list(dict.fromkeys(r["escenario"] for r in filas))
    politicas = list(dict.fromkeys(r["politica"] for r in filas))

    metricas = [
        ("supervivencia_presas", "Supervivencia de presas", "supervivencia_por_escenario.png"),
        ("plantas_comidas", "Plantas consumidas", "plantas_por_escenario.png"),
        ("capturas", "Capturas de depredadores", "capturas_por_escenario.png"),
        ("energia_presas", "Energía final promedio de presas", "energia_presas_por_escenario.png"),
    ]

    for campo, titulo, nombre_archivo in metricas:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        x = np.arange(len(escenarios))
        ancho = 0.82 / max(1, len(politicas))

        for idx, pol in enumerate(politicas):
            medias = []
            stds = []
            for esc in escenarios:
                vals = [
                    float(r[campo])
                    for r in filas
                    if r["escenario"] == esc and r["politica"] == pol
                ]
                arr = np.array(vals, dtype=np.float64)
                medias.append(float(arr.mean()) if len(arr) else 0.0)
                stds.append(float(arr.std()) if len(arr) else 0.0)

            desplazamiento = (idx - (len(politicas) - 1) / 2.0) * ancho
            ax.bar(
                x + desplazamiento,
                medias,
                width=ancho,
                yerr=stds,
                capsize=3,
                label=pol,
            )

        ax.set_title(f"{titulo} por escenario (media ± std)")
        ax.set_xticks(x)
        ax.set_xticklabels(escenarios)
        ax.set_ylabel(campo)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Política", loc="best")
        fig.tight_layout()
        fig.savefig(salida / nombre_archivo, dpi=150)
        plt.close(fig)

    print(f"Gráficas guardadas en: {salida.resolve()}")


if __name__ == "__main__":
    generar_graficas()