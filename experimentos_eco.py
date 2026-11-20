"""Experimentos reproducibles para NeuroEcosistema (3 escenarios, semillas y métricas)."""
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import random
import numpy as np

from ecosistema_red import (
    RedEspecieMLP,
    generar_datos_especie,
)
from laberinto_mundo import Laberinto, vecino, valido


@dataclass
class EscenarioEco:
    nombre: str
    indice_mapa: int
    num_plantas: int
    num_presas: int
    num_depredadores: int


ESCENARIOS = (
    EscenarioEco("A_recursos_suficientes", indice_mapa=1, num_plantas=14, num_presas=6, num_depredadores=1),
    EscenarioEco("B_escasez_alimento", indice_mapa=1, num_plantas=3, num_presas=6, num_depredadores=2),
    EscenarioEco("C_alta_depredacion", indice_mapa=2, num_plantas=8, num_presas=6, num_depredadores=5),
)


class PoliticaAleatoria:
    """Agente de referencia que elige al azar entre acciones válidas."""

    def __init__(self, semilla: int):
        self.rng = random.Random(semilla)

    def elegir_accion(self, obs: np.ndarray, solo_validas: bool = True) -> int:
        if solo_validas:
            validas = [i for i in range(4) if obs[1 + i] > 0.5]
            if validas:
                return self.rng.choice(validas)
        return self.rng.randrange(4)


def construir_redes(semilla: int, epocas: int = 200, o1: int = 16, o2: int = 12, lr: float = 0.08):
    (Xp_tr, yp_tr), (Xp_te, yp_te) = generar_datos_especie("presa", n_muestras=1000, semilla=semilla)
    (Xd_tr, yd_tr), (Xd_te, yd_te) = generar_datos_especie("depredador", n_muestras=1000, semilla=semilla + 1)

    red_p_antes = RedEspecieMLP(11, oculta1_dim=o1, oculta2_dim=o2, lr=lr, semilla=semilla)
    red_d_antes = RedEspecieMLP(7, oculta1_dim=o1, oculta2_dim=o2, lr=lr, semilla=semilla + 10)

    red_p_despues = RedEspecieMLP(11, oculta1_dim=o1, oculta2_dim=o2, lr=lr, semilla=semilla)
    red_d_despues = RedEspecieMLP(7, oculta1_dim=o1, oculta2_dim=o2, lr=lr, semilla=semilla + 10)

    perdida_p = red_p_despues.entrenar(Xp_tr, yp_tr, epocas=epocas)
    perdida_d = red_d_despues.entrenar(Xd_tr, yd_tr, epocas=epocas)

    return {
        "antes": (red_p_antes, red_d_antes),
        "despues": (red_p_despues, red_d_despues),
        "acc_presa": red_p_despues.exactitud(Xp_te, yp_te),
        "acc_dep": red_d_despues.exactitud(Xd_te, yd_te),
        "perdida_p_ini": perdida_p[0] if perdida_p else None,
        "perdida_p_fin": perdida_p[-1] if perdida_p else None,
        "perdida_d_ini": perdida_d[0] if perdida_d else None,
        "perdida_d_fin": perdida_d[-1] if perdida_d else None,
    }


def simular_episodio(escenario: EscenarioEco, semilla: int, red_presa=None, red_dep=None, pasos_max: int = 80):
    mundo = Laberinto(
        indice=escenario.indice_mapa,
        semilla=semilla,
        num_presas=escenario.num_presas,
        num_depredadores=escenario.num_depredadores,
        num_plantas=escenario.num_plantas,
        red_presa=red_presa,
        red_depredador=red_dep,
    )

    capturas_previas = 0
    pasos_captura = []

    for paso in range(1, pasos_max + 1):
        mundo.paso_ecosistema(accion_neurito=None)
        if mundo.capturas > capturas_previas:
            nuevas = mundo.capturas - capturas_previas
            pasos_captura.extend([paso] * nuevas)
            capturas_previas = mundo.capturas

    presas_vivas = sum(1 for p in mundo.presas if p.vivo)
    supervivencia = presas_vivas / max(1, len(mundo.presas))
    energia_presas = float(np.mean([p.energia for p in mundo.presas])) if mundo.presas else 0.0
    energia_deps = float(np.mean([d.energia for d in mundo.depredadores])) if mundo.depredadores else 0.0
    tiempo_medio_captura = float(np.mean(pasos_captura)) if pasos_captura else float(pasos_max)

    return {
        "escenario": escenario.nombre,
        "semilla": semilla,
        "supervivencia_presas": supervivencia,
        "plantas_comidas": mundo.plantas_comidas,
        "capturas": mundo.capturas,
        "energia_presas": energia_presas,
        "energia_deps": energia_deps,
        "tiempo_medio_captura": tiempo_medio_captura,
    }


def correr_experimentos(semillas=(10, 20, 30, 40, 50), pasos_max: int = 80, carpeta_salida: str = "evidencia_eco"):
    salida = Path(carpeta_salida)
    salida.mkdir(parents=True, exist_ok=True)

    filas_detalle = []
    resumen = []

    print("ESCENARIO | POLITICA | SUP_PRESAS | PLANTAS_COM | CAPTURAS | ENERG_PRESA | T_CAPTURA")
    print("-" * 92)

    for esc in ESCENARIOS:
        for politica_nombre in ("aleatorio", "reglas", "sin_entrenar", "mlp_16_12", "mlp_24_16"):
            metricas_semillas = []
            for s in semillas:
                redes_base = construir_redes(semilla=s, epocas=200, o1=16, o2=12, lr=0.08)
                redes_alt = construir_redes(semilla=s, epocas=200, o1=24, o2=16, lr=0.05)

                if politica_nombre == "aleatorio":
                    rp, rd = PoliticaAleatoria(s), PoliticaAleatoria(s + 100)
                elif politica_nombre == "reglas":
                    rp, rd = None, None
                elif politica_nombre == "sin_entrenar":
                    rp, rd = redes_base["antes"]
                elif politica_nombre == "mlp_16_12":
                    rp, rd = redes_base["despues"]
                else:
                    rp, rd = redes_alt["despues"]

                m = simular_episodio(esc, semilla=s, red_presa=rp, red_dep=rd, pasos_max=pasos_max)
                fila = {"politica": politica_nombre, **m}
                filas_detalle.append(fila)
                metricas_semillas.append(m)

            sup_arr = np.array([x["supervivencia_presas"] for x in metricas_semillas], dtype=np.float64)
            pla_arr = np.array([x["plantas_comidas"] for x in metricas_semillas], dtype=np.float64)
            cap_arr = np.array([x["capturas"] for x in metricas_semillas], dtype=np.float64)
            ene_arr = np.array([x["energia_presas"] for x in metricas_semillas], dtype=np.float64)
            tca_arr = np.array([x["tiempo_medio_captura"] for x in metricas_semillas], dtype=np.float64)

            print(
                f"{esc.nombre:<22} | {politica_nombre:<12} | "
                f"{sup_arr.mean():>5.2f}±{sup_arr.std():>4.2f} | {pla_arr.mean():>5.1f}±{pla_arr.std():>4.1f} | "
                f"{cap_arr.mean():>4.1f}±{cap_arr.std():>3.1f} | {ene_arr.mean():>5.1f}±{ene_arr.std():>4.1f} | "
                f"{tca_arr.mean():>5.1f}"
            )

            resumen.append({
                "escenario": esc.nombre,
                "politica": politica_nombre,
                "semillas": list(semillas),
                "pasos_max": pasos_max,
                "sup_media": float(sup_arr.mean()),
                "sup_std": float(sup_arr.std()),
                "plantas_media": float(pla_arr.mean()),
                "plantas_std": float(pla_arr.std()),
                "capturas_media": float(cap_arr.mean()),
                "capturas_std": float(cap_arr.std()),
                "energia_presas_media": float(ene_arr.mean()),
                "energia_presas_std": float(ene_arr.std()),
                "tiempo_captura_media": float(tca_arr.mean()),
            })

    csv_path = salida / "resultados_detalle.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_detalle[0].keys()))
        writer.writeheader()
        writer.writerows(filas_detalle)

    json_path = salida / "resumen_experimentos.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "escenarios": [asdict(e) for e in ESCENARIOS],
                "semillas": list(semillas),
                "pasos_max": pasos_max,
                "resumen": resumen,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nEvidencia guardada en: {csv_path} y {json_path}")


if __name__ == "__main__":
    correr_experimentos()