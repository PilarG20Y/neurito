"""Redes neuronales multicapa (2 capas ocultas por especie) para presas y depredadores."""
from dataclasses import dataclass, field
import numpy as np

from laberinto_mundo import DIRECCIONES, vecino, valido


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    desplazado = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(desplazado)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _norm_celdas(mapa, dist_rc: float) -> float:
    max_dim = max(1.0, float(max(len(mapa), len(mapa[0]))))
    return float(np.clip(dist_rc / max_dim, -1.0, 1.0))


def observar_presa(mundo, presa) -> np.ndarray:
    """Vector de 11 características normalizadas percibidas por una presa."""
    f, c = presa.pos
    energia_norm = float(np.clip(presa.energia / 100.0, 0.0, 1.0))
    libres_4d = [
        1.0 if valido(mundo.mapa, vecino((f, c), a)) else 0.0 for a in range(4)
    ]

    if mundo.plantas:
        pf, pc = min(
            mundo.plantas,
            key=lambda p: abs(p[0] - f) + abs(p[1] - c),
        )
        df_planta = _norm_celdas(mundo.mapa, pf - f)
        dc_planta = _norm_celdas(mundo.mapa, pc - c)
    else:
        df_planta, dc_planta = 0.0, 0.0

    deps_vivos = [d for d in mundo.depredadores if d.vivo]
    if deps_vivos:
        dep_cercano = min(
            deps_vivos,
            key=lambda d: abs(d.pos[0] - f) + abs(d.pos[1] - c),
        )
        df_c, dc_c = dep_cercano.pos
        df_dep = _norm_celdas(mundo.mapa, df_c - f)
        dc_dep = _norm_celdas(mundo.mapa, dc_c - c)
        energia_dep_norm = float(np.clip(dep_cercano.energia / 120.0, 0.0, 1.0))
    else:
        df_dep, dc_dep = 0.0, 0.0
        energia_dep_norm = 0.0

    plantas_norm = float(
        np.clip(len(mundo.plantas) / max(1.0, float(mundo.num_plantas)), 0.0, 1.0)
    )

    return np.array(
        [
            energia_norm,
            *libres_4d,
            df_planta,
            dc_planta,
            df_dep,
            dc_dep,
            energia_dep_norm,
            plantas_norm,
        ],
        dtype=np.float64,
    )


def observar_depredador(mundo, dep) -> np.ndarray:
    """Vector de 7 características normalizadas percibidas por un depredador."""
    f, c = dep.pos
    energia_norm = float(np.clip(dep.energia / 120.0, 0.0, 1.0))
    libres_4d = [
        1.0 if valido(mundo.mapa, vecino((f, c), a)) else 0.0 for a in range(4)
    ]

    presas_vivas = [p.pos for p in mundo.presas if p.vivo]
    if presas_vivas:
        pf, pc = min(
            presas_vivas,
            key=lambda p: abs(p[0] - f) + abs(p[1] - c),
        )
        df_presa = _norm_celdas(mundo.mapa, pf - f)
        dc_presa = _norm_celdas(mundo.mapa, pc - c)
    else:
        df_presa, dc_presa = 0.0, 0.0

    return np.array(
        [energia_norm, *libres_4d, df_presa, dc_presa],
        dtype=np.float64,
    )


def _accion_objetivo_presa(obs: np.ndarray) -> int:
    """Política guía para generar ejemplos de entrenamiento de presas."""
    libres_4d = obs[1:5]
    df_planta, dc_planta = obs[5], obs[6]
    df_dep, dc_dep = obs[7], obs[8]
    dist_dep = abs(df_dep) + abs(dc_dep)

    if dist_dep > 0 and dist_dep < 0.35:
        mejor_a = None
        mejor_puntaje = -1e9
        for a, (df, dc) in enumerate(DIRECCIONES):
            if libres_4d[a] < 0.5:
                continue
            alejar = df * df_dep + dc * dc_dep
            puntaje = -alejar + 0.3 * (df * df_planta + dc * dc_planta)
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_a = a
        if mejor_a is not None:
            return mejor_a

    mejor_a = None
    mejor_puntaje = -1e9
    for a, (df, dc) in enumerate(DIRECCIONES):
        if libres_4d[a] < 0.5:
            continue
        acercar = df * df_planta + dc * dc_planta
        if acercar > mejor_puntaje:
            mejor_puntaje = acercar
            mejor_a = a
    if mejor_a is not None:
        return mejor_a

    validas = [a for a in range(4) if libres_4d[a] >= 0.5]
    return validas[0] if validas else 0


def _accion_objetivo_depredador(obs: np.ndarray) -> int:
    """Política guía para generar ejemplos de entrenamiento de depredadores."""
    libres_4d = obs[1:5]
    df_presa, dc_presa = obs[5], obs[6]

    mejor_a = None
    mejor_puntaje = -1e9
    for a, (df, dc) in enumerate(DIRECCIONES):
        if libres_4d[a] < 0.5:
            continue
        acercar = df * df_presa + dc * dc_presa
        if acercar > mejor_puntaje:
            mejor_puntaje = acercar
            mejor_a = a
    if mejor_a is not None:
        return mejor_a

    validas = [a for a in range(4) if libres_4d[a] >= 0.5]
    return validas[0] if validas else 0


def generar_datos_especie(especie: str, n_muestras: int = 1200, semilla: int = 42):
    """Genera entradas y etiquetas divididas en entrenamiento (80%) y prueba (20%)."""
    rng = np.random.default_rng(semilla)
    dim = 11 if especie == 'presa' else 7
    X = np.zeros((n_muestras, dim), dtype=np.float64)
    y = np.zeros(n_muestras, dtype=np.int64)

    for i in range(n_muestras):
        energia = float(rng.uniform(0.1, 1.0))
        libres_4d = rng.integers(0, 2, size=4).astype(np.float64)
        if not np.any(libres_4d > 0):
            libres_4d[int(rng.integers(0, 4))] = 1.0
        df_obj = float(rng.uniform(-0.8, 0.8))
        dc_obj = float(rng.uniform(-0.8, 0.8))

        if especie == 'presa':
            df_dep = float(rng.uniform(-0.8, 0.8))
            dc_dep = float(rng.uniform(-0.8, 0.8))
            energia_dep_norm = float(rng.uniform(0.0, 1.0))
            plantas_norm = float(rng.uniform(0.0, 1.0))
            fila = np.array(
                [
                    energia,
                    *libres_4d,
                    df_obj,
                    dc_obj,
                    df_dep,
                    dc_dep,
                    energia_dep_norm,
                    plantas_norm,
                ],
                dtype=np.float64,
            )
            y[i] = _accion_objetivo_presa(fila)
        else:
            fila = np.array(
                [energia, *libres_4d, df_obj, dc_obj],
                dtype=np.float64,
            )
            y[i] = _accion_objetivo_depredador(fila)
        X[i] = fila

    corte = int(0.8 * n_muestras)
    return (X[:corte], y[:corte]), (X[corte:], y[corte:])


@dataclass
class RedEspecieMLP:
    """Red neuronal multicapa con 2 capas ocultas (ReLU -> ReLU -> Softmax)."""

    entrada_dim: int
    oculta1_dim: int = 16
    oculta2_dim: int = 12
    salida_dim: int = 4
    lr: float = 0.08
    semilla: int = 42
    epocas: int = 0
    historial_perdida: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.reiniciar()

    def reiniciar(self):
        rng = np.random.default_rng(self.semilla)
        self.W1 = rng.normal(0.0, 0.4, size=(self.entrada_dim, self.oculta1_dim))
        self.b1 = np.zeros((1, self.oculta1_dim), dtype=np.float64)
        self.W2 = rng.normal(0.0, 0.4, size=(self.oculta1_dim, self.oculta2_dim))
        self.b2 = np.zeros((1, self.oculta2_dim), dtype=np.float64)
        self.W3 = rng.normal(0.0, 0.4, size=(self.oculta2_dim, self.salida_dim))
        self.b3 = np.zeros((1, self.salida_dim), dtype=np.float64)
        self.epocas = 0
        self.historial_perdida = []

    def adelante(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = _relu(z2)
        logits = h2 @ self.W3 + self.b3
        probs = _softmax(logits)
        return z1, h1, z2, h2, probs

    def probabilidades(self, obs: np.ndarray) -> np.ndarray:
        x_2d = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        _, _, _, _, probs = self.adelante(x_2d)
        return probs[0]

    def elegir_accion(self, obs: np.ndarray, solo_validas: bool = True) -> int:
        probs = self.probabilidades(obs)
        if solo_validas:
            libres_4d = obs[1:5]
            mascara = np.where(libres_4d > 0.5, probs, -1.0)
            if np.max(mascara) >= 0.0:
                return int(np.argmax(mascara))
        return int(np.argmax(probs))

    def entrenar_epoca(self, X: np.ndarray, y: np.ndarray) -> float:
        z1, h1, z2, h2, probs = self.adelante(X)
        n = len(X)
        perdida = float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))

        dz3 = probs.copy()
        dz3[np.arange(n), y] -= 1.0
        dz3 /= n

        dW3 = h2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        dh2 = dz3 @ self.W3.T
        dz2 = dh2 * (z2 > 0.0)
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        self.epocas += 1
        self.historial_perdida.append(perdida)
        return perdida

    def entrenar(self, X: np.ndarray, y: np.ndarray, epocas: int = 200):
        for _ in range(epocas):
            self.entrenar_epoca(X, y)
        return self.historial_perdida

    def exactitud(self, X: np.ndarray, y: np.ndarray) -> float:
        _, _, _, _, probs = self.adelante(X)
        return float(np.mean(np.argmax(probs, axis=1) == y))