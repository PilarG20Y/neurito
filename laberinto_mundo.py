"""Mapas, búsqueda BFS experta, ecosistema neuronal y simulación independiente de Pygame."""
from collections import deque
from dataclasses import dataclass, field
import random
from typing import Any

# (fila, columna), orden fijo de desempate
DIRECCIONES = ((-1, 0), (1, 0), (0, -1), (0, 1))
NOMBRES = ('ARRIBA', 'ABAJO', 'IZQUIERDA', 'DERECHA')
MAPAS = (
    (
        '#########',
        '#...#...#',
        '#.#.#.#.#',
        '#.#...#.#',
        '#.###.#.#',
        '#...#...#',
        '###.#.###',
        '#.......#',
        '#########',
    ),
    (
        '#########',
        '#...#...#',
        '#.#...#.#',
        '#...#...#',
        '###.#.###',
        '#...#...#',
        '#.#...#.#',
        '#...#...#',
        '#########',
    ),
    (
        '#########',
        '#.......#',
        '#.###.#.#',
        '#...#.#.#',
        '###.#...#',
        '#...###.#',
        '#.#.....#',
        '#.......#',
        '#########',
    ),
)


def libres(mapa):
    return [(f, c) for f, fila in enumerate(mapa) for c, v in enumerate(fila) if v == '.']


def vecino(pos, accion):
    df, dc = DIRECCIONES[accion]
    return pos[0] + df, pos[1] + dc


def valido(mapa, pos):
    f, c = pos
    return 0 <= f < len(mapa) and 0 <= c < len(mapa[0]) and mapa[f][c] != '#'


def distancias_bfs(mapa, meta):
    """Distancia real por celdas hasta la comida, calculada SOLO por el profesor BFS."""
    dist = {meta: 0}
    cola = deque([meta])
    while cola:
        actual = cola.popleft()
        for a in range(4):
            siguiente = vecino(actual, a)
            if valido(mapa, siguiente) and siguiente not in dist:
                dist[siguiente] = dist[actual] + 1
                cola.append(siguiente)
    return dist


def accion_experta(mapa, origen, dist):
    """BFS indica el siguiente movimiento que reduce la distancia real a la meta."""
    if origen not in dist or dist[origen] == 0:
        return None
    opciones = [(dist.get(vecino(origen, a), 10**9), a) for a in range(4)]
    minimo, accion = min(opciones)
    assert minimo < dist[origen]
    return accion


@dataclass
class AgenteEco:
    pos: tuple[int, int]
    energia: float = 100.0
    vivo: bool = True


@dataclass
class Laberinto:
    indice: int = 0
    semilla: int = 99
    num_presas: int = 5
    num_depredadores: int = 2
    num_plantas: int = 8
    red_presa: Any = None
    red_depredador: Any = None

    def __post_init__(self):
        self.azar = random.Random(self.semilla)
        self.exitos = 0
        self.pasos = 0
        self.choques = 0
        self.capturas = 0
        self.plantas_comidas = 0
        self.plantas: set[tuple[int, int]] = set()
        self.presas: list[AgenteEco] = field(default_factory=list)
        self.depredadores: list[AgenteEco] = field(default_factory=list)
        self.reiniciar(self.indice)

    def reiniciar(self, indice=None):
        if indice is not None:
            self.indice = indice % len(MAPAS)
        self.mapa = MAPAS[self.indice]
        celdas = libres(self.mapa)
        self.meta = self.azar.choice(celdas)
        opciones = [p for p in celdas if p != self.meta and p in distancias_bfs(self.mapa, self.meta)]
        self.posicion = self.azar.choice(opciones)
        self.visitadas = {self.posicion: 1}
        self.pasos = 0
        self.intentos = 0
        self.terminado = False
        self.ultimo = None

        disponibles = [p for p in celdas if p != self.posicion]
        self.azar.shuffle(disponibles)

        self.plantas = set(disponibles[: min(self.num_plantas, len(disponibles))])
        resto = [p for p in disponibles if p not in self.plantas]

        self.presas = [
            AgenteEco(pos=resto[i % len(resto)], energia=100.0, vivo=True)
            for i in range(self.num_presas)
        ] if resto else []

        self.depredadores = [
            AgenteEco(pos=resto[(len(resto) - 1 - i) % len(resto)], energia=120.0, vivo=True)
            for i in range(self.num_depredadores)
        ] if resto else []

    def _reponer_plantas_si_faltan(self):
        if len(self.plantas) < max(3, self.num_plantas // 2):
            celdas = libres(self.mapa)
            ocupadas = set(self.plantas) | {self.posicion, self.meta}
            ocupadas |= {p.pos for p in self.presas if p.vivo}
            ocupadas |= {d.pos for d in self.depredadores if d.vivo}
            libres_eco = [c for c in celdas if c not in ocupadas]
            self.azar.shuffle(libres_eco)
            faltan = max(0, self.num_plantas - len(self.plantas))
            self.plantas.update(libres_eco[:faltan])

    def _reponer_seres_si_caen(self):
        celdas = libres(self.mapa)
        if not celdas:
            return
        if self.presas and all(not p.vivo for p in self.presas):
            for p in self.presas:
                p.pos = self.azar.choice(celdas)
                p.energia = 100.0
                p.vivo = True
        if self.depredadores and all(not d.vivo for d in self.depredadores):
            for d in self.depredadores:
                d.pos = self.azar.choice(celdas)
                d.energia = 120.0
                d.vivo = True

    def _elegir_paso_presa(self, presa: AgenteEco) -> int:
        if self.red_presa is not None:
            from ecosistema_red import observar_presa
            obs = observar_presa(self, presa)
            return self.red_presa.elegir_accion(obs, solo_validas=True)

        vecinos_validos = [
            (a, vecino(presa.pos, a))
            for a in range(4)
            if valido(self.mapa, vecino(presa.pos, a))
        ]
        if not vecinos_validos:
            return 0
        for a, pos_v in vecinos_validos:
            if pos_v in self.plantas:
                return a
        pos_deps = {d.pos for d in self.depredadores if d.vivo}
        seguros = [(a, pv) for a, pv in vecinos_validos if pv not in pos_deps]
        candidatos = seguros if seguros else vecinos_validos
        return self.azar.choice([a for a, _ in candidatos])

    def _elegir_paso_depredador(self, dep: AgenteEco) -> int:
        if self.red_depredador is not None:
            from ecosistema_red import observar_depredador
            obs = observar_depredador(self, dep)
            return self.red_depredador.elegir_accion(obs, solo_validas=True)

        vecinos_validos = [
            (a, vecino(dep.pos, a))
            for a in range(4)
            if valido(self.mapa, vecino(dep.pos, a))
        ]
        if not vecinos_validos:
            return 0
        presas_vivas = [p.pos for p in self.presas if p.vivo]
        if not presas_vivas:
            return self.azar.choice([a for a, _ in vecinos_validos])
        mejor_a, _ = min(
            vecinos_validos,
            key=lambda item: min(
                abs(item[1][0] - px) + abs(item[1][1] - py) for px, py in presas_vivas
            ),
        )
        return mejor_a

    def mover(self, accion):
        self.ultimo = accion
        self.intentos += 1
        nuevo = vecino(self.posicion, accion)
        if not valido(self.mapa, nuevo):
            self.choques += 1
            return 'pared'
        self.posicion = nuevo
        self.pasos += 1
        self.visitadas[nuevo] = self.visitadas.get(nuevo, 0) + 1
        if nuevo == self.meta:
            self.exitos += 1
            self.terminado = True
            return 'comida'
        return 'avanza'

    def paso_ecosistema(self, accion_neurito=None):
        """Avanza un turno completo de Neurito, plantas, presas y depredadores."""
        self._reponer_seres_si_caen()
        self._reponer_plantas_si_faltan()

        resultado_neurito = None
        if accion_neurito is not None:
            resultado_neurito = self.mover(accion_neurito)
            if resultado_neurito == 'comida':
                celdas = libres(self.mapa)
                candidatas = [c for c in celdas if c != self.posicion]
                if candidatas:
                    self.meta = self.azar.choice(candidatas)
                self.terminado = False

        for presa in self.presas:
            if not presa.vivo:
                continue
            a_p = self._elegir_paso_presa(presa)
            dest_p = vecino(presa.pos, a_p)
            if valido(self.mapa, dest_p):
                presa.pos = dest_p
            presa.energia -= 2.0
            if presa.pos in self.plantas:
                self.plantas.remove(presa.pos)
                self.plantas_comidas += 1
                presa.energia = min(100.0, presa.energia + 35.0)
            if presa.energia <= 0:
                presa.vivo = False

        for dep in self.depredadores:
            if not dep.vivo:
                continue
            a_d = self._elegir_paso_depredador(dep)
            dest_d = vecino(dep.pos, a_d)
            if valido(self.mapa, dest_d):
                dep.pos = dest_d
            dep.energia -= 2.5

            for presa in self.presas:
                if presa.vivo and presa.pos == dep.pos:
                    presa.vivo = False
                    self.capturas += 1
                    dep.energia = min(120.0, dep.energia + 45.0)

            if dep.energia <= 0:
                dep.vivo = False

        self._reponer_plantas_si_faltan()
        return resultado_neurito


def elegir_accion(red, mundo, visitas, asistido):
    """Argmax puro; asistencia opcional para paredes y visitas repetidas."""
    probs = red.probabilidades(mundo.indice, mundo.posicion, mundo.meta)
    if not asistido:
        return int(probs.argmax()), probs
    opciones = [
        (float(probs[a]) - 0.35 * visitas.get(vecino(mundo.posicion, a), 0), -a, a)
        for a in range(4) if valido(mundo.mapa, vecino(mundo.posicion, a))
    ]
    return max(opciones)[2], probs