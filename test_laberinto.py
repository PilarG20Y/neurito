"""Pruebas reproducibles sin interfaz; solo stdlib y NumPy."""
import unittest
import numpy as np

from ecosistema_red import (
    RedEspecieMLP,
    generar_datos_especie,
    observar_depredador,
    observar_presa,
)
from laberinto_mundo import (
    MAPAS,
    Laberinto,
    accion_experta,
    distancias_bfs,
    elegir_accion,
    libres,
    valido,
    vecino,
)
from laberinto_red import RedLaberinto, generar_datos


class PruebaLaberinto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ent, cls.prueba, cls.casos = generar_datos(semilla=42)
        cls.red = RedLaberinto()
        for _ in range(400):
            cls.red.entrenar_epoca(*cls.ent)

    def test_mapas_conectados_y_bfs_correcto(self):
        for mapa in MAPAS:
            celdas = libres(mapa)
            self.assertGreaterEqual(len(celdas), 25)
            for meta in celdas:
                dist = distancias_bfs(mapa, meta)
                self.assertEqual(len(dist), len(celdas))
                for p in celdas:
                    if p == meta:
                        continue
                    accion = accion_experta(mapa, p, dist)
                    self.assertTrue(valido(mapa, vecino(p, accion)))
                    self.assertEqual(dist[vecino(p, accion)], dist[p] - 1)

    def test_datos_separados_y_red_aprende(self):
        X, y = self.ent
        Xt, yt = self.prueba
        self.assertEqual(X.shape[1], 165)
        self.assertEqual(len(set(map(tuple, X)) & set(map(tuple, Xt))), 0)
        self.assertEqual(len(self.red.historial), 400)
        self.assertLess(self.red.historial[-1], self.red.historial[0])
        self.assertGreaterEqual(self.red.exactitud(Xt, yt), 0.90)

    def test_probabilidades_validas(self):
        X, _ = self.prueba
        p = self.red.adelante(X[:20])[2]
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_asistencia_no_cruza_paredes(self):
        for indice, inicio, meta, _ in self.casos[:80]:
            mundo = Laberinto(indice)
            mundo.posicion, mundo.meta = inicio, meta
            accion, _ = elegir_accion(self.red, mundo, {inicio: 1}, True)
            self.assertTrue(valido(mundo.mapa, vecino(inicio, accion)))

    def test_exito_en_retos_no_entrenados(self):
        exitos = 0
        sin_asistir = 0
        for indice, inicio, meta, _ in self.casos:
            mapa = MAPAS[indice]
            distancia_optima = distancias_bfs(mapa, meta)[inicio]
            limite = min(65, 2 * distancia_optima + 8)
            for asistido in (False, True):
                mundo = Laberinto(indice)
                mundo.posicion, mundo.meta = inicio, meta
                visitas = {inicio: 1}
                for _ in range(limite):
                    accion, _ = elegir_accion(self.red, mundo, visitas, asistido)
                    siguiente = vecino(mundo.posicion, accion)
                    if not valido(mapa, siguiente):
                        break
                    mundo.posicion = siguiente
                    visitas[siguiente] = visitas.get(siguiente, 0) + 1
                    if siguiente == meta:
                        if asistido:
                            exitos += 1
                        else:
                            sin_asistir += 1
                        break
        self.assertGreaterEqual(sin_asistir / len(self.casos), 0.75)
        self.assertGreaterEqual(exitos / len(self.casos), 0.98)

    def test_ecosistema_en_laberinto(self):
        mundo = Laberinto(indice=0, semilla=99, num_presas=4, num_depredadores=2, num_plantas=6)
        self.assertTrue(hasattr(mundo, "paso_ecosistema"))
        for paso in range(25):
            mundo.paso_ecosistema(accion_neurito=paso % 4)
            self.assertTrue(valido(mundo.mapa, mundo.posicion))
            for p in mundo.presas:
                self.assertTrue(valido(mundo.mapa, p.pos))
            for d in mundo.depredadores:
                self.assertTrue(valido(mundo.mapa, d.pos))

    def test_redes_especie_y_entrenamiento_eco(self):
        (Xp_tr, yp_tr), (Xp_te, yp_te) = generar_datos_especie("presa", n_muestras=250, semilla=11)
        (Xd_tr, yd_tr), (Xd_te, yd_te) = generar_datos_especie("depredador", n_muestras=250, semilla=12)

        red_p = RedEspecieMLP(entrada_dim=11, oculta1_dim=16, oculta2_dim=12, lr=0.08, semilla=11)
        red_d = RedEspecieMLP(entrada_dim=7, oculta1_dim=16, oculta2_dim=12, lr=0.08, semilla=12)

        p_ini = red_p.entrenar_epoca(Xp_tr, yp_tr)
        red_p.entrenar(Xp_tr, yp_tr, epocas=60)
        self.assertLess(red_p.historial_perdida[-1], p_ini)
        self.assertGreater(red_p.exactitud(Xp_te, yp_te), 0.50)

        d_ini = red_d.entrenar_epoca(Xd_tr, yd_tr)
        red_d.entrenar(Xd_tr, yd_tr, epocas=60)
        self.assertLess(red_d.historial_perdida[-1], d_ini)
        self.assertGreater(red_d.exactitud(Xd_te, yd_te), 0.50)

        mundo = Laberinto(
            indice=1,
            semilla=21,
            num_presas=4,
            num_depredadores=2,
            num_plantas=8,
            red_presa=red_p,
            red_depredador=red_d,
        )
        self.assertEqual(observar_presa(mundo, mundo.presas[0]).shape, (11,))
        self.assertEqual(observar_depredador(mundo, mundo.depredadores[0]).shape, (7,))
        for _ in range(15):
            mundo.paso_ecosistema(accion_neurito=None)


if __name__ == '__main__':
    unittest.main()