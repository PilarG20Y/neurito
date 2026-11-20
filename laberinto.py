"""Neurito 3.0: laberinto supervisado con visualización de entrenamiento y ecosistema."""
from pathlib import Path
import numpy as np
import pygame

from ecosistema_red import RedEspecieMLP, generar_datos_especie
from laberinto_mundo import Laberinto, MAPAS, NOMBRES, elegir_accion
from laberinto_red import RedLaberinto, generar_datos

ANCHO, ALTO = 1080, 690
TAM = 55
ORIGEN_X, ORIGEN_Y = 30, 106
FONDO = (250, 246, 252)
TEXTO = (58, 48, 76)

PLANTA_COLOR = (56, 178, 108)
PRESA_COLOR = (45, 132, 212)
DEPREDADOR_COLOR = (214, 68, 80)


def texto(superficie, fuente, contenido, x, y, color=TEXTO):
    superficie.blit(fuente.render(str(contenido), True, color), (x, y))


def barras_probabilidad(pantalla, fuente, probs, x, y):
    for k, nombre in enumerate(NOMBRES):
        yy = y + k * 42
        texto(pantalla, fuente, f'{nombre}: {probs[k]:.1%}', x, yy)
        pygame.draw.rect(pantalla, (226, 216, 240), (x + 175, yy + 4, 150, 14), border_radius=6)
        pygame.draw.rect(pantalla, (150, 104, 202), (x + 175, yy + 4, int(150 * probs[k]), 14), border_radius=6)


def grafica(pantalla, historial, area):
    pygame.draw.rect(pantalla, (239, 232, 245), area, border_radius=11)
    if len(historial) < 2:
        return
    salto = max(1, len(historial) // 180)
    datos = historial[::salto]
    ymax = max(0.2, max(datos))
    puntos = [
        (int(area.left + 10 + i * (area.width - 20) / (len(datos) - 1)),
         int(area.bottom - 10 - v / ymax * (area.height - 20)))
        for i, v in enumerate(datos)
    ]
    pygame.draw.lines(pantalla, (139, 96, 199), False, puntos, 3)


def cargar_mascotas():
    carpeta = Path(__file__).resolve().parent / 'assets'
    nombres = ('neutral_1.png', 'neutral_2.png', 'neutral_3.png')
    return [pygame.transform.smoothscale(
        pygame.image.load(str(carpeta / nombre)).convert_alpha(), (49, 49)
    ) for nombre in nombres]


def _centro_celda(pos):
    f, c = pos
    return (ORIGEN_X + c * TAM + TAM // 2, ORIGEN_Y + f * TAM + TAM // 2)


def _construir_redes_eco(semilla: int = 42, epocas: int = 200):
    (Xp_tr, yp_tr), _ = generar_datos_especie('presa', n_muestras=1000, semilla=semilla)
    (Xd_tr, yd_tr), _ = generar_datos_especie('depredador', n_muestras=1000, semilla=semilla + 1)

    red_p_sin = RedEspecieMLP(11, oculta1_dim=16, oculta2_dim=12, lr=0.08, semilla=semilla)
    red_d_sin = RedEspecieMLP(7, oculta1_dim=16, oculta2_dim=12, lr=0.08, semilla=semilla + 10)

    red_p_ent = RedEspecieMLP(11, oculta1_dim=16, oculta2_dim=12, lr=0.08, semilla=semilla)
    red_d_ent = RedEspecieMLP(7, oculta1_dim=16, oculta2_dim=12, lr=0.08, semilla=semilla + 10)

    red_p_ent.entrenar(Xp_tr, yp_tr, epocas=epocas)
    red_d_ent.entrenar(Xd_tr, yd_tr, epocas=epocas)

    return {
        'entrenada': (red_p_ent, red_d_ent),
        'sin_entrenar': (red_p_sin, red_d_sin),
        'reglas': (None, None),
    }


def ejecutar():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption('Neurito 3.0 - Aprende a encontrar comida')
    reloj = pygame.time.Clock()
    grande = pygame.font.SysFont(None, 37)
    normal = pygame.font.SysFont(None, 26)
    pequeno = pygame.font.SysFont(None, 22)
    sprites = cargar_mascotas()
    red = RedLaberinto()
    (X, y), (Xt, yt), _ = generar_datos()

    redes_eco = _construir_redes_eco(semilla=42, epocas=200)
    modos_eco = ('entrenada', 'sin_entrenar', 'reglas')
    idx_modo_eco = 0
    rp, rd = redes_eco[modos_eco[idx_modo_eco]]

    mundo = Laberinto(red_presa=rp, red_depredador=rd)
    asistido = False
    automatico = True
    continuo = False
    ecosistema = True
    fracasos = 0
    ultimo_movimiento = '---'
    ultima_accion = 0
    ultimo_tick = pygame.time.get_ticks()
    activo = True

    while activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                activo = False
            elif evento.type == pygame.KEYDOWN:
                tecla = evento.key
                if tecla == pygame.K_ESCAPE:
                    activo = False
                elif tecla == pygame.K_1:
                    red.entrenar_epoca(X, y)
                elif tecla == pygame.K_2:
                    for _ in range(100):
                        red.entrenar_epoca(X, y)
                elif tecla == pygame.K_3:
                    continuo = not continuo
                elif tecla == pygame.K_e:
                    ecosistema = not ecosistema
                elif tecla == pygame.K_m:
                    idx_modo_eco = (idx_modo_eco + 1) % len(modos_eco)
                    mundo.red_presa, mundo.red_depredador = redes_eco[modos_eco[idx_modo_eco]]
                elif tecla == pygame.K_a:
                    asistido = not asistido
                    mundo.reiniciar()
                elif tecla == pygame.K_p:
                    automatico = not automatico
                elif tecla == pygame.K_n:
                    mundo.reiniciar(mundo.indice + 1)
                elif tecla == pygame.K_g:
                    mundo.reiniciar()
                elif tecla == pygame.K_c:
                    mundo.exitos = 0
                    mundo.choques = 0
                    mundo.capturas = 0
                    mundo.plantas_comidas = 0
                    fracasos = 0
                    mundo.reiniciar()
                elif tecla == pygame.K_r:
                    red.reiniciar()
                    continuo = False
                    mundo.reiniciar()
                    mundo.exitos = 0
                    mundo.choques = 0
                    mundo.capturas = 0
                    mundo.plantas_comidas = 0
                    fracasos = 0
                elif tecla == pygame.K_SPACE and not automatico:
                    ultimo_tick = 0
        if continuo:
            red.entrenar_epoca(X, y)

        ahora = pygame.time.get_ticks()
        dar_paso = (automatico and ahora - ultimo_tick >= 210) or (not automatico and ultimo_tick == 0)
        if dar_paso:
            ultimo_tick = ahora
            accion, _ = elegir_accion(red, mundo, mundo.visitadas, asistido)
            ultima_accion = accion
            if ecosistema:
                resultado = mundo.paso_ecosistema(accion)
            else:
                resultado = mundo.mover(accion)
            ultimo_movimiento = resultado or 'eco'
            if resultado == 'comida':
                mundo.reiniciar()
            elif mundo.intentos >= 45:
                fracasos += 1
                mundo.reiniciar()

        pantalla.fill(FONDO)
        texto(pantalla, grande, 'NEURITO 3.0 - LABERINTO INTELIGENTE', 30, 25)
        texto(
            pantalla,
            pequeno,
            f'Mapa {mundo.indice+1}/{len(MAPAS)}  |  Meta: amarillo  Planta: verde  Presa: azul  Dep: rojo',
            30,
            70,
        )
        for f, fila in enumerate(mundo.mapa):
            for c, celda in enumerate(fila):
                rect = pygame.Rect(ORIGEN_X + c * TAM, ORIGEN_Y + f * TAM, TAM-2, TAM-2)
                pygame.draw.rect(pantalla, (123, 94, 153) if celda == '#' else (234, 243, 245), rect, border_radius=6)

        if ecosistema:
            for p_pos in getattr(mundo, 'plantas', ()):
                pygame.draw.circle(pantalla, PLANTA_COLOR, _centro_celda(p_pos), 9)
            for presa in getattr(mundo, 'presas', ()):
                if presa.vivo:
                    pygame.draw.circle(pantalla, PRESA_COLOR, _centro_celda(presa.pos), 12)
            for dep in getattr(mundo, 'depredadores', ()):
                if dep.vivo:
                    cx, cy = _centro_celda(dep.pos)
                    pygame.draw.polygon(pantalla, DEPREDADOR_COLOR, [(cx, cy - 13), (cx - 12, cy + 11), (cx + 12, cy + 11)])

        cf, cc = mundo.meta
        centro = (ORIGEN_X + cc * TAM + TAM // 2, ORIGEN_Y + cf * TAM + TAM // 2)
        pygame.draw.circle(pantalla, (246, 191, 64), centro, 16)
        pygame.draw.circle(pantalla, (253, 236, 167), centro, 7)

        pf, pc = mundo.posicion
        sprite = sprites[(ahora // 500) % len(sprites)]
        pantalla.blit(sprite, (ORIGEN_X + pc * TAM + 2, ORIGEN_Y + pf * TAM + 2))

        presas_vivas = sum(1 for p in getattr(mundo, 'presas', ()) if p.vivo)
        deps_vivos = sum(1 for d in getattr(mundo, 'depredadores', ()) if d.vivo)
        energ_p = float(np.mean([p.energia for p in mundo.presas])) if mundo.presas else 0.0
        modo_eco_actual = modos_eco[idx_modo_eco]

        x = 565
        texto(pantalla, normal, f'Épocas: {red.epocas}', x, 89)
        texto(pantalla, normal, f'Pérdida: {red.historial[-1]:.4f}' if red.historial else 'Pérdida: sin entrenar', x, 121)
        texto(pantalla, normal, f'Exactitud prueba: {red.exactitud(Xt, yt):.1%}', x, 153)
        texto(pantalla, normal, f'Modo: {"ASISTIDO" if asistido else "RED PURA"} | Eco: {"ON" if ecosistema else "OFF"} ({modo_eco_actual})', x, 185)
        texto(pantalla, normal, f'Navegación: {"AUTOMÁTICA" if automatico else "PASO A PASO"}', x, 217)
        texto(pantalla, normal, f'Comidas: {mundo.exitos} | Presas: {presas_vivas} | Dep: {deps_vivos} | Caza: {getattr(mundo, "capturas", 0)}', x, 249)
        texto(pantalla, pequeno, f'Plantas com: {mundo.plantas_comidas} | Energ. presa: {energ_p:.1f} | Último: {NOMBRES[ultima_accion]} ({ultimo_movimiento})', x, 281)
        texto(pantalla, normal, 'SALIDAS SOFTMAX DE LA RED', x, 320)
        _, probs = elegir_accion(red, mundo, mundo.visitadas, asistido)
        barras_probabilidad(pantalla, pequeno, probs, x, 358)
        texto(pantalla, pequeno, 'EVOLUCIÓN DE LA PÉRDIDA', x, 540)
        grafica(pantalla, red.historial, pygame.Rect(x, 565, 460, 73))
        texto(pantalla, pequeno, '1: +1 época   2: +100 épocas   3: continuo   E: eco   M: modo eco', 32, 618)
        texto(pantalla, pequeno, 'A: asistencia   P: automático/pasos   ESPACIO: avanzar', 32, 640)
        texto(pantalla, pequeno, 'N: otro mapa   G: nuevo reto   C: contadores   R: cerebro   ESC: salir', 32, 662)
        pygame.display.flip()
        reloj.tick(30)
    pygame.quit()


if __name__ == '__main__':
    ejecutar()