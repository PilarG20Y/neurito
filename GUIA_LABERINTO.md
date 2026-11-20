# Neurito 3.0: aprende a encontrar comida en un laberinto

Neurito busca comida dentro de tres laberintos. Su cerebro debe elegir entre arriba, abajo, izquierda y derecha. La red se construye con NumPy, y Pygame dibuja el escenario, la mascota y una gráfica de la pérdida.

## Cómo aprende

1. BFS calcula rutas cortas y genera ejemplos etiquetados: mapa, posición, meta y movimiento recomendado.
2. Los ejemplos se dividen en entrenamiento (80 %) y prueba (20 %). Los pares posición-meta de prueba no aparecen en entrenamiento.
3. Cada entrada incluye un vector one-hot de tres mapas y dos vectores one-hot de 81 casillas: 165 características.
4. La arquitectura es 165 → 96 → 4. Usa ReLU en la capa oculta y Softmax en las cuatro salidas.
5. El entrenamiento minimiza entropía cruzada multiclase mediante retropropagación y descenso de gradiente con mini-lotes.
6. Durante el juego, el modelo decide sin consultar BFS. BFS es el experto que proporcionó las etiquetas.

Esto es aprendizaje supervisado por imitación, NO aprendizaje por refuerzo.

## Dos modos

- Red pura: toma la salida de mayor probabilidad. Puede chocar con paredes o quedarse en ciclos.
- Asistido: descarta movimientos contra paredes y penaliza casillas visitadas. Son reglas adicionales, no conocimientos aprendidos por la red.

La precisión de clasificación no equivale al porcentaje de laberintos completados.

## Estructura

- laberinto_mundo.py: mapas, movimiento, BFS y navegación asistida.
- laberinto_red.py: red neuronal y entrenamiento con NumPy.
- laberinto.py: juego Pygame, sprites y pérdida.
- test_laberinto.py: pruebas automáticas.
- GUIA_LABERINTO.md: esta guía.

Reutiliza assets/neutral_1.png, neutral_2.png y neutral_3.png. No modifica las versiones anteriores.

## Ejecutar

    uv sync
    uv run laberinto.py

## Controles

| Tecla | Acción |
|---|---|
| 1 | Entrenar una época |
| 2 | Entrenar cien épocas |
| 3 | Entrenamiento continuo |
| A | Alternar red pura y navegación asistida |
| P | Alternar navegación automática y paso a paso |
| Espacio | Avanzar un paso en modo manual |
| N | Cambiar mapa |
| G | Nuevas posiciones y meta |
| C | Reiniciar contadores |
| R | Reiniciar pesos y entrenamiento |
| Esc | Salir |

## Probar automáticamente

    uv run python -m unittest -v test_laberinto.py

## Experimentos

1. Observa Neurito sin entrenar. ¿Por qué se atasca?
Neurito sin entrenar se atasca porque no logra diferenciar la comida de un muro; al intentar acercarse al alimento, lo confunde con un obstáculo e interrumpe su movimiento para evitarlo.
2. Entrena 100, 200 y 400 épocas. Compara pérdida, precisión y movimientos.
Entre las 100 y 200 épocas, la pérdida disminuye considerablemente y la exactitud de prueba aumenta. Al duplicar el entrenamiento a 400 épocas, la red se optimiza aún más, logrando la pérdida mínima:

100 épocas: Neurito aprende patrones básicos.

200 épocas: Muestra capacidad de resolución al recolectar la comida.

400 épocas: Su eficiencia aumenta drásticamente en la recolección del alimento.
3. Compara red pura frente a navegación asistida. ¿Qué parte corresponde al modelo y cuál a las reglas?
El modelo: Corresponde a la red neuronal entrenada (por ejemplo, a 400 épocas), donde Neurito confía única y exclusivamente en su propio criterio aprendido, incrementando drásticamente su rendimiento de recolección de manera autónoma.

Las reglas: Corresponden al sistema de navegación asistida (o algoritmos de guía), los cuales intervienen externamente para corregir decisiones erróneas o guiar de forma lógica al agente cuando la red por sí sola flaquea.
4. Cambia las 96 neuronas ocultas por 16, 32 o 128.
Cada modelo se entrenó bajo las mismas condiciones estables de 200 épocas para contrastar su evolución y métricas. Los resultados demuestran que incrementar el número de neuronas ocultas se traduce en una disminución progresiva de la función de pérdida y un mejor acoplamiento a los ejemplos de entrenamiento (obtenidos mediante BFS), permitiendo una navegación más precisa.

Ejemplo con 16 neuronas en la época 200: Se registró una pérdida de 0.0992 y una exactitud del 88.7%. Al subir a 96 o 128 neuronas, la capacidad de representación de la red mejora notablemente.
5. Cambia la tasa de aprendizaje 0.45 y observa si converge.
Sí presenta un proceso de convergencia, pero de manera muy lenta e insuficiente para el límite de épocas establecido. Al ser una tasa de aprendizaje demasiado baja, el algoritmo de optimización da pasos muy pequeños en la dirección del gradiente. Como resultado, la red no alcanza a minimizar la función de costo ni a aprender con la eficiencia requerida dentro de las 200 épocas, lo que reduce la capacidad de acierto del agente frente a la configuración óptima.
6. Cambia de mapa. Los mapas forman parte del entrenamiento, pero las combinaciones de inicio y meta usadas para probar no se entrenaron.
Al cambiar el mapa, Neurito no se confunde; sabe hacia dónde moverse y logra llegar a la comida. Esto demuestra que no memorizó los mapas, sino que entendió la lógica y los patrones generales de cómo moverse en cualquier entorno.
7. Haz vibecoding para verificar como mejora o empeora Neurito al cambiar BFS, por DFS y A*
BFS (por defecto): Enseña los caminos más cortos y directos hacia la comida, sirviendo como una base de entrenamiento sólida y eficiente.

DFS (Depth-First Search): El desempeño de la red empeoraría drásticamente. Como DFS suele tomar caminos largos y dar muchos rodeos antes de encontrar la salida, le enseñaría "malos hábitos" eineficiencias a la red neuronal.

A (A-star):* El rendimiento se mantendría excelente o mejoraría aún más. Al buscar el camino más inteligente, óptimo y directo, permite que la red neuronal aprenda con mayor rapidez y mueva al axolote con absoluta precisión.
## Alcance y limitaciones

El modelo no interpreta capturas de pantalla: trabaja con características estructuradas. Conoce el identificador de cada mapa. Los tres mapas pertenecen al mismo universo de entrenamiento. La evaluación demuestra generalización a pares de posición-meta no entrenados, no a laberintos totalmente desconocidos. Para eso habría que representar también las paredes de cualquier mapa nuevo y entrenar y evaluar en mapas separados.
