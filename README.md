# Neurito — NeuroEcosistema
Maria del Pilar Guzman -
Ximena Patricia Garcia Espinel

Simulación en Python (`9x9`) de navegación con redes neuronales y ecosistema artificial con presas, depredadores y plantas regenerativas controlado por redes neuronales multicapa (`MLP` de dos capas ocultas por especie), interfaz visual en Pygame y experimentos reproducibles.

## Ejecución rápida

```powershell
# Punto de entrada / interfaz gráfica (1080x690)
uv run python main.py
## Orden recomendado de ejecución

### 1. Ejecutar las pruebas unitarias primero
```bash
uv run python -m unittest test_laberinto.py
¿Por qué ejecutar este comando primero?

Permite verificar que la lógica base del laberinto y los componentes clave del entorno funcionen correctamente antes de abrir la interfaz gráfica o lanzar experimentos largos. Así detectas a tiempo cualquier error de entorno o código y aseguras que las pruebas sean reproducibles.

2. Abrir la simulación / entorno del laberinto
uv run python laberinto.py
Una vez verificadas las pruebas unitarias, este comando ejecuta directamente el módulo del laberinto para observar la dinámica y el comportamiento de los agentes.

3. Otros comandos para probar el resto del proyecto
Si deseas explorar la interfaz principal, reproducir los experimentos por escenarios o generar las gráficas comparativas, puedes ejecutar los siguientes comandos:

# Experimentos reproducibles en tres escenarios y múltiples semillas
# Genera resultados_detalle.csv y resumen_experimentos.json en evidencia_eco/
uv run python experimentos_eco.py

# Generación de gráficas comparativas en evidencia_eco/
uv run python graficas_eco.py

# Pruebas unitarias reproducibles

## Escenarios y políticas evaluadas

En las corridas experimentales se evalúan políticas de referencia y modelos neuronales **MLP** a través de tres escenarios:

### Políticas
* **`aleatorio`**: Decisiones al azar como línea base mínima de comparación.
* **`reglas`**: Política heurística con condiciones fijas programadas a mano (como buscar alimento, huir o cazar), usada únicamente como referencia de comparación y no como reemplazo de la decisión salida de la red neuronal.
* **`sin_entrenar`**: Red con pesos iniciales sin fase de entrenamiento, útil como línea base para contrastar qué aporta el entrenamiento frente a una red sin entrenar (no es un resultado *"malo"* por sí mismo, sino un punto de referencia que en este experimento incluso muestra supervivencia de presas alta junto a menor consumo/capturas).
* **`mlp_16_12` y `mlp_24_16`**: Configuraciones de red neuronal MLP evaluadas en el ecosistema.

### Escenarios
* **`A_recursos_suficientes`**
* **`B_escasez_alimento`**
* **`C_alta_depredacion`**

---

## Cierre académico y análisis de resultados

### Qué funcionó
* Ejecutar pruebas unitarias (`test_laberinto.py`), experimentos reproducibles (`experimentos_eco.py`) y generación de gráficas (`graficas_eco.py`) permite respaldar el comportamiento del sistema con evidencia verificable en `evidencia_eco/` en lugar de depender únicamente de la visualización.
* Incluir baselines como `aleatorio`, `reglas` y `sin_entrenar` junto a `mlp_16_12` y `mlp_24_16` da contexto claro sobre qué aporta cada enfoque frente a una red sin entrenar o a reglas directas.
* Los modelos MLP (`mlp_16_12` y `mlp_24_16`) muestran respuestas diferenciadas frente a `sin_entrenar` en consumo de plantas y capturas de depredadores en escenarios como `B_escasez_alimento` y `C_alta_depredacion`.

### Qué no funcionó o presentó limitaciones
* La política de `reglas` obtuvo promedios más altos en plantas consumidas y capturas de depredadores que las configuraciones MLP evaluadas, mostrando que los modelos neuronales aún no superan consistentemente a una heurística directa en esas métricas.
* Se observó **alta dispersión (desviación estándar amplia)** en varias métricas y escenarios (por ejemplo, `mlp_16_12` en plantas consumidas en `A_recursos_suficientes`, `mlp_24_16` en capturas en `B_escasez_alimento`, y `reglas` en capturas en `C_alta_depredacion`), lo que refleja variabilidad entre semillas y casos de baja estabilidad o convergencia irregular.
* Un mayor consumo o captura (`reglas`) no se tradujo en mayor supervivencia de presas; de hecho, `sin_entrenar` registró promedios de supervivencia de presas más altos en los tres escenarios, evidenciando compromisos (*trade-offs*) y dinámicas ecológicas sensibles al equilibrio entre especies.

### Por qué pudieron producirse estos resultados
* Una política basada en `reglas` actúa de forma directa sobre condiciones inmediatas, mientras que las redes MLP dependen de la calidad y extensión del entrenamiento, la señal de recompensa y la inicialización/semillas para estabilizar su política.
* En entornos con escasez o alta depredación, las interacciones entre presas, depredadores y recursos amplifican pequeñas diferencias de conducta, generando fluctuaciones amplias entre repeticiones.

### Cómo se podría mejorar
* Incrementar el número de semillas y episodios de entrenamiento para evaluar con mayor rigor la convergencia y la varianza estadística.
* Refinar la configuración de entrenamiento, arquitecturas o funciones de recompensa de las especies y documentar explícitamente cómo se diferencian los modelos o entrenamientos de cada especie (así como verificar en el código que las decisiones de al menos una población salgan de su red neuronal).
* Añadir métricas adicionales de estabilidad poblacional (como trayectorias por paso de tiempo, extinciones y curvas de aprendizaje) para complementar los promedios finales por escenario.
uv run python -m unittest test_laberinto.py

