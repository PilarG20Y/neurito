# Neurito — NeuroEcosistema
Maria del Pilar Guzman -
Ximena Patricia Garcia Espinel

Simulación en Python (`9x9`) de navegación con redes neuronales y ecosistema artificial con presas, depredadores y plantas regenerativas controlado por redes neuronales multicapa (`MLP` de dos capas ocultas por especie), interfaz visual en Pygame y experimentos reproducibles.

## Ejecución rápida

```powershell
# Punto de entrada / interfaz gráfica (1080x690)
uv run python main.py
# o directamente:
uv run python laberinto.py

# Experimentos reproducibles en tres escenarios y múltiples semillas
# Genera resultados_detalle.csv y resumen_experimentos.json en evidencia_eco/
uv run python experimentos_eco.py

# Generación de gráficas comparativas en evidencia_eco/
uv run python graficas_eco.py

# Pruebas unitarias reproducibles
uv run python -m unittest test_laberinto.py

