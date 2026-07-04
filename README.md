# Práctica Calificada 5 - CC0C2 Procesamiento del Lenguaje Natural

## Título del proyecto
**Proyecto 8: Despliegue y optimización de inferencia**

---

## Objetivo
El objetivo principal de este proyecto es cargar un modelo de lenguaje en un entorno reproducible, medir con precisión la latencia de inferencia y el throughput ante diferentes tamaños de prompt de entrada, implementar y auditar estrategias de optimización de memoria (cuantización en FP16 e INT8 dinámico), y justificar técnicamente el trade-off existente entre la velocidad computacional, el consumo de memoria del sistema y la calidad sintáctica/semántica del texto generado.

---

## Resumen de la línea base
En la línea base, se carga el modelo causal preentrenado (`gpt2`) en precisión de punto flotante por defecto de 32 bits (`torch.float32`). Se registramos la cantidad exacta de parámetros, la huella en memoria VRAM/RAM y se mide experimentalmente el tiempo de generación (latencia promedio en milisegundos) y la velocidad de generación (throughput en tokens/segundo) ante prompts de longitudes acotadas sin aplicar ningún tipo de técnica de compresión.

---

## Modificación realizada (Estrategias de Optimización)
Para la modificación obligatoria del Proyecto 8, se comparan dos estrategias rigurosas de optimización computacional:
1. **Estrategia 1: Cuantización a Precisión Media (FP16)**: Reducción de los tensores de pesos a formato de coma flotante de 16 bits (`torch.float16`).
2. **Estrategia 2: Cuantización Dinámica a Enteros (INT8)**: Aplicación de compresión lineal en tiempo de ejecución al mapear las capas `nn.Linear` del modelo a enteros de 8 bits (`torch.qint8`) mediante PyTorch nativo (`torch.quantization.quantize_dynamic`), lo cual permite lograr hasta ~75% de reducción en la memoria de los pesos lineales siendo ideal para entornos Edge o de hardware acotado.

Además, se midió de forma empírica y comparativa cómo estas optimizaciones responden al variar sistemáticamente el **tamaño de entrada** (prompts corto, mediano y largo).

---

## Cómo ejecutar el notebook


1. **Construir la imagen de Docker**:
   ```bash
   docker build -t nlp-pc5 .
   ```
2. **Ejecutar el contenedor**:
   ```bash
   docker run -it --rm -p 8891:8891 -v "${PWD}:/workspace" nlp-pc5
   ```
3. **Acceder a Jupyter Lab**:
   Abre un navegador web en `http://localhost:8891` y ejecuta todas las celdas del archivo `PC5_Proyecto8.ipynb`.

---

## Principales resultados
- **Reducción de Huella de Memoria**: La cuantización demostró una relación matemática directa con la representación en bytes (`element_size`). El modelo en FP16 redujo el consumo de RAM/VRAM de los pesos en un 50% exacto respecto a FP32, mientras que la cuantización dinámica en INT8 redujo la memoria en cerca del ~75%.
- **Mejora en Latencia y Throughput**: Al reducir el ancho de palabra computacional, disminuye el cuello de botella en la transferencia del bus de memoria (*memory-bound*), permitiendo al modelo procesar más tokens por segundo (throughput superior) y acelerar el tiempo de respuesta total (latencia reducida).
- **Trade-off Técnico**: La reducción a FP16 conservó intacta la calidad coherente del texto generado en comparación con FP32. En INT8, se observan variaciones menores de redondeo en secuencias largas, lo que representa el trade-off esperado entre compresión extrema y fidelidad numérica.

---

## Defensa Técnica: Preguntas y Respuestas Clave

En el notebook `PC5_Proyecto8.ipynb` (Sección 8) se incluye una defensa técnica completa e integral con respuestas detalladas a **las 5 preguntas específicas del Proyecto 8** y a las **preguntas transversales** obligatorias. A continuación se resumen los conceptos evaluados:

1. **¿Por qué la cuantización reduce memoria pero puede introducir error de redondeo en activaciones?**  
   Al reducir los bits de representación (ej. FP32 a INT8), el espacio continuo de flotantes se mapea a un conjunto discreto de 256 valores, generando errores de truncamiento o aproximación numérica. En los Transformers, estos errores se acumulan en las activaciones a lo largo de las capas, distorsionando ligeramente los logits finales.
2. **Diferencia entre latencia de primer token (TTFT) y latencia por token (TPOT)**:  
   El *TTFT* corresponde al tiempo de procesamiento y contextualización inicial de todo el prompt (*prefill phase*, dominado por cómputo). El *TPOT* es el tiempo por cada token sucesivo generado en el bucle autorregresivo (*decoding step*, dominado por el ancho de banda de la memoria).
3. **¿Por qué el batching dinámico mejora throughput pero puede aumentar latencia de cola (tail latency)?**  
   Al agrupar peticiones asíncronas en vuelo, el uso de recursos de la GPU se maximiza incrementando los tokens por segundo globales (*throughput*). Sin embargo, esperar a completar o sincronizar lotes con peticiones heterogéneas introduce esperas en la cola, elevando los tiempos del percentil peor o latencia de cola (*p95 / p99*).
4. **Relación entre tamaño de modelo y costo de producción**:  
   El consumo de VRAM de los pesos inactivos, junto a los buffers de *KV Cache* y activaciones, crece linealmente con la cantidad de parámetros. Modelos más grandes exigen hardware de grado servidor (múltiples GPUs A100/H100) elevando el costo operativo e infraestructura en el cloud o edge.
5. **Métricas críticas en monitorización industrial (MLOps)**:  
   Métricas de rendimiento (TTFT, TPOT en p50/p95/p99, throughput de tokens/s, saturación de VRAM/OOM) y métricas de deriva o calidad del modelo (tasa de error en generación, longitud de respuestas, entropía de predicción y feedback del usuario).
6. **Percepción vs Generación vs Despliegue en el Notebook**:  
   La percepción es el encoding computacional del texto por el tokenizer y las capas de atención del Transformer; la generación es el ciclo de predicción autorregresiva de tokens (`generate`); y el despliegue es toda la ingeniería computacional de entornos, temporización milimétrica, huella de memoria y estrategias de cuantización FP16/INT8.

---

## Limitaciones
- **Sensibilidad al Redondeo en Tareas Complejas**: Si bien en prompts de lenguaje natural generales el impacto de INT8 es bajo, en tareas de razonamiento lógico profundo o matemáticas la acumulación del error numérico por cuantización puede generar desviaciones perceptibles.
- **Dependencia de Hardware en Inferencia**: La aceleración real en milisegundos observada depende directamente de si el hardware subyacente (CPU/GPU) posee instrucciones nativas de vectorización optimizadas para tipos de datos reducidos (como FP16 o INT8/AVX512).

---

## Qué se muestra en el video (Guía para Grabación y Defensa)
El video de defensa técnica asíncrona de la Práctica Calificada 5 (con una duración estrictamente mayor a 12 minutos) está programado para demostrar los siguientes elementos:
1. **Presentación del Proyecto y Problema Técnico**: Identificación del Proyecto 8 y el desafío fundamental de optimizar modelos de lenguaje para entornos de recursos acotados.
2. **Línea Base en Vivo**: Ejecución y auditoría de las celdas del modelo `gpt2` en FP32, interpretando la huella en memoria y el throughput.
3. **Ejercicio A (Modificación Común Obligatoria)**: Modificación en vivo del tamaño del prompt de entrada en pantalla, prediciendo teóricamente antes de ejecutar cómo cambiará la latencia del tiempo al primer token (*TTFT*) y validando la hipótesis al observar la salida.
4. **Ejercicio B (Modificación Específica del Proyecto)**: Codificación e implementación en vivo del paso de FP32 hacia FP16 y cuantización dinámica INT8 (`torch.quantization.quantize_dynamic`), explicando el cambio de tipo de dato (`dtype` y `element_size`), prediciendo la reducción exacta de memoria de los tensores e interpretando el resultado computacional posterior a la celda.
5. **Explicación de Estructuras y Respuestas Avanzadas**: Sustentación verbal con el notebook en pantalla de las respuestas a las preguntas del Proyecto 8 (diferencias TTFT/TPOT, batching dinámico, costos en producción y monitorización MLOps).
6. **Cierre Técnico y Puente al Curso**: Conclusiones del trade-off calidad-velocidad y vinculación de este despliegue con las arquitecturas multimodales, cuantización en Edge computing y seguridad del curso CC0C2.

---

## Declaración de autoría y uso de IA

```
Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta práctica.
Si utilicé herramientas de IA, las use como apoyo para redacción, depuración o consulta, pero la implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mia.
```

