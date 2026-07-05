# Práctica Calificada 5 - CC0C2 Procesamiento del Lenguaje Natural

## Título del proyecto
**Proyecto 8: Despliegue y optimización de inferencia**

---

## Línea de proyecto elegida
Proyecto 8: RLHF, serving, monitorización, latencia, privacidad, seguridad y edge vs cloud.

---

## Objetivo
Cargar un modelo de lenguaje en un entorno reproducible, medir con precisión la latencia de inferencia y el throughput ante diferentes tamaños de prompt de entrada, implementar y auditar estrategias de optimización de memoria (cuantización en FP16 e INT8 dinámico), y justificar técnicamente el trade-off existente entre la velocidad computacional, el consumo de memoria del sistema y la calidad del texto generado.

---

## Resumen de la línea base
En la línea base, se carga el modelo causal preentrenado (`gpt2`) en precisión de punto flotante de 32 bits (`torch.float32`). El modelo cuenta con 124,439,808 parámetros y ocupa 474.70 MB de memoria. Se mide experimentalmente la latencia promedio (1178.43 ms) y el throughput (16.97 tokens/s) ante un prompt mediano, sin aplicar ninguna técnica de compresión.

---

## Modificación realizada (Estrategias de Optimización)
Se comparan dos estrategias de optimización computacional contra la línea base:

1. **Estrategia 1: Cuantización a Precisión Media (FP16)**: Reducción de los tensores de pesos a formato de coma flotante de 16 bits (`torch.float16`), reduciendo la memoria a la mitad.
2. **Estrategia 2: Cuantización Dinámica a Enteros (INT8)**: GPT-2 de HuggingFace usa `Conv1D` en vez de `nn.Linear`. Como `torch.quantization.quantize_dynamic` solo reconoce `nn.Linear`, primero convertimos las capas `Conv1D` → `nn.Linear` y luego aplicamos cuantización dinámica a enteros de 8 bits (`torch.qint8`).

Además, se midió cómo estas optimizaciones responden al variar el tamaño de entrada (prompts corto, mediano y largo).

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

- **Reducción de Huella de Memoria**: FP16 redujo la memoria de 474.70 MB a 237.35 MB (50% exacto). INT8 dinámico (tras conversión Conv1D → nn.Linear) redujo la memoria a 150.38 MB (68% de reducción).

- **Latencia y dependencia del hardware**: FP16 en CPU resultó **más lento** que FP32 (8102.86 ms vs 1178.43 ms), porque las CPUs convencionales carecen de unidades de cómputo FP16 nativas y cada operación requiere casting interno a FP32. INT8 en CPU mostró latencia de 762.43 ms, siendo la estrategia más favorable para este entorno.

- **Trade-off Técnico**: FP16 reduce memoria a la mitad pero penaliza severamente la latencia en CPU; su beneficio se materializa en GPU con Tensor Cores. INT8 reduce memoria y mantiene o mejora la latencia en CPU. La calidad del texto generado se conserva en ambos casos para prompts de lenguaje general, pero en tareas de razonamiento complejo la cuantización puede introducir desviaciones por error acumulado.

---

## Limitaciones
- **FP16 en CPU**: La ejecución en CPU sin soporte nativo FP16 genera overhead de casting, produciendo latencias superiores a FP32. El beneficio de FP16 requiere GPU con Tensor Cores.
- **Conv1D vs nn.Linear**: La cuantización dinámica de PyTorch no reconoce la clase `Conv1D` propia de HuggingFace, requiriendo una conversión previa a `nn.Linear`. Este es un obstáculo práctico real en despliegue.
- **Calidad de generación en español**: GPT-2 tiene capacidad limitada en español. Se utilizó `repetition_penalty=1.2` para mitigar repeticiones degenerativas inherentes al greedy decoding.
- **Dependencia de hardware**: La aceleración real depende de si el hardware posee instrucciones nativas de vectorización para tipos de datos reducidos (FP16/INT8, AVX512-VNNI).

---

## Qué se muestra en el video
1. **Presentación del proyecto**: Identificación del Proyecto 8 y el problema de optimizar modelos para recursos acotados.
2. **Línea base en vivo**: Ejecución del modelo `gpt2` en FP32, interpretando memoria y throughput.
3. **Ejercicio A (modificación común)**: Modificación en vivo del tamaño del prompt, prediciendo el impacto en latencia del TTFT y validando la hipótesis.
4. **Ejercicio B (modificación específica)**: Implementación en vivo de la conversión Conv1D → nn.Linear y posterior cuantización INT8, prediciendo la reducción de memoria e interpretando el resultado.
5. **Preguntas avanzadas**: Respuestas orales a las 5 preguntas del Proyecto 8 y a las preguntas transversales.
6. **Cierre técnico y puente al curso**: Conexión con cuantización, edge vs cloud y monitorización MLOps.

---

## Declaración de autoría y uso de IA

```
Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta práctica.
Si utilicé herramientas de IA, las use como apoyo para redacción, depuración o consulta, pero la implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mia.
```

**Uso específico de herramientas de IA en esta entrega**:
- Utilicé IA para auditar la claridad técnica de la redacción en el README.
- Utilicé IA para identificar que GPT-2 usa `Conv1D` en vez de `nn.Linear` y diseñar la conversión previa a cuantización.
- La implementación del pipeline de medición, la interpretación de resultados y la defensa técnica son propias.
