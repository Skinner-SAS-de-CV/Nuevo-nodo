# Clasificador de Resultados de Aprendizaje - Taxonomia de Bloom

Pipeline ETL y clasificador de machine learning para datos curriculares de universidades salvadorenas (mallas curriculares). El sistema clasifica aproximadamente 500 resultados de aprendizaje en los seis niveles de la taxonomia de Bloom utilizando TF-IDF + RandomForest, alcanzando un 93% de accuracy y 0.92 de F1 macro. Todo el procesamiento de texto se realiza en espanol.

## Estructura del proyecto

```
data/
  csv_raw/                  # 12 archivos CSV con datos crudos por programa academico
  csv_limpio/               # Dataset limpio consolidado (500 filas x 10 columnas)
  csv_unido/                # CSV unificado intermedio
  mallas_curriculares_pdf/  # Documentos PDF fuente de las mallas curriculares
scripts/
  etl_pipeline.py           # Pipeline ETL de 7 pasos
notebook/
  etl.ipynb                 # Entrenamiento del modelo (TF-IDF + RandomForest)
```

## Requisitos

- Python 3.14
- Dependencias principales: pandas, nltk, scikit-learn, jupyter

## Instalacion y uso

```bash
# Activar el entorno virtual
source virtual/bin/activate

# Ejecutar el pipeline ETL (genera data/csv_limpio/mallas_curriculares_limpio.csv)
python scripts/etl_pipeline.py

# Ejecutar el notebook de entrenamiento
jupyter nbconvert --to notebook --execute notebook/etl.ipynb --output etl_out.ipynb
```

## Pipeline ETL

El script `scripts/etl_pipeline.py` ejecuta los siguientes pasos:

1. **Carga** de los 12 archivos CSV desde `data/csv_raw/`
2. **Reparacion** de filas corruptas (comas sin escapar, campos adicionales)
3. **Limpieza** de espacios en blanco y caracteres no imprimibles
4. **Normalizacion de etiquetas Bloom** de 11 variantes a 6 niveles estandar
5. **Normalizacion de area_dominio** de 22 variantes a 19 programas academicos en formato SCREAMING_SNAKE_CASE
6. **Generacion de `texto_limpio_tfidf`** con stemming en espanol (SnowballStemmer) y eliminacion de stop words
7. **Generacion de `texto_transformer`** con texto crudo concatenado para uso futuro con sentence-transformers

## Datos

### Variable objetivo: nivel_bloom

Los seis niveles de la taxonomia de Bloom utilizados como clases:

- Recordar
- Comprender
- Aplicar
- Analizar
- Evaluar
- Crear

### Columnas del dataset limpio

| Columna | Descripcion |
|---|---|
| `id` | Identificador unico |
| `area_dominio` | Programa academico (19 categorias, SCREAMING_SNAKE_CASE) |
| `nombre_asignatura` | Nombre de la asignatura |
| `resultado_aprendizaje` | Texto del resultado de aprendizaje |
| `nivel_bloom` | Nivel de la taxonomia de Bloom (variable objetivo) |
| `competencia_laboral_relacionada` | Competencia laboral asociada |
| `complejidad_estructural` | Indice de complejidad |
| `archivo_origen` | Archivo CSV fuente |
| `texto_limpio_tfidf` | Texto procesado para TF-IDF (stemmed, sin stop words) |
| `texto_transformer` | Texto crudo para sentence-transformers |

## Modelo

El clasificador baseline utiliza TF-IDF como vectorizacion y RandomForest como algoritmo de clasificacion. Los resultados actuales son:

- **Accuracy**: 93%
- **F1 macro**: 0.92

Se contempla como siguiente paso el fine-tuning con `all-MiniLM-L6-v2` usando la columna `texto_transformer`.

## Licencia

Copyright (C) 2026 SKINNER S.A.S de C.V.

Este proyecto esta licenciado bajo la GNU General Public License v3.0. Consultar el archivo [LICENSE](LICENSE) para mas detalles.
