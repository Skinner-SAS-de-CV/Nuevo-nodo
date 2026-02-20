"""
ETL Pipeline para mallas curriculares universitarias salvadoreñas.

Limpia, normaliza y prepara los datos para modelos TF-IDF y sentence-transformer.
"""

import os
import re
import pandas as pd
from nltk.stem import SnowballStemmer

# ── Stop words en español (embebidas para evitar dependencia de nltk.download) ──
STOP_WORDS_ES = frozenset([
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
    "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas",
    "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus",
    "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos",
    "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas",
    "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros",
    "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis",
    "están", "esté", "estés", "estemos", "estéis", "estén", "estaré", "estarás",
    "estará", "estaremos", "estaréis", "estarán", "estaría", "estarías",
    "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos",
    "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos",
    "estuvisteis", "estuvieron", "estuviera", "estuvieras", "estuviéramos",
    "estuvierais", "estuvieran", "estuviese", "estuvieses", "estuviésemos",
    "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados",
    "estadas", "estad", "he", "has", "ha", "hemos", "habéis", "han", "haya",
    "hayas", "hayamos", "hayáis", "hayan", "habré", "habrás", "habrá",
    "habremos", "habréis", "habrán", "habría", "habrías", "habríamos",
    "habríais", "habrían", "había", "habías", "habíamos", "habíais", "habían",
    "hube", "hubiste", "hubo", "hubimos", "hubisteis", "hubieron", "hubiera",
    "hubieras", "hubiéramos", "hubierais", "hubieran", "hubiese", "hubieses",
    "hubiésemos", "hubieseis", "hubiesen", "habiendo", "habido", "habida",
    "habidos", "habidas", "soy", "eres", "es", "somos", "sois", "son", "sea",
    "seas", "seamos", "seáis", "sean", "seré", "serás", "será", "seremos",
    "seréis", "serán", "sería", "serías", "seríamos", "seríais", "serían",
    "era", "eras", "éramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos",
    "fuisteis", "fueron", "fuera", "fueras", "fuéramos", "fuerais", "fueran",
    "fuese", "fueses", "fuésemos", "fueseis", "fuesen", "siendo", "sido",
    "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen", "tenga",
    "tengas", "tengamos", "tengáis", "tengan", "tendré", "tendrás", "tendrá",
    "tendremos", "tendréis", "tendrán", "tendría", "tendrías", "tendríamos",
    "tendríais", "tendrían", "tenía", "tenías", "teníamos", "teníais", "tenían",
    "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera",
    "tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese", "tuvieses",
    "tuviésemos", "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida",
    "tenidos", "tenidas", "tened",
])

# ── Mapeo Bloom: sustantivo → infinitivo ──
BLOOM_SUSTANTIVO_A_VERBO = {
    "Análisis":    "Analizar",
    "Aplicación":  "Aplicar",
    "Comprensión": "Comprender",
    "Creación":    "Crear",
    "Evaluación":  "Evaluar",
}

# ── Mapeo area_dominio: nombre legible → SCREAMING_SNAKE_CASE ──
AREA_DOMINIO_MAP = {
    "Ciencias de la Salud - Medicina General":  "MEDICINA_GENERAL",
    "Ciencias de la Salud - Enfermería":        "ENFERMERIA",
    "Ciencias de la Salud - Nutrición":         "NUTRICION",
    "Ingeniería y Tecnología":                  "INGENIERIA_INFORMATICA",
    "Profesorado en Lenguaje y Literatura":     "PROFESORADO_LENGUAJE_LITERATURA",
}

# Regex para las variantes de Sociología con paréntesis
_RE_SOCIOLOGIA = re.compile(r"^Licenciatura en Sociología")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga
# ─────────────────────────────────────────────────────────────────────────────
def _reparar_linea_csv(campos: list[str]) -> list[str]:
    """
    Callback para on_bad_lines: cuando una fila tiene 8 campos en vez de 7,
    la coma extra está dentro de resultado_aprendizaje (campo índice 3).
    Fusiona campos[3] y campos[4] y desplaza el resto.
    """
    if len(campos) == 8:
        campos[3] = campos[3] + "," + campos[4]
        return campos[:4] + campos[5:]
    return campos[:7]


def cargar_csv_raw(carpeta: str) -> pd.DataFrame:
    """Lee todos los CSV de la carpeta y los concatena con trazabilidad."""
    frames = []
    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.endswith(".csv"):
            continue
        ruta = os.path.join(carpeta, archivo)
        df = pd.read_csv(ruta, on_bad_lines=_reparar_linea_csv, engine="python")
        df.columns = df.columns.str.strip()
        df["archivo_origen"] = archivo
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No se encontraron CSV en {carpeta}")

    resultado = pd.concat(frames, ignore_index=True)
    resultado["id"] = range(1, len(resultado) + 1)
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normalización Bloom
# ─────────────────────────────────────────────────────────────────────────────
def normalizar_bloom(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte etiquetas Bloom en sustantivo a su forma infinitivo."""
    df["nivel_bloom"] = df["nivel_bloom"].str.strip()
    df["nivel_bloom"] = df["nivel_bloom"].replace(BLOOM_SUSTANTIVO_A_VERBO)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Normalización area_dominio
# ─────────────────────────────────────────────────────────────────────────────
def normalizar_area_dominio(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica area_dominio a formato SCREAMING_SNAKE_CASE."""
    df["area_dominio"] = df["area_dominio"].str.strip()

    # Mapeo directo
    df["area_dominio"] = df["area_dominio"].replace(AREA_DOMINIO_MAP)

    # Sociología con variantes entre paréntesis → LIC_SOCIOLOGIA
    mask = df["area_dominio"].apply(lambda x: bool(_RE_SOCIOLOGIA.match(str(x))))
    df.loc[mask, "area_dominio"] = "LIC_SOCIOLOGIA"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Limpieza de whitespace
# ─────────────────────────────────────────────────────────────────────────────
def limpiar_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina espacios extra y non-breaking spaces en columnas string."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
        )
        # Restaurar NaN reales
        df.loc[df[col] == "nan", col] = pd.NA
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reparación de filas corruptas
# ─────────────────────────────────────────────────────────────────────────────
def reparar_filas_corruptas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repara filas de tecnico_electronica_compu.csv donde comas sin escapar
    en resultado_aprendizaje desplazaron texto a competencia_laboral_relacionada.

    Patrón: competencia_laboral_relacionada empieza con espacio y es continuación
    del resultado_aprendizaje (ej: " eficiencia y escalabilidad...").
    """
    mask = (
        (df["archivo_origen"] == "tecnico_electronica_compu.csv")
        & df["competencia_laboral_relacionada"].notna()
        & df["competencia_laboral_relacionada"].str.match(r"^\s")
    )

    n_reparadas = mask.sum()
    if n_reparadas > 0:
        df.loc[mask, "resultado_aprendizaje"] = (
            df.loc[mask, "resultado_aprendizaje"]
            + ","
            + df.loc[mask, "competencia_laboral_relacionada"]
        )
        df.loc[mask, "resultado_aprendizaje"] = (
            df.loc[mask, "resultado_aprendizaje"].str.strip()
        )
        df.loc[mask, "competencia_laboral_relacionada"] = pd.NA
        print(f"  → {n_reparadas} filas corruptas reparadas en tecnico_electronica_compu.csv")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Preparación texto TF-IDF (stemming + sin stop words)
# ─────────────────────────────────────────────────────────────────────────────
def preparar_texto_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera columna texto_limpio_tfidf con stemming español y sin stop words.
    Usa SnowballStemmer porque spacy no es compatible con Python 3.14.
    """
    stemmer = SnowballStemmer("spanish")

    def procesar(texto: str) -> str:
        if pd.isna(texto):
            return ""
        tokens = re.findall(r"\b[a-záéíóúñü]+\b", texto.lower())
        tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS_ES]
        return " ".join(tokens)

    df["texto_limpio_tfidf"] = (
        df["resultado_aprendizaje"].fillna("")
        + " "
        + df["nombre_asignatura"].fillna("")
    ).apply(procesar)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Preparación texto transformer (texto crudo, sin lematizar)
# ─────────────────────────────────────────────────────────────────────────────
def preparar_texto_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera columna texto_transformer con texto crudo para sentence-transformers.
    No se lematiza ni se eliminan stop words para mantener semántica completa.
    """
    df["texto_transformer"] = (
        df["resultado_aprendizaje"].fillna("")
        + " "
        + df["nombre_asignatura"].fillna("")
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Orquestador
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_etl(ruta_entrada: str, ruta_salida: str) -> pd.DataFrame:
    """Ejecuta el pipeline ETL completo e imprime reporte."""
    print("=" * 60)
    print("ETL Pipeline — Mallas Curriculares")
    print("=" * 60)

    # Paso 1: Carga
    print("\n[1/7] Cargando CSV raw...")
    df = cargar_csv_raw(ruta_entrada)
    print(f"  → {len(df)} filas cargadas de {df['archivo_origen'].nunique()} archivos")

    # Paso 2: Reparar filas corruptas (antes de limpiar whitespace)
    print("\n[2/7] Reparando filas corruptas...")
    df = reparar_filas_corruptas(df)

    # Paso 3: Limpiar whitespace
    print("\n[3/7] Limpiando whitespace y \\xa0...")
    df = limpiar_whitespace(df)

    # Paso 4: Normalizar Bloom
    print("\n[4/7] Normalizando etiquetas Bloom...")
    bloom_antes = df["nivel_bloom"].nunique()
    df = normalizar_bloom(df)
    bloom_despues = df["nivel_bloom"].nunique()
    print(f"  → {bloom_antes} etiquetas únicas → {bloom_despues} etiquetas únicas")
    print(f"  → Valores: {sorted(df['nivel_bloom'].unique())}")

    # Paso 5: Normalizar area_dominio
    print("\n[5/7] Normalizando area_dominio...")
    areas_antes = df["area_dominio"].nunique()
    df = normalizar_area_dominio(df)
    areas_despues = df["area_dominio"].nunique()
    print(f"  → {areas_antes} áreas únicas → {areas_despues} áreas únicas")
    print(f"  → Valores: {sorted(df['area_dominio'].unique())}")

    # Paso 6: Texto TF-IDF
    print("\n[6/7] Preparando texto para TF-IDF (stemming español)...")
    df = preparar_texto_tfidf(df)
    print(f"  → Columna 'texto_limpio_tfidf' creada")

    # Paso 7: Texto transformer
    print("\n[7/7] Preparando texto para sentence-transformer...")
    df = preparar_texto_transformer(df)
    print(f"  → Columna 'texto_transformer' creada")

    # Guardar
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f"\n{'=' * 60}")
    print(f"CSV limpio guardado en: {ruta_salida}")
    print(f"Dimensiones finales: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    print(f"{'=' * 60}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ejecutar_etl(
        ruta_entrada=os.path.join(BASE, "data", "csv_raw"),
        ruta_salida=os.path.join(BASE, "data", "csv_limpio", "mallas_curriculares_limpio.csv"),
    )
