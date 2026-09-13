from pathlib import Path
import re
import unicodedata

import joblib
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from Api.utils import preprocess_text, predict


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "tfidf_svd_ods.joblib"
HERO_IMAGE = (
    "https://images.unsplash.com/photo-1677442136019-21780ecad995?"
    "auto=format&fit=crop&w=1800&q=85"
)

LANGUAGES = {
    "Español": ("spanish", "spanish"),
    "English": ("english", "english"),
    "Français": ("french", "french"),
    "Português": ("portuguese", "portuguese"),
}


st.set_page_config(page_title="ODS Intelligence", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #eaf6ff;
        --muted: #91afc7;
        --panel: rgba(10, 28, 49, .72);
        --line: rgba(126, 211, 255, .20);
        --cyan: #55e7ff;
        --blue: #5f8cff;
        --lime: #b6ff7a;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        color: var(--ink);
        background: radial-gradient(circle at 85% 5%, rgba(48, 134, 255, .22), transparent 30%),
                    radial-gradient(circle at 10% 55%, rgba(0, 229, 255, .10), transparent 28%),
                    #06111f;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: rgba(4, 17, 31, .92); border-right: 1px solid var(--line); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h1 { font-size: clamp(2.2rem, 5vw, 4.8rem); line-height: 1.02; }
    .hero {
        min-height: 320px; padding: 42px 46px; margin-bottom: 26px;
        border: 1px solid var(--line); border-radius: 28px; overflow: hidden;
        background: linear-gradient(110deg, rgba(4, 19, 36, .96) 0%, rgba(7, 27, 50, .82) 48%, rgba(7, 27, 50, .30) 100%),
                    url('https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=1800&q=85') center/cover;
        box-shadow: 0 24px 80px rgba(0, 0, 0, .28);
    }
    .eyebrow { color: var(--cyan); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .hero p { max-width: 620px; color: #b9d4e9; font-size: 1.05rem; }
    .glass { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 22px; }
    .tag { display: inline-block; color: var(--lime); border: 1px solid rgba(182,255,122,.35); border-radius: 999px; padding: 5px 11px; font-size: .76rem; }
    .stButton > button { border-radius: 12px; border: 1px solid rgba(85,231,255,.55); min-height: 46px; font-weight: 700; }
    .stTextArea textarea, .stTextInput input { background: rgba(5, 20, 36, .82); color: var(--ink); border: 1px solid var(--line); border-radius: 14px; }
    [data-testid="stMetric"] { background: rgba(14, 43, 70, .65); border: 1px solid var(--line); border-radius: 16px; padding: 14px; }
    .stCaption, small { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No existe tfidf_svd_ods.joblib. Ejecuta primero las celdas de entrenamiento del notebook."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource
def build_text_processor(language):
    stop_language, stem_language = LANGUAGES[language]
    try:
        stop_words = set(stopwords.words(stop_language))
    except LookupError as error:
        raise RuntimeError(
            "Faltan las stopwords de NLTK. Ejecuta en el entorno de Streamlit: "
            "python -m nltk.downloader stopwords"
        ) from error
    return stop_words, SnowballStemmer(stem_language)


def preprocess_text(text, stop_words, stemmer):
    text = "" if text is None else str(text)
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) >= 2 and not token.isnumeric()
    ]
    return " ".join(tokens)


def predict(text, artifact, language):
    stop_words, stemmer = build_text_processor(language)
    processed_text = preprocess_text(text, stop_words, stemmer)
    if not processed_text:
        return processed_text, []
    tfidf = artifact["vectorizer"].transform([processed_text])
    lsa = artifact["svd"].transform(tfidf)
    probabilities = artifact["classifier"].predict_proba(lsa)[0]
    order = probabilities.argsort()[::-1][:3]
    results = [
        (str(artifact["classifier"].classes_[index]), float(probabilities[index]))
        for index in order
    ]
    return processed_text, results


try:
    artifact = load_artifact()
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.markdown("## ODS Intelligence")
    st.markdown("Clasificación semántica para textos con TF-IDF + LSA")
    language = st.selectbox("Idioma del texto", list(LANGUAGES))
    st.divider()
    st.markdown("**Estado del modelo**")
    st.success("Modelo listo")
    st.caption(f"{len(artifact['labels'])} clases · accuracy {artifact['accuracy']:.2%}")
    st.caption("El modelo fue entrenado principalmente con textos en español. El selector adapta la limpieza lingüística; al usar otro idioma, la predicción puede perder precisión.")

st.markdown(
    '<section class="hero"><div class="eyebrow">NLP · IA · DESARROLLO SOSTENIBLE</div>'
    '<h1>Convierte lenguaje en impacto.</h1>'
    '<p>Explora qué Objetivo de Desarrollo Sostenible representa mejor cada texto, desde una frase hasta un archivo completo.</p>'
    '<span class="tag">TF-IDF + LSA + Regresión logística</span></section>',
    unsafe_allow_html=True,
)

st.markdown("### Centro de análisis")
tab_text, tab_file = st.tabs(["✦ Texto individual", "↥ Archivo CSV / Excel"])

with tab_text:
    text = st.text_area(
        "Texto para clasificar",
        height=190,
        placeholder="Escribe aquí una iniciativa, noticia o descripción relacionada con los ODS...",
    )
    if st.button("Analizar texto", type="primary", disabled=not text.strip(), key="predict_text"):
        processed_text, results = predict(text, artifact, language)
        if not results:
            st.warning("El procesamiento no encontró tokens utilizables.")
        else:
            top_label, top_probability = results[0]
            left, right = st.columns([1.35, 1])
            with left:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.markdown(f"### ODS predicho: **{top_label}**")
                st.progress(top_probability, text=f"Confianza estimada · {top_probability:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            with right:
                st.metric("Idioma seleccionado", language)
                st.metric("Tokens procesados", len(processed_text.split()))
            with st.expander("Ver alternativas y texto procesado"):
                st.dataframe(pd.DataFrame(results, columns=["ODS", "Probabilidad"]).assign(Probabilidad=lambda frame: frame["Probabilidad"].map(lambda value: f"{value:.2%}")), hide_index=True, use_container_width=True)
                st.code(processed_text or "(sin tokens despues del procesamiento)")

with tab_file:
    uploaded_file = st.file_uploader("Sube una tabla con textos", type=["csv", "xlsx"], key="input_file")
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                data = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as error:
            st.error(f"No fue posible leer el archivo: {error}")
        else:
            text_columns = [column for column in data.columns if data[column].dtype == "object"]
            if not text_columns:
                st.error("El archivo no contiene columnas de texto seleccionables.")
            else:
                column = st.selectbox("Columna que contiene el texto", text_columns)
                st.caption(f"{len(data):,} filas detectadas · idioma: {language}")
                if st.button("Clasificar archivo", type="primary", key="predict_file"):
                    progress = st.progress(0, text="Procesando textos...")
                    rows = []
                    total = len(data)
                    for position, original_text in enumerate(data[column].fillna("")):
                        processed_text, results = predict(original_text, artifact, language)
                        label, probability = results[0] if results else ("Sin tokens", 0.0)
                        rows.append({"texto_original": original_text, "texto_procesado": processed_text, "ODS_predicho": label, "confianza": probability})
                        progress.progress((position + 1) / total if total else 1.0, text=f"Procesando {position + 1} de {total}")
                    result_data = pd.DataFrame(rows)
                    st.session_state["file_results"] = result_data

                if "file_results" in st.session_state:
                    result_data = st.session_state["file_results"]
                    st.dataframe(result_data.style.format({"confianza": "{:.2%}"}), hide_index=True, use_container_width=True)
                    st.download_button("Descargar resultados CSV", result_data.to_csv(index=False).encode("utf-8-sig"), "resultados_ods.csv", "text/csv", key="download_results")
