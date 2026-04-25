# app.py
import streamlit as st
from rag_pipeline import load_resources, ask

st.set_page_config(
    page_title="Asistente de codigos de error",
    layout="centered"
)

st.markdown("""
    <style>
        body, .stApp {
            background-color: #FFFFFF;
        }
        .block-container {
            max-width: 700px;
            padding-top: 2rem;
            background-color: #FFFFFF;
        }
        .titulo {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #111111;
        }
        .etiqueta {
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
            color: #333333;
        }
        .fuente-box {
            border: 1.5px solid #2563EB;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.85rem;
            background-color: #FFFFFF;
            color: #222222;
        }
        .divider {
            border: none;
            border-top: 1px solid #2563EB;
            margin: 1.2rem 0;
        }
        /* Input de busqueda */
        .stTextInput input {
            background-color: #FFFFFF !important;
            color: #111111 !important;
            border: 1.5px solid #2563EB !important;
            border-radius: 24px !important;
            padding-left: 1rem !important;
        }
        .stTextInput input:focus {
            border: 2px solid #1D4ED8 !important;
            box-shadow: none !important;
        }
        /* Area de respuesta */
        .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #111111 !important;
            border: 1.5px solid #9CA3AF !important;
            border-radius: 8px !important;
        }
        /* Boton */
        .stButton button {
            background-color: #FFFFFF !important;
            color: #111111 !important;
            border: 1.5px solid #6B7280 !important;
            border-radius: 8px !important;
            font-weight: 400 !important;
        }
        .stButton button:hover {
            background-color: #F3F4F6 !important;
            border: 1.5px solid #374151 !important;
        }
        /* Slider */
        .stSlider .st-emotion-cache-1tqd7mb {
            color: #2563EB !important;
        }
        p, label, span {
            color: #222222;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_resources():
    with st.spinner("Cargando base de conocimiento..."):
        chunks, index, model = load_resources()
    return chunks, index, model

chunks, index, model = get_resources()

st.markdown('<div class="titulo">BMW IA</div>', unsafe_allow_html=True)

query = st.text_input("", placeholder="Search", label_visibility="collapsed")

st.markdown('<div class="etiqueta">Fragmentos usados</div>', unsafe_allow_html=True)
top_k = st.slider("", min_value=1, max_value=10, value=5, label_visibility="collapsed")

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    buscar = st.button("Buscar", use_container_width=True)

if buscar and query:
    with st.spinner("Buscando..."):
        result = ask(query, chunks, index, model, top_k=top_k)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.text_area("Respuesta", value=result["answer"], height=300, label_visibility="visible")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("**Fuentes utilizadas**")
    for i, source in enumerate(result["sources"], 1):
        st.markdown(f"""
        <div class="fuente-box">
            <strong>Fragmento {i}</strong> &nbsp;|&nbsp; {source['source']} &nbsp;|&nbsp; score: {source['score']}<br>
            <span style="color: #555555; font-size: 0.8rem;">{source['fragment'][:250]}...</span>
        </div>
        """, unsafe_allow_html=True)
