import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# ... (el resto de tus imports) ...

# --- IMPORTANTE: Configura tu clave de API aquí si ejecutas localmente sin Streamlit Cloud secrets ---
# En Streamlit Cloud, la API Key debe gestionarse vía st.secrets (configuración avanzada)
# Si lo ejecutas localmente, puedes descomentar la línea de abajo y poner tu clave.
# os.environ["GOOGLE_API_KEY"] = "TU_API_KEY_AQUI" 

# --- Credenciales de Autenticación para el prototipo ---
# En un entorno de producción, esto debería gestionarse de forma más segura (ej. st.secrets, bases de datos)
USERNAME = "muriel.espinosa"
PASSWORD = "MurEsp2017..??" # !!! CAMBIAR ESTO POR UN VALOR MÁS FUERTE Y SEGURO !!!

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Asistente Académico de Ingeniería", page_icon="🎓", layout="wide")

# --- Función de Autenticación ---
def check_password():
    """Returns `True` if the user's password is correct, `False` otherwise."""

    # Inicializar el estado de autenticación si no existe
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if st.session_state["password_correct"]:
        return True # Ya autenticado

    # Mostrar formulario de login si no está autenticado
    st.sidebar.title("Acceso al Asistente")
    with st.sidebar.form("login_form"):
        st.text_input("Usuario", key="username_input")
        st.text_input("Contraseña", type="password", key="password_input")
        submitted = st.form_submit_button("Entrar")

        if submitted:
            if (st.session_state.username_input == USERNAME and 
                st.session_state.password_input == PASSWORD):
                st.session_state["password_correct"] = True
                st.success("¡Acceso concedido! Recargando...")
                st.rerun() # Forzar un rerun para ocultar el login
            else:
                st.error("😕 Usuario o contraseña incorrectos")
        return False

# --- Inclusión de Logos en el Encabezado ---
col1, col2 = st.columns([1, 5]) # Ajusta la proporción de las columnas para los logos y el título

with col1:
    try:
        logo_uoh = Image.open("logo_uoh.png")
        st.image(logo_uoh, width=100) # Ajusta el ancho según sea necesario
    except FileNotFoundError:
        st.error("Logo UOH no encontrado. Asegúrate de que 'logo_uoh.png' esté en la carpeta.")
    
    try:
        logo_eIng = Image.open("logo_eIng.png")
        st.image(logo_eIng, width=100) # Ajusta el ancho según sea necesario
    except FileNotFoundError:
        st.error("Logo E.Ing no encontrado. Asegúrate de que 'logo_eIng.png' esté en la carpeta.")

with col2:
    st.title("🎓 Asistente Académico IA de Ingeniería")
st.write("Bienvenido/a. Soy tu asistente para consultar patrones y tendencias de los datos agregados de estudiantes.")


@st.cache_resource
def cargar_recursos():
    """
    Carga los recursos pesados (modelo, DB) una sola vez para que la app sea rápida.
    """
    if not os.path.exists("chroma_db"):
        st.error("¡La base de datos 'chroma_db' no ha sido creada! Por favor, ejecuta primero el script 'preparar_base_de_datos.py'.")
        return None

    st.info("Cargando base de datos y modelo... Por favor, espera.")
    
    # Comprobamos la API Key antes de cargar los embeddings
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Error: GOOGLE_API_KEY no configurada. No se pueden cargar los modelos.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.2, convert_system_message_to_human=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # Recupera 5 documentos más relevantes
    
    template = """
    Eres un asistente académico experto en analizar datos de estudiantes. Tu objetivo en esta FASE 1 es brindar insights sobre **patrones, tendencias y perfiles cualitativos a nivel agregado**. NO debes responder preguntas sobre datos específicos de un solo estudiante (ej. RUT, nombre, historial individual).

    Usa la siguiente información de contexto para responder la pregunta de manera precisa y concisa, siempre enfocándote en las tendencias generales.
    El contexto se compone de fichas de estudiantes desidentificadas. Analízalas para identificar patrones.
    Si la información para identificar un patrón no está en el contexto, indica que no tienes datos suficientes. No inventes respuestas.
    Organiza tu respuesta de forma clara y fácil de leer.

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta útil:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    st.success("¡Recursos cargados! Ya puedes hacer tus preguntas sobre patrones.")
    return qa_chain

# --- Lógica principal de la aplicación, protegida por la autenticación ---
if not check_password():
    st.stop() # Detiene la ejecución si el usuario no está autenticado

# --- Si la autenticación es exitosa, el resto del código se ejecuta ---
if not os.environ.get("GOOGLE_API_KEY"): # Doble chequeo por si no se cargó por Streamlit Cloud o local
    st.warning("🚨 ¡ALERTA! La GOOGLE_API_KEY no está configurada. La aplicación no funcionará correctamente.")
else:
    chain = cargar_recursos()

    if chain:
        # Inicializamos las variables en el estado de la sesión si no existen
        if 'pregunta_usuario' not in st.session_state:
            st.session_state.pregunta_usuario = ""
        if 'respuesta' not in st.session_state:
            st.session_state.respuesta = None
        if 'fuentes' not in st.session_state:
            st.session_state.fuentes = None
        
        with st.form(key='query_form'):
            pregunta_actual = st.text_area(
                "Escribe tu pregunta aquí (enfocada en patrones o tendencias):",
                value=st.session_state.pregunta_usuario,
                height=150, 
                placeholder="Ej: Describe los factores comunes en los estudiantes con alto rendimiento en el primer año."
            )
            submit_button = st.form_submit_button(label='Generar Respuesta 🚀')

        if submit_button:
            st.session_state.pregunta_usuario = pregunta_actual # Guardar la pregunta en sesión

            # --- Lógica de FASE 1: Limitar a preguntas sobre patrones/tendencias, rechazando preguntas individuales ---
            # Palabras clave que sugieren una pregunta sobre un estudiante específico
            pregunta_individual_keywords = ["rut", "id de estudiante", "nombre", "historial de", "situación de", "perfil de", "dame la info de"]
            es_pregunta_individual = any(keyword in pregunta_actual.lower() for keyword in pregunta_individual_keywords)

            if es_pregunta_individual:
                st.session_state.respuesta = (
                    "**Respuesta del Sistema (Fase 1):**\n\n"
                    "En esta Fase 1, el asistente está diseñado para brindar información sobre **tendencias y patrones agregados**. "
                    "Las consultas sobre datos específicos de un estudiante individual (ej. usando RUT, nombre o ID) "
                    "no están habilitadas para garantizar la privacidad y se implementarán en fases futuras "
                    "con estrictos controles de acceso."
                )
                st.session_state.fuentes = [] # No hay fuentes para este tipo de respuesta de política
            elif not pregunta_actual: # Manejar el caso de una pregunta vacía
                st.warning("Por favor, ingresa una pregunta para generar una respuesta.")
                st.session_state.respuesta = None
                st.session_state.fuentes = None
            else: # Si la pregunta NO es individual y no está vacía, intentamos procesarla con RAG (para patrones agregados)
                with st.spinner("Buscando en los archivos desidentificados y generando una respuesta... 🧠"):
                    try:
                        resultado = chain.invoke({"query": st.session_state.pregunta_usuario}) 
                        st.session_state.respuesta = resultado["result"]
                        st.session_state.fuentes = resultado["source_documents"]
                    except Exception as e:
                        st.error(f"Ocurrió un error al contactar la API. Revisa tu clave de API y la conexión a internet. Error: {e}")
                        st.session_state.respuesta = None
                        st.session_state.fuentes = None

        # Mostramos la respuesta si existe en el estado de la sesión
        if st.session_state.respuesta:
            st.write("### Respuesta a tu consulta:")
            st.info(st.session_state.respuesta)
            
            with st.expander("Ver fuentes de datos consultadas (Fichas desidentificadas)"):
                if st.session_state.fuentes: # Solo iterar si hay fuentes
                    for doc in st.session_state.fuentes:
                        st.write("---")
                        st.write(doc.page_content)
                else:
                    st.write("No hay fuentes de datos directas para esta respuesta (respuesta de política de la Fase 1).")