import streamlit as st
import os
import io
import pandas as pd
from PIL import Image

# --- SOLUCIÓN TEMPORAL PARA COMPATIBILIDAD DE SQLITE3 EN STREAMLIT CLOUD (SI ES NECESARIO) ---
# Si el error "sqlite3.OperationalError" persiste, intenta descomentar las siguientes dos líneas.
# Sin embargo, con "pysqlite3-binary" y la inicialización explícita, a menudo no son necesarias.
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ------------------------------------------------------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente de Información de Estudiantes - UOH",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Autenticación (con corrección de st.experimental_rerun a st.rerun) ---
def check_password():
    """Returns `True` if the user's password is correct, `False` otherwise."""

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
            # Leer credenciales desde st.secrets si están configuradas
            if "USERNAME" in st.secrets and "PASSWORD" in st.secrets:
                USERNAME = st.secrets["USERNAME"]
                PASSWORD = st.secrets["PASSWORD"]
            else:
                # Fallback a credenciales hardcodeadas (¡CAMBIAR ESTO EN PRODUCCIÓN!)
                USERNAME = "usuario_sochedi"
                PASSWORD = "password_seguro_y_largo" # <--- ¡CAMBIA ESTO!

            if (st.session_state.username_input == USERNAME and 
                st.session_state.password_input == PASSWORD):
                st.session_state["password_correct"] = True
                st.success("¡Acceso concedido! Recargando...")
                st.rerun() # <-- CORRECCIÓN AQUÍ
            else:
                st.error("😕 Usuario o contraseña incorrectos")
        return False

# --- Cargar recursos (LLM y Vector Store) ---
@st.cache_resource
def cargar_recursos():
    # Cargar la GOOGLE_API_KEY desde los secretos de Streamlit Cloud
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("🚨 La clave GOOGLE_API_KEY no está configurada en los secretos de Streamlit.")
        st.stop()
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Inicializar el modelo de Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Inicializar el modelo de lenguaje (LLM)
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.2, convert_system_message_to_human=True) # <-- MODELO ESTABLE

    # Inicializar ChromaDB (con cliente persistente explícito)
    from chromadb import PersistentClient
    try:
        # Intenta cargar el cliente de persistencia
        client = PersistentClient(path="chroma_db")
        # Asegúrate de que el collection_name coincida con el que usaste en preparar_base_de_datos.py
        # Si no especificaste uno, el valor por defecto de LangChain suele ser "langchain"
        vector_store = Chroma(
            client=client,
            collection_name="langchain", # O el nombre que usaste si lo especificaste
            embedding_function=embeddings
        )
        st.sidebar.success("✅ Base de conocimiento cargada.")
    except Exception as e:
        st.error(f"❌ Error al cargar ChromaDB: {e}. Asegúrate que la carpeta 'chroma_db' está completa y no está corrupta.")
        st.stop() # Detener la ejecución si falla la carga
        return None # Devuelve None si falla la carga

    # Plantilla de Prompt (adaptada para un asistente académico)
    template = """Eres un asistente virtual de la Escuela de Ingeniería UOH.
    Tu objetivo es ayudar a los usuarios a obtener información sobre el desempeño académico de los estudiantes, sus datos y sus situaciones.
    Responde las preguntas del usuario basándote únicamente en el siguiente contexto:
    {context}
    Si la pregunta no se puede responder con la información proporcionada en el contexto, simplemente di que no tienes suficiente información para responder.
    No intentes inventar una respuesta.
    Mantén tus respuestas claras, concisas y en español.

    Pregunta del usuario: {question}
    Respuesta:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Crear la cadena de RetrievalQA
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Busca los 3 documentos más relevantes
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    return chain

# --- Lógica principal de la aplicación ---
def main():
    if not check_password():
        st.stop() # Detener la ejecución si no está autenticado

    # Cargar y mostrar logos
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.image(Image.open(io.BytesIO(open("logo_uoh.png", "rb").read())), width=80)
    with col2:
        st.title("Asistente de Información de Estudiantes")
        st.markdown("##### Escuela de Ingeniería UOH")
    with col3:
        st.image(Image.open(io.BytesIO(open("logo_eIng.png", "rb").read())), width=80)

    st.markdown("---")
    st.write("¡Hola! Soy tu asistente virtual. Puedes preguntarme sobre el desempeño académico de los estudiantes de la UOH.")
    st.write("Por ejemplo: '¿Cuál es el PPA de un estudiante con RUT 12.345.678-9?' o '¿Qué asignaturas ha cursado el estudiante con RUT 12.345.678-9 y cómo le fue?'")

    # Botón para limpiar caché y recargar la aplicación (útil para depuración)
    if st.button("🔄 Limpiar caché y Recargar la App"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Cargar recursos (LLM y Vector Store)
    chain = cargar_recursos()

    if chain is None:
        st.error("No se pudo inicializar el asistente. Por favor, revisa los logs.")
        st.stop()

    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de usuario
    if prompt := st.chat_input("¿En qué puedo ayudarte?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando y generando respuesta..."):
                response = chain.invoke(prompt)
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})

                # Opcional: Mostrar documentos fuente para depuración
                # with st.expander("Ver documentos fuente"):
                #     for doc in response["source_documents"]:
                #         st.write(doc.page_content)
                #         st.write(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    # Asegurarse de que el directorio 'chroma_db' exista para que PersistentClient no falle al inicio.
    # En Streamlit Cloud, esto ya debería estar manejado porque la carpeta se sube.
    if not os.path.exists("chroma_db"):
        st.warning("La carpeta 'chroma_db' no se encontró. Asegúrate de que está en el mismo directorio que app.py.")
        # Aquí podrías detener la app o redirigir a un mensaje de error si no es para despliegue.
        # st.stop() # Descomentar esto si la ausencia de la DB debe ser un error fatal.
        
    main()