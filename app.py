import streamlit as st
import os
import io
import pandas as pd
from PIL import Image
import re # <-- Añadir esta importación para expresiones regulares

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

# --- Función para detectar RUTs (NUEVA FUNCIÓN) ---
def contains_rut(text):
    # Expresión regular para detectar RUTs chilenos (XX.XXX.XXX-X o X.XXX.XXX-X)
    # Permite puntos opcionales y guion.
    rut_pattern = r'\b\d{1,2}\.\d{3}\.\d{3}[-][0-9kK]\b|\b\d{7,8}[-][0-9kK]\b'
    return re.search(rut_pattern, text) is not None

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
    Tu objetivo es ayudar a los usuarios a obtener información SOBRE DATOS AGREGADOS DE LOS ESTUDIANTES.
    NO PUEDES PROPORCIONAR INFORMACIÓN INDIVIDUALIZADA O PERSONAL DE NINGÚN ESTUDIANTE.
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
    st.write("Por ejemplo: '¿Cuál es el PPA promedio de los estudiantes de Ingeniería Civil?' o '¿Cuántos estudiantes reprobaron Cálculo I el semestre pasado?'")
    st.write("**⚠️ Importante:** Por motivos de privacidad, no puedo responder preguntas sobre información personal o datos individualizados de estudiantes específicos (como su RUT, nombre o calificaciones personales).")


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
                # --- LÓGICA DE PRIVACIDAD AÑADIDA AQUÍ ---
                if contains_rut(prompt) or "personal" in prompt.lower() or "individual" in prompt.lower() or "nombre" in prompt.lower():
                    response_text = "Lo siento, como asistente de la UOH, no puedo proporcionar información personal o individualizada de los estudiantes por razones de privacidad. Solo puedo responder preguntas sobre datos agregados."
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    # Si no es una pregunta de privacidad, procede con el RAG normal
                    response = chain.invoke(prompt)
                    st.markdown(response["result"])
                    st.session_state.messages.append({"role": "assistant", "content": response["result"]})

                    # Opcional: Mostrar documentos fuente para depuración (descomentar si es necesario)
                    with st.expander("Ver documentos fuente"):
                        if response["source_documents"]:
                            for doc in response["source_documents"]:
                                st.write(doc.page_content)
                                st.write(f"Metadata: {doc.metadata}")
                        else:
                            st.write("No se encontraron documentos relevantes.")
                # ----------------------------------------

if __name__ == "__main__":
    # Asegurarse de que el directorio 'chroma_db' exista para que PersistentClient no falle al inicio.
    if not os.path.exists("chroma_db"):
        st.warning("La carpeta 'chroma_db' no se encontró. Asegúrate de que está en el mismo directorio que app.py.")
        # st.stop() # Si la ausencia de la DB debe ser un error fatal, descomentar
        
    main()