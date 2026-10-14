import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # Changed to Chat class
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

st.title("🤖 Chatbot with LangChain")
st.write("This is a chat application using LangChain and Streamlit.")

# Initialize the chat model (optimized for chat history)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])    

# Accept user input
query = st.chat_input("Enter your query here...")
if query:
    # 1. Display user message and append to session state
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # 2. Convert raw session state into LangChain format to give model context memory
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))

    # 3. Get response from LLM passing the FULL message history
    with st.chat_message("assistant"):
        response = llm.invoke(langchain_messages)
        st.markdown(response.text)
        
    # 4. Save assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response.text})
