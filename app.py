# app.py
import os
import streamlit as st
import qdrant_client
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import PromptTemplate

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    return qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )

# Initialize embedding model
@st.cache_resource
def init_embedding():
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        trust_remote_code=True
    )

# Initialize OpenAI
def init_openai():
    return OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=512
    )

# Initialize query engine
@st.cache_resource
def init_query_engine(_client, collection_name):
    try:
        # Set up embedding model
        embed_model = init_embedding()
        
        # Set up vector store
        vector_store = QdrantVectorStore(
            client=_client, 
            collection_name=collection_name
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Create reranker
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
            top_n=3
        )
        
        # Define prompt template
        template = """Context information is below:
                    ---------------------
                    {context_str}
                    ---------------------
                    Based on the context above, analyze the query and provide the response in the following format:
                    
                    Scenario: [Describe the situation from matching context]
                    Remediation: [Provide specific prevention/remediation steps]
                    Points of contact: [List relevant contact information/helplines]
                    
                    If no relevant information is found in the context, respond with "No matching scenario found."
                    
                    Query: {query_str}
                    
                    Response:"""
        
        qa_prompt_tmpl = PromptTemplate(template)
        
        # Create and configure query engine
        query_engine = index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[rerank]
        )
        
        # Update query engine with template
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        
        return query_engine
    except Exception as e:
        st.error(f"Error initializing query engine: {str(e)}")
        raise e

def main():
    st.title("Cyber Security RAG Chatbot")
    
    # Set OpenAI API key
    if 'OPENAI_API_KEY' not in st.session_state:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state['OPENAI_API_KEY'] = api_key
            st.success("API key set successfully!")
    
    if 'OPENAI_API_KEY' in st.session_state:
        try:
            # Initialize components
            client = init_qdrant()
            collection_name = "demo_29thJan"  # Use your collection name
            
            # Initialize LLM
            llm = init_openai()
            
            # Initialize query engine
            query_engine = init_query_engine(client, collection_name)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("What would you like to know about cyber security?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = query_engine.query(prompt)
                        st.markdown(str(response))
                        st.session_state.messages.append({"role": "assistant", "content": str(response)})
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()