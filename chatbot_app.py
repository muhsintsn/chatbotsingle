import os
import streamlit as st
import google.generativeai as ggi
import tempfile


#####################
with st.sidebar:





    
    tab11, tab22 = st.tabs(["Cat", "Dog" ])

    with tab11:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

    with tab22:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

################
tab1, tab2 = st.tabs(["Chatbot", "summary" ])


with tab1:
        st.header("Chatbot")
        st.image("https://static.streamlit.io/examples/cat.jpg", width=200)


# Using "with" notation

###################

# Using object notation

        fetcheed_api_key = os.getenv("AIzaSyDOAB75tWm2-34Ynlqt6c3trG1ijwkyGx8")
        ggi.configure(api_key = fetcheed_api_key)
        #file uploader and reader
        from langchain.document_loaders import PyPDFium2Loader
        # Belgeyi okuma fonksiyonu
        def read_doc(file_path):
            try:
                #st.write(f"Dosya yolu: {file_path}")
                if not os.path.exists(file_path):
                    raise ValueError(f"Dosya {file_path} yolunda mevcut değil.")
                # Dosya uzantısının pdf olduğundan emin olun
                if not file_path.lower().endswith('.pdf'):
                    raise ValueError("Dosya uzantısı .pdf olmalı.")
                # PyPDFium2Loader kullanarak PDF'yi yükleme
                file_loader = PyPDFium2Loader(file_path)
                pdf_documents = file_loader.load()
                #st.write(f"PDF belgeleri: {pdf_documents}")
                return pdf_documents
            except Exception as e:
                st.error(f"Belge okunurken hata oluştu: {e}")
                return None
        # Streamlit dosya yükleyici
        uploaded_file = st.file_uploader('Bir PDF dosyası seçin', type='pdf')
        if uploaded_file is not None:
            try:
                # Yüklenen dosyayı geçici olarak kaydetme
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                # Dosya yolunu ve varlığını doğrulama
                #st.write(f"Geçici dosya yolu: {tmp_file_path}")
                #if os.path.exists(tmp_file_path):
                    #st.write("Geçici dosya mevcut.")
                #else:
                    #st.error("Geçici dosya mevcut değil.")
                # PyPDFium2Loader kullanarak PDF'yi okuma
                pdf = read_doc(tmp_file_path)
                if pdf:
                    #st.write(pdf)
                    st.success('Başarıyla yüklendi!')
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
        else:
            st.markdown('Lütfen bir PDF dosyası yükleyin')
        #Textsplit
        from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
        def chunk_data(docs, chunk_size=800, chunk_overlap=200):
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap)
            pdf=text_splitter.split_documents(docs)
            return pdf

        pdf_doc=chunk_data(docs=pdf)

        #Embedding
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type='retrieval_document', google_api_key='AIzaSyDOAB75tWm2-34Ynlqt6c3trG1ijwkyGx8')
        from langchain_community.vectorstores import Chroma
        index=Chroma().from_documents(documents=pdf_doc,
                                    embedding=embeddings,
                                    persist_directory="/Users/muhsi/Desktop/Streamlit_chatbot/vectorstore")
        loaded_index=Chroma(persist_directory="/Users/muhsi/Desktop/Streamlit_chatbot/vectorstore",
                            embedding_function=embeddings)
        def retrieve_query(query,k=5):
            matching_results=index.similarity_search(query,k=k)
            return matching_results
        from langchain_google_genai import  GoogleGenerativeAI, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
        from langchain.chains.question_answering import load_qa_chain
        import textwrap
        llm = GoogleGenerativeAI(
            model="gemini-1.5-pro-latest",temperature=0,google_api_key='AIzaSyDOAB75tWm2-34Ynlqt6c3trG1ijwkyGx8',
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        chain=load_qa_chain(llm, chain_type="stuff")
        def get_answers(query):
            doc_search=retrieve_query(query)
            response=chain.invoke(input={"input_documents":doc_search, "question":query})["output_text"]
            wrapped_text = textwrap.fill(response, width=100)
            return wrapped_text
        st.title("Chat Application using Gemini Pro")
        user_quest = st.text_input("Ask a question:")
        btn = st.button("Ask")
        if btn and user_quest:
            result = get_answers(user_quest)
            st.subheader("Response : ")
            st.write(result)



##summery
    
with tab2:
        st.header("summary")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)