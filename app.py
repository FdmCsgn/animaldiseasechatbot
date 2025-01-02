import pyodbc
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai import ChatMistralAI
import numpy as np

# Veritabanına bağlanma fonksiyonu
def connect_db():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=AnimalDisease;'
        'UID=sa;'
        'PWD=?'
    )
    return conn

# Konuları txt dosyasından okuma
def load_topics(file_path):
    with open(file_path, 'r') as file:
        topics = file.readlines()
    return [topic.strip() for topic in topics]

# Veritabanından konuya ait chunk'ları alma
def get_chunks_by_topic(conn, topic_name):
    cursor = conn.cursor()
    query = """
        SELECT ChunkText, VectorData
        FROM PDFChunks
        INNER JOIN PDF ON PDFChunks.DocumentID = PDF.DocumentID
        INNER JOIN TOPICS ON PDF.TopicID = TOPICS.TopicID
        WHERE TOPICS.TopicName = ?
    """
    cursor.execute(query, (topic_name,))
    results = cursor.fetchall()
    chunks = [row[0] for row in results]
    vectors = [np.frombuffer(row[1], dtype=np.float32) for row in results]
    return chunks, vectors

# Veritabanında konu var mı kontrol etme
def get_topic_id(conn, topic_name):
    cursor = conn.cursor()
    query = "SELECT TopicID FROM TOPICS WHERE TopicName = ?"
    cursor.execute(query, (topic_name,))
    result = cursor.fetchone()
    
    if result:
        return result[0]  # Konu zaten varsa, ID'sini döndür
    else:
        return None  # Konu yoksa None döndür

# Veritabanında döküman var mı kontrol etme
def get_document_id(conn, document_name, topic_id):
    cursor = conn.cursor()
    query = "SELECT DocumentID FROM PDF WHERE DocumentName = ? AND TopicID = ?"
    cursor.execute(query, (document_name, topic_id))
    result = cursor.fetchone()
    
    if result:
        return result[0]  # Döküman zaten varsa, ID'sini döndür
    else:
        return None  # Döküman yoksa None döndür

# PDF'yi metne dönüştürme
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Textleri chunk'lara ayırma
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Vektör store oluşturma
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()   
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  
    return vectorstore

# Vektör store oluşturma
def get_vectorstore_from_db(text_chunks,vectors):
    embeddings = OpenAIEmbeddings()   
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  
    return vectorstore

# Veritabanına konu ekleme
def insert_topic(conn, topic):
    topic_id = get_topic_id(conn, topic)
    if not topic_id:  # Eğer konu yoksa ekle
        cursor = conn.cursor()
        query = "INSERT INTO TOPICS (TopicName) OUTPUT INSERTED.TopicID VALUES (?)"
        cursor.execute(query, (topic,))
        topic_id = cursor.fetchone()[0]
        conn.commit()
    return topic_id

# Veritabanına PDF verisini ekleme
def insert_document(conn, topic_id, document_data, document_name):
    document_id = get_document_id(conn, document_name, topic_id)
    if not document_id:  # Eğer döküman yoksa ekle
        cursor = conn.cursor()
        query = "INSERT INTO PDF (TopicID, DocumentData, DocumentName) OUTPUT INSERTED.DocumentID VALUES (?, ?, ?)"
        cursor.execute(query, (topic_id, document_data, document_name))
        document_id = cursor.fetchone()[0]
        conn.commit()
    return document_id

# Veritabanına metin chunk'larını ve vektör verilerini kaydetme
def insert_chunk(conn, document_id, chunk_text, vector_data):
    cursor = conn.cursor()
    vector_bytes = vector_data.tobytes()  # FAISS vektör verisini byte formatına dönüştürme
    cursor.execute("INSERT INTO PDFChunks (DocumentID, ChunkText, VectorData) VALUES (?, ?, ?)",
                   (document_id, chunk_text, vector_bytes))
    conn.commit()

# Konuşma zinciri oluşturma
def get_conversion_chain(vectorstore):
    llm = ChatMistralAI(model="open-mistral-nemo")  # Burada doğru model adını kullanın
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Kullanıcı inputunu işleme
def handle_userinput(user_question):
    user_question = "Türkçe cevap ver: " + user_question
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

# Ana işlem fonksiyonu
def main():
    load_dotenv()
    st.set_page_config(page_title="Hayvan Hastalığı Chatbot", page_icon="🐱🐶🐮")

    conn = connect_db()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Konuları txt dosyasından yükle
    topics = load_topics('HASTALIKLAR.txt')

    st.header("Hayvan Hastalıkları Botu 🐱🐶🐮")
    user_question = st.text_input("Hastalık giriniz:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("PDF'leriniz")
        pdf_docs = st.file_uploader(
            "Dosyalarınızı buraya yükleyin", accept_multiple_files=True)

        # Konu seçimi
        selected_topic = st.selectbox("Hangi konu ile ilgili?", ["Konu seçiniz"] + topics)

        if st.button("PDF YÜKLE VE SOR"):
            with st.spinner("İşleniyor..."):
                # PDF'leri texte dönüştür
                raw_text = get_pdf_text(pdf_docs)
                st.write("PDF'den çıkarılan metin:")
                st.write(raw_text)
                
                # Textleri chunk'lara ayır
                text_chunks = get_text_chunks(raw_text)
                st.write("Metin parçalara ayrıldı:")
                st.write(text_chunks)

                # Vector database oluştur
                vectorstore = get_vectorstore(text_chunks)

                # Seçilen konuyu veritabanına kaydet veya ID'yi al
                topic_id = insert_topic(conn, selected_topic)

                # PDF'yi veritabanına kaydet
                document_id = insert_document(conn, topic_id, pdf_docs[0].read(), pdf_docs[0].name)

                # Chunk'ları ve vektörleri veritabanına kaydet
                for chunk in text_chunks:
                    vector = vectorstore.index.reconstruct(text_chunks.index(chunk))  # Her chunk için vektör alıyoruz
                    insert_chunk(conn, document_id, chunk, np.array(vector))  # Vektör verisi numpy array olarak kaydedilir

                # Konuşma zincirini oluştur
                st.session_state.conversation = get_conversion_chain(vectorstore)

                st.success(f"Seçilen konu: {selected_topic}")
                st.success(f"Yüklenen dosya: {pdf_docs[0].name}")
                st.success("PDF başarıyla kaydedildi ve işleme alındı.")

        if st.button("KONUYU SEÇ VE SOR"):
            with st.spinner("Veriler getiriliyor..."):
                chunks, vectors = get_chunks_by_topic(conn, selected_topic)
                if chunks and vectors:
                    vectorstore = get_vectorstore_from_db(chunks, vectors)
                    st.session_state.conversation = get_conversion_chain(vectorstore)
                    st.success(f"'{selected_topic}' konusu başarıyla yüklendi.")
                else:
                    st.error("Seçilen konu için veri bulunamadı.")

if __name__ == "__main__":
    main()
