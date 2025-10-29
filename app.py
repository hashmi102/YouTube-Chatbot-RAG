# app.py

import os
import subprocess
import glob
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#  Function to download subtitles 
def download_subtitles(video_url, output_file="subtitle.vtt"):
    try:
        subprocess.run([
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "-o", output_file,
            video_url
        ], check=True)

        # yt-dlp may append .en.vtt, find actual file
        downloaded_files = glob.glob("subtitle*.vtt*")
        if downloaded_files:
            return downloaded_files[0] 
        else:
            return None
    except Exception as e:
        print(e)
        return None

#  Function to convert .vtt to plain text 
def vtt_to_text(vtt_file):
    with open(vtt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text_lines = []
    for line in lines:
        line = line.strip()
        if "-->" in line or line.isdigit() or line == "":
            continue
        text_lines.append(line)
    
    return " ".join(text_lines)

#  Building a chain
def build_pipeline(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain

#      Streamlit App 
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥")
st.title("🎬 YouTube Video Q&A Chatbot")
st.markdown("Paste any YouTube link below and ask questions!")

video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    st.info("Fetching subtitles... Please wait.")

    sub_file = download_subtitles(video_url)
    if sub_file:
        transcript = vtt_to_text(sub_file)
        if transcript.strip() == "":
            st.error("❌ Subtitles fetched but empty. Try another video.")
        else:
            st.success("✅ Subtitles fetched successfully!")
            main_chain = build_pipeline(transcript)

            user_question = st.text_input("Ask a question about this video:")
            if user_question:
                with st.spinner("Thinking..."):
                    answer = main_chain.invoke(user_question)
                st.success("Answer:")
                st.write(answer)
    else:
        st.error("❌ Could not fetch subtitles for this video. Try another link.")
