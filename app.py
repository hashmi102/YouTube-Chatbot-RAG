import os
import subprocess
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== Load API Key ==========
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ========== Subtitle Download ==========
def download_subtitles(video_url, output_file="subtitle.vtt"):
    """Downloads YouTube subtitles using yt-dlp"""
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

        downloaded_files = glob.glob("subtitle*.vtt*")
        return downloaded_files[0] if downloaded_files else None
    except Exception as e:
        print("Download error:", e)
        return None

# ========== Convert VTT to Text ==========
def vtt_to_text(vtt_file):
    """Converts .vtt subtitle file to clean text"""
    with open(vtt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text_lines = []
    for line in lines:
        line = line.strip()
        if "-->" in line or line.isdigit() or not line:
            continue
        text_lines.append(line)
    
    return " ".join(text_lines)

# ========== Build RAG Pipeline ==========
def build_pipeline(transcript):
    """Creates embeddings, vectorstore, retriever, and LLM chain"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    prompt = PromptTemplate(
        template="""
        You are a smart assistant that helps users understand YouTube videos.
        Use the transcript context to answer if it is relevant.
        If the context doesn’t contain a clear answer, use your general reasoning to help.
        Never say “I don’t know”; try to respond naturally and informatively.

        Transcript Context:
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

# ========== Cache Processing ==========
@st.cache_resource(show_spinner=False)
def process_video(video_url):
    """Fetch subtitles, process transcript, and build RAG pipeline (cached)"""
    sub_file = download_subtitles(video_url)
    if not sub_file:
        return None, "❌ Could not fetch subtitles for this video."
    
    transcript = vtt_to_text(sub_file)
    if transcript.strip() == "":
        return None, "❌ Subtitles fetched but empty."
    
    main_chain = build_pipeline(transcript)
    return main_chain, "✅ Subtitles fetched and processed successfully!"

# ========== Streamlit App ==========
st.set_page_config(page_title="🎬 YouTube Chatbot", page_icon="🤖")
st.title("🎬 YouTube Video Q&A Chatbot")
st.markdown("Ask smart questions about any YouTube video with subtitles!")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_video" not in st.session_state:
    st.session_state.current_video = None

# Input YouTube Link
video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    # Only process if it's a new video
    if video_url != st.session_state.current_video:
        with st.spinner("Fetching and processing subtitles..."):
            main_chain, message = process_video(video_url)

        if main_chain:
            st.session_state.main_chain = main_chain
            st.session_state.current_video = video_url
            st.session_state.chat_history = []  # reset for new video
            st.success(message)
        else:
            st.error(message)
    else:
        st.info("✅ Using cached subtitles and embeddings.")

    # User Question
    if "main_chain" in st.session_state and st.session_state.main_chain:
        user_question = st.text_input("Ask a question about this video:")
        if user_question:
            with st.spinner("Thinking..."):
                answer = st.session_state.main_chain.invoke(user_question)

            # Store in chat history
            st.session_state.chat_history.append({"question": user_question, "answer": answer})

            # Limit to last 5 messages
            if len(st.session_state.chat_history) > 5:
                st.session_state.chat_history = st.session_state.chat_history[-5:]

#  Display Chat History
if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")
        st.markdown("---")
