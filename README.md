
# YouTube Chatbot RAG

A Streamlit app that allows users to ask questions about **any YouTube video** with English subtitles.  
Powered by **OpenAI**, **LangChain**, and **FAISS** for retrieval-augmented generation (RAG).

---
![image alt](https://github.com/hashmi102/YouTube-Chatbot-RAG/blob/main/demo_ss1.jpg?raw=true)

![image alt](https://github.com/hashmi102/YouTube-Chatbot-RAG/blob/main/demo_ss2.jpg?raw=true)

---
🚀 Features

✅Ask natural questions about YouTube videos by providing a URL.
✅ Automatically fetches video transcripts (manual or auto-generated).
✅ Implements Recursive Chunking for context preservation.
✅ Uses OpenAI Embeddings (text-embedding-3-small) + FAISS Vector Store for fast semantic search.
✅ Built with GPT-4o-mini for intelligent and contextual responses.
✅ Maintains a short chat history for multi-turn interactions.
✅ Fully interactive web UI via Streamlit.

---

## Installation

1️⃣ Clone the Repository (optional if running locally)**

```bash
git clone https://github.com/yourusername/YouTube-Chatbot-RAG.git
cd YouTube-Chatbot-RAG

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate       # On Windows

3️⃣ Install Required Libraries
pip install -r requirements.txt

4️⃣ Set Your OpenAI API Key

Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key_here

5️⃣ Run the App
streamlit run app.py


