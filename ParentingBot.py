import streamlit as st
import os
from dotenv import load_dotenv 
import csv
from datetime import datetime
import pandas as pd
from newspaper import Article
import requests 
from rapidfuzz import fuzz  
from collections import defaultdict
import re
import zipfile
import gdown
import asyncio
from requests.exceptions import RequestException


from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import Document
from llama_index.readers.web import SimpleWebPageReader

# --- DEVELOPMENT ONLY: Clear Streamlit cache ---
# Uncomment the following lines only when you need to reset the cache.

#st.cache_resource.clear()
#st.cache_data.clear()

#st.warning("⚠️ Cache has been cleared. Remember to comment this out again.")

# -----------------------------------------------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000"  # Must have API running in terminal FIRST
# Instructions: Navigate to the API folder in terminal:   cd mock_api
# Then: Run the FastAPI application: uvicorn mock_api:app --reload

# ----- PAGE PRESENTATION ------------------------------------------------------------------------------------
st.title("ElternLeben Bot: Hilfe, wann immer du sie brauchst")
st.markdown("""
Frag mich alles über Erziehung.  Meine Antworten basieren auf Hunderten von Artikeln, die von unseren Fachleuten geschrieben wurden und auf der [Elternleben-Website](https://www.elternleben.de) verfügbar sind.
""")

st.markdown("**Beispielfragen:**")
st.markdown("- Mein Kind hört nicht auf zu weinen; wie kann ich ihm helfen?")
st.markdown("- Mein Baby hat am ganzen Körper rote Flecken, was soll ich tun?")

# ----- SIDEBAR ----------------------------------------------------------------------------------------------
with st.sidebar:
    st.image(os.path.join(os.path.dirname(__file__), "Images", "elternleben_holding_hands.jpg"), use_container_width=True)
    
    st.title("Hallo! Ich bin ElternLeben Bot")
    st.markdown("Hier zur Unterstützung auf dem Weg zum Elternsein und zum Leben als Eltern.")
    st.markdown("""
        <style>
            .stButton button {
                background-color: #80A331;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
            }
            .stButton button:hover {
                background-color: #0056b3;
            }
        </style>
    """, unsafe_allow_html=True)
    
# ----- SESSION STATES ----------------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "mentioned_topics" not in st.session_state:
    st.session_state["mentioned_topics"] = set()
    
if "topic_counts" not in st.session_state:
    st.session_state["topic_counts"] = {}

# ----- SIDEBAR CONTROLS -------------------------------------------------------------------------------
with st.sidebar:
    if st.button("🔄 Möchtest du das Gespräch neu beginnen?"):
        st.session_state["chat_history"] = []
        st.session_state["mentioned_topics"] = set()
        st.session_state["topic_counts"] = defaultdict(int)
        st.success("Der Gesprächsverlauf wurde gelöscht. Wir können neu beginnen 😊")
        st.experimental_rerun() 

    st.markdown("---")
    st.markdown("### Ressourcen:")

    st.markdown("""
        [An einem Live-Webinar teilnehmen](https://www.elternleben.de/elternsprechstunde/)
    """)

    st.markdown("""
        [Kontakt mit uns aufnehmen](https://www.elternleben.de/ueber-stell-uns-deine-frage/)
    """)


# ----- LOAD DATA (ARTICLES) FROM CSV------------------------------------------------------------------------------------
@st.cache_resource
def load_article_urls(csv_path): #  Loading and returning a list of URLs from the CSV file (just the URLs, no article processing).
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].dropna().tolist()  # Pulls from first column only, where the urls are listed.


def download_articles(urls):
    documents = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.strip()
            if text:
                doc = Document(text=text, metadata={"url":url})
                documents.append(doc)
                print(f"Document created for URL: {url}, Metadata: {doc.metadata}")  # Debug log to confirm metadata
        except Exception as e:
            print(f"Failed to download article from {url}: {e}")
            continue
    return documents

@st.cache_resource(show_spinner=False)
def get_documents_with_metadata(csv_path):
    urls = load_article_urls(csv_path)
    return download_articles(urls)
    



# ----- TOGGLE: Set to True for first run; False for subsequent runs -------
IS_FIRST_RUN = False

# ----- PATHS & ENVIRONMENT SETUP ------------------------------------------------------------------

base_dir = os.path.dirname(__file__)

data_extract_path = os.path.join(base_dir, "data")
embeddings_extract_path = os.path.join(base_dir, "embeddings")
vector_index_extract_path = os.path.join(base_dir, "vector_index")

resources = {
    "data.zip": {
        "path": os.path.join(base_dir, "data.zip"),
        "extract_to": data_extract_path,
        "gdrive_id": "14AJXPCmjmmQpXaOlhwN4dHZlfBLseLiA"
    },
    "embeddings.zip": {
        "path": os.path.join(base_dir, "embeddings.zip"),
        "extract_to": embeddings_extract_path,
        "gdrive_id": "1WOfzzc9zT5y3_zp70SQhkRVFOEJvOXxG"
    },
    "vector_index.zip": {
        "path": os.path.join(base_dir, "vector_index.zip"),
        "extract_to": vector_index_extract_path,
        "gdrive_id": "1mziUxOAKJ_UehlAQEDBfgT1UbORYDbk-"
    }
}


# ----- UTILITIES ------------------------------------------------------------------------------------
def remove_directory(directory_path):
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(directory_path)

async def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    await asyncio.to_thread(gdown.download, url, dest_path, quiet=False)

def folder_exists_and_up_to_date(folder_path, zip_path):
    if os.path.exists(folder_path):
        if not any(os.scandir(folder_path)):
            print(f"{folder_path} is empty, extracting...")
            return False
        folder_time = os.path.getmtime(folder_path)
        zip_time = os.path.getmtime(zip_path)
        return folder_time >= zip_time
    return False

def unzip_file(zip_path, extract_to_path, gdrive_id=None):
    if not os.path.exists(zip_path):
        print(f"{zip_path} not found, downloading...")
        if gdrive_id is None:
            raise ValueError(f"No GDrive ID provided for {zip_path}")
        asyncio.run(download_from_gdrive(gdrive_id, zip_path))

    if folder_exists_and_up_to_date(extract_to_path, zip_path):
        print(f"{extract_to_path} is up to date.")
        return

    if os.path.exists(extract_to_path):
        print(f"Removing old {extract_to_path}")
        remove_directory(extract_to_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f"Extracted {zip_path}.")


# ----- SEARCH ENGINE SETUP ------------------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"), override=True)
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    st.error("Missing API token.")
    st.stop()

headers = {"Authorization": f"Bearer {API_TOKEN}"}


# Initializing LLM
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(
    model_name=hf_model,
    task="text-generation",
    headers=headers,
    generation_kwargs={
        "temperature": 0.3,           # Controls randomness
        "max_new_tokens": 300,        # Limits response length
        "top_p": 0.85,                 # Sampling: Limits the number of tokens considered during sampling
        "repetition_penalty": 1.2,   # Reduces repetitive outputs
        "do_sample": True             # Enables sampling 
    }
)

# Embedding model
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=embeddings_extract_path
)
print("Embedding model loaded:", embeddings)


text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)
print("Splitter settings:", text_splitter.chunk_size)

# ----- DATA PREP -----------------------------------------------------------------------
# Loading documents from CSV into the extracted data folder
csv_path = os.path.join(data_extract_path, "metadata.csv")
documents = get_documents_with_metadata(csv_path)

print(f"Loaded {len(documents)} documents")



# ----- FIRST RUN (generate and save) vs SUBSEQUENT RUNS (load from storage) -------------
if IS_FIRST_RUN:
    print("🔄 First run: creating and saving index/embeddings...")
    print(f"Document count going into index: {len(documents)}")

    # --- DIAGNOSTIC: Check documents ---
    if len(documents) == 0:
        print("⚠️ No documents to index. Check your CSV or download process.")
    else:
        print("✅ Documents successfully loaded.")

    # --- DIAGNOSTIC: Folder check & write test ---
    if os.path.exists(vector_index_extract_path):
        print("📁 vector_index folder exists.")
        test_file_path = os.path.join(vector_index_extract_path, "test_write.txt")
        try:
            with open(test_file_path, "w") as f:
                f.write("test")
            print("✅ Write permissions OK.")
            os.remove(test_file_path)
        except Exception as e:
            print(f"❌ Cannot write to vector_index folder: {e}")
    else:
        print("📂 vector_index folder does NOT exist. Will be created by persist().")

    # --- Optional: remove old folder to ensure clean write ---
    if os.path.exists(vector_index_extract_path):
        print("🧹 Removing existing vector_index folder to force fresh persist.")
        remove_directory(vector_index_extract_path)

    # --- Create and persist index ---
    try:
        print("⚙️ Creating vector index from documents...")
        vector_index = VectorStoreIndex.from_documents(
            documents,
            transformations=[text_splitter],
            embed_model=embeddings
        )
        print("💾 Persisting vector index to disk...")
        vector_index.storage_context.persist(persist_dir=vector_index_extract_path)
        print("✅ Vector index persisted successfully.")
    except Exception as e:
        print(f"❌ Failed during vector index creation or persist: {e}")

    st.stop() # Stop after first run to avoid executing rest of the app

else:
    print("⬇️ Subsequent run: downloading and loading saved index...")
    for resource in resources.values():
        unzip_file(resource["path"], resource["extract_to"], resource["gdrive_id"])

    storage_context = StorageContext.from_defaults(persist_dir=vector_index_extract_path)
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
    print("✅ Vector index loaded successfully.")
        

# ---------------CHATBOT SETUP ------------------------------------------------------------------------------------
retriever = vector_index.as_retriever(similarity_top_k=2)
memory = ChatMemoryBuffer.from_defaults()

prompts = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=("""
            
You are a helpful and professional parenting assistant for ElternLeben.de — a trusted platform offering expert articles, webinars, consultations, and email advice on parenting, child development, baby care, nutrition, and more.

OBJECTIVE:  
Your goal is to understand users’ parenting concerns and guide them to helpful resources and services **only from ElternLeben.de**.

INSTRUCTIONS:

1. **Identify the user's concern**  
Ask the user to briefly describe what they need help with (e.g., baby sleep, toddler behavior, screen time, nutrition, etc.).

2. **Clarify what kind of support they’re looking for**  
Ask follow-up questions to determine whether they need:
- Quick, personal advice
- General or specific information
- A deep dive into a topic (e.g., via a course or ebook)

Always ask at least one or two follow-up questions before making recommendations.

3. **Provide recommendations based on retrieved content**  
Use only information from the retrieved knowledge base (ElternLeben.de articles or resources).  
Do not answer based on general knowledge or guesswork.

If an article answers their concern:
- Provide a **concise summary** of the solution.
- Always include the **exact source link** from ElternLeben.de.
- Add a relevant **service recommendation** (see below).

SERVICE OPTIONS:

- **Articles**: Expert-written articles. Only give tips based on actual content. Always cite the URL.

- **Email Advice** (for specific questions):  
  [Ask a question](https://www.elternleben.de/ueber-stell-uns-deine-frage/)

- **Parent Consultation** (for baby sleep, parenting, potty training, kindergarten, screen time):  
  [Parent consultation](https://www.elternleben.de/elternsprechstunde/)

- **Webinars**:  
  Recommend these if the user wants to explore a topic in depth. Only suggest known URLs from the knowledge base (e.g., metadata file).

GUIDELINES:

- Always recommend **Email Advice** at the end of the chat for ongoing questions.
- Link **only to ElternLeben.de**.
- Never fabricate links or give advice outside your scope.
- If a question falls outside your domain, politely explain and redirect the user to an appropriate topic.
- Keep your tone friendly, supportive, and professional — with light emojis 😊 where appropriate.
- Keep answers **short, mobile-friendly, and focused**.
- Respond **only in German** unless told otherwise.
- All answers must be grounded in the retrieved knowledge (RAG context).


        """)
    )
]


# ----- Display chat history -----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----- DEFINING KEYWORDS AND TOPICS ----------------------------------------------------------------------------------
# Defining topic keywords for matching user input
topic_keywords = {
    "Wutausbrüch": ["wut", "treten", "schlagen", "wutausbrüch"],
    "Schwangerschaft": ["geburtsvorbereitung", "rückbildung", "schwangerschaft"],
    "Ernährung": ["essen", "trinken", "ernährung"],
    "Elternzeit": ["elternzeit", "elternkarenz"],
    "Schlafstörung": ["schlaf", "schlafstörung"],
    "Sprachentwicklung": ["sprachen", "sprachentwicklung"],
    "Geschwister": ["geschwister"]
}


# ----- HELPER FUNCTIONS ----------------------------------------------------------------------------------
def get_webinars():
    response = requests.get(f"{API_URL}/webinars")
    if response.status_code == 200:
        webinars_json = response.json()  
        return webinars_json
    else:
        st.error("Failed to fetch webinars.")
        return []  

def filter_webinars_by_topic(webinars_json, topic, threshold=70):
    return [w for w in webinars_json
            if fuzz.partial_ratio(topic.lower(), w["agenda"].lower()) >= threshold]

def get_consultations():
    # Getting experts
    response = requests.get(f"{API_URL}/experts/")
    experts = response.json()
    experts_list = [(item['name'], item['uuid']) for item in experts]

    # Formatting date-times and extracting date
    def format_slot(slot):
        start = datetime.fromisoformat(slot['start_datetime'].replace("Z", "+00:00"))
        end = datetime.fromisoformat(slot['end_datetime'].replace("Z", "+00:00")) 
        date_str = start.strftime("%A, %B %d, %Y")
        time_range = f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')}"
        
        return start, f"{date_str} — {time_range}"  # return datetime and formatted string

    # Listing experts by first availabilities
    expert_slots = {}
    
    for name, expert_id in experts_list:
        slots_response = requests.get(f"{API_URL}/experts/{expert_id}/available-slots")
        
        if slots_response.status_code == 200:
            slots = slots_response.json()
            if slots:
                # Sorting slots by the start datetime
                sorted_slots = sorted(slots, key=lambda slot: slot['start_datetime'])
                sorted_slots_all = format_slot(sorted_slots[0])[1]
                expert_slots[name] = sorted_slots_all
            else:
                expert_slots[name] = "Keine freien Termine."
        else:
            expert_slots[name] = "Momentan sind keine Termine verfügbar."
            
# Formating and spacing
    available_consultations = "" 
    for expert, slot in expert_slots.items():
        available_consultations += f"\n{expert}:\n   - {slot}\n"  # Adding a newline after each expert's details for spacing

    return available_consultations 

# Extracting relevant topics from user input based on keywords using fuzzy
def extract_topics(user_input, threshold=70):
    input_lower = user_input.lower()
    user_words = re.split(r'\W+', input_lower)  
    user_words = [word for word in user_words if word]  # Remove empty strings

    extracted_topics = []

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for word in user_words:
                if fuzz.partial_ratio(keyword_lower, word) >= threshold:
                    extracted_topics.append(topic)
                    break  
            else:
                continue
            break  # Break out once a match is found

    return extracted_topics


def fuzzy_keyword_match(text, keywords):
    text = text.lower()
    for keyword in keywords:
        if keyword.lower() in text:
            return True
    return False


# ----- HANDLING USER INPUT ----------------------------------------------------------------------------------
if user_input := st.chat_input("Womit kann ich heute helfen"):
    st.chat_message("human", avatar=os.path.join(os.path.dirname(__file__), "Images", "parent.jpg")).markdown(user_input)
    st.session_state.chat_history.append({"role": "human", "content": user_input})

#----- PRIORITY: Referring to emergency services
    emergency_keywords = ["dringend", "Notfall", "verzweifelt"]
    if fuzzy_keyword_match(user_input, emergency_keywords):
        emergency_message = "„Es scheint sich um einen Notfall zu handeln. Bitte wende dich sofort an den Notdienst!"
        st.write(emergency_message)
        st.stop()

# ----- Initializing bot -----
    @st.cache_resource
    def init_bot():
        return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)

    rag_bot = init_bot()


# ------- Bot answers using RAG ----------
    with st.spinner("📂 Suche nach einer Antwort in den Elternartikeln..."):
        try:
            result = rag_bot.chat(user_input)
            answer = result.response

  # 🔎 Inspect and extract URLs in one loop
            urls = set()
            for node in result.source_nodes:
                print("Source node metadata:", node.metadata)
                url = node.metadata.get("url")
                print("URL:", url)
                if url:
                    urls.add(url)

        # Append URLs to the answer
            if urls:
                answer += "\n\n**Sources:**\n" + "\n".join(f"- [{url}]({url})" for url in urls)
        except Exception as e:
            answer = f"Entschuldigung, ich hatte Probleme bei der Bearbeitung Ihrer Frage: {e}"

# -------Show main RAG answer ----------
    with st.chat_message("assistant", avatar=os.path.join(os.path.dirname(__file__), "Images", "helping_hands.jpg")):
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ---- Extract topics ----
    matched_topics = extract_topics(user_input)
    st.session_state.mentioned_topics.update(matched_topics)

# ---- Count topics and offer webinars ----
    webinar_suggestions = []
    for topic in matched_topics:
        st.session_state.topic_counts[topic] += 1
        count = st.session_state.topic_counts[topic]

        if count == 3:
            st.session_state.topic_counts[topic] = st.session_state.topic_counts.get(topic, 0) + 1
            count = st.session_state.topic_counts[topic]

            if matching_webinars:
                webinars_text = "\n\n".join(
                    f"- [**{webinar['topic']}**]({webinar['join_url']})  \n  📅 Datum: {webinar['start_time']}  \n  ⏰ Duration: {webinar['duration']}"
                    for webinar in matching_webinars
                )
                webinar_response = (
                    f"Ich habe deine Interesse an **{topic}** bemerkt, da du das Thema einige Male erwähnt hast.  \n"
                    f"Vielleicht bist du an einem Webinar interessierst, das dieses Thema vertieft. Hier sind einige bevorstehende Webinare, die ich sehr empfehlen kann, da sie von unseren Experten präsentiert werden:\n\n"
                    f"{webinars_text}"
                )
            else:
                webinar_response = (
                    f"Ich habe dein Interesse an **{topic}** festgestellt und möchte dir einige mögliche Webinare empfehlen, die von unseren Experten geleitet werden. \n"
                    f'Leider konnte ich in nächster Zeit nichts Relevantes finden. Siehe unsere Liste der Webinare an, indem du auf die Schaltfläche unter „Ressourcen“ auf der linken Seite klickst.'

                )        
            st.chat_message("assistant").markdown(webinar_response) #show webinar suggestions
 
    # Referring to consultations
    consultation_keywords = ["Beratung", "Termin"]
    if fuzzy_keyword_match(user_input, consultation_keywords):
        consultation_message = "„Ich verstehe, dass du an einer Beratung interessiert bist.  Unten findest du unsere verfügbaren Experten und Termine.  Bitte kontaktiere uns, um einen Termin zu vereinbaren:"
        st.write(consultation_message) 
        available_consultations = get_consultations()
        st.write(available_consultations)

        
# ----- FEEDBACK COLLECTION --------------------------------------------------------------------
def log_feedback(feedback_data):
    folder_path = os.path.join(os.path.dirname(__file__), "ChatBot_Feedback")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "chat_feedback_log.csv")

    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "question", "answer", "feedback"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(feedback_data)

def log_chat_history():
    if len(st.session_state.chat_history) > 1:
        user_question = st.session_state.chat_history[-2]["content"]
        last_message = st.session_state.chat_history[-1] 
        assistant_answer = last_message["content"] if len(st.session_state.chat_history) > 0 else ""
    else:
        user_question = ""
        assistant_answer = ""
    
    # Proceed with logging if both user question and assistant answer are available
    if user_question and assistant_answer:
        chat_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": user_question,
            "answer": assistant_answer,
            "feedback": "N/A"  # Placeholder for feedback until it's given
        }
        log_feedback(chat_data)


# ----- Collecting feedback -----
if st.session_state.chat_history:
    if len(st.session_state.chat_history) > 1:
        last_message = st.session_state.chat_history[-1]
        if last_message["role"] == "assistant":
            st.session_state.feedback_submitted = False 

            feedback_key = f"feedback_radio_{len(st.session_state.chat_history)}"
            st.markdown("### War die Antwort hilfreich??")
            feedback = st.radio(" ", ("👍 Ja", "👎 Nein"), index=None, key=feedback_key)

            if feedback:
                st.write("Thank you for your feedback!")
                feedback_mapping = {
                    "👍 Ja": "Ja",
                    "👎 Nein": "Nein"
                }

                # Logging feedback to CSV
                user_question = st.session_state.chat_history[-2]["content"]
                assistant_answer = last_message["content"]
                feedback_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": user_question,
                    "answer": assistant_answer,
                    "feedback": feedback_mapping.get(feedback, feedback),
                }

                log_feedback(feedback_data)
                st.session_state.feedback_submitted = True 

                del st.session_state[feedback_key]

                # Optionally, store webinar suggestions here to preserve them
                if "webinar_suggestions" in st.session_state:
                    st.session_state.webinar_suggestions = webinar_response

log_chat_history()

