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

API_URL = "http://127.0.0.1:8000"  # must have API running in terminal FIRST
# instructions: navigate to the API folder in terminal:   cd mock_api
# then: run the FastAPI application: uvicorn mock_api:app --reload

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
    st.session_state["topic_counts"] = defaultdict(int)

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
    docs = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.strip()
            if text:
                docs.append(Document(text=text))
        except Exception as e:
            print(f"Der Download ist fehlgeschlagen {url}: {e}")
    return docs

@st.cache_resource
def get_documents_from_urls(csv_path): # Loading the URLs; returns the parsed articles (the full process of extracting and processing article content).
    urls = load_article_urls(csv_path)
    return download_articles(urls)

# ----- PATHS & ENVIRONMENT SETUP -------------------------------------------------------------------
base_dir = os.path.dirname(__file__)

data_extract_path = os.path.join(base_dir, "data")
embeddings_extract_path = os.path.join(base_dir, "embeddings")
vector_index_extract_path = os.path.join(base_dir, "vector_index")

resources = {
    "data.zip": {
        "path": os.path.join(base_dir, "data.zip"),
        "extract_to": os.path.join(base_dir, "data"),
        "gdrive_id": "14AJXPCmjmmQpXaOlhwN4dHZlfBLseLiA"
    },
    "embeddings.zip": {
        "path": os.path.join(base_dir, "embeddings.zip"),
        "extract_to": os.path.join(base_dir, "embeddings"),
        "gdrive_id": "1psh_F3yCalADAuP2xUeS8Yzbmg6zru1E"
    },
    "vector_index.zip": {
        "path": os.path.join(base_dir, "vector_index.zip"),
        "extract_to": os.path.join(base_dir, "vector_index"),
        "gdrive_id": "1sKAAKuwE0-I_Hb2J5bgDu4Kacs8Hriiw"
    }
}

# ----- UTILITIES ------------------------------------------------------------------------------------
async def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    await asyncio.to_thread(gdown.download, url, dest_path, quiet=False)

def unzip_file(zip_path, extract_to_path, gdrive_id=None):
    if not os.path.exists(zip_path):
        print(f"{zip_path} not found, downloading from Google Drive...")
        if gdrive_id is None:
            raise ValueError(f"No Google Drive ID provided for {zip_path}")
        asyncio.run(download_from_gdrive(gdrive_id, zip_path))

    if not os.path.exists(extract_to_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Extracted {zip_path} to {extract_to_path}")
    else:
        print(f"Extracted folder already exists: {extract_to_path}, skipping extraction.")

# ----- PROCESS ALL RESOURCES ------------------------------------------------------------------------
for resource in resources.values():
    unzip_file(resource["path"], resource["extract_to"], resource["gdrive_id"])



# ----- SEARCH ENGINE SETUP ------------------------------------------------------------------------------------
secrets_path = os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

if os.path.exists(secrets_path):
    API_TOKEN = st.secrets.get("API_TOKEN")
else:
    load_dotenv(dotenv_path=dotenv_path, override=True)
    API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    msg = (
        "API token is missing in the .env file (Error Code: LOCAL-001)."
        if os.path.exists(dotenv_path)
        else "API token is missing, and .env file is not found (Error Code: LOCAL-002).")
    st.error(msg)
    st.stop()

headers = {"Authorization": f"Bearer {API_TOKEN}"}


# Initializing LLM
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(
    model_name=hf_model,
    task="text-generation",
    headers=headers
)

# Loading documents from CSV into the extracted data folder
csv_path = os.path.join(data_extract_path, "metadata.csv")
documents = get_documents_from_urls(csv_path)

# Embedding model
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=embeddings_extract_path
)

text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)

## First run - Create vector index
## Uncomment to run the first time (this creates the vector index and stores it in a zipped file)
vector_index = VectorStoreIndex.from_documents(
    documents,
    transformations=[text_splitter],
    embed_model=embeddings
)
vector_index.storage_context.persist(persist_dir=vector_index_extract_path)
## Save the vector index as a zip file for future use
vector_index_zip_path = resources["vector_index.zip"]["path"]
with zipfile.ZipFile(vector_index_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(vector_index_extract_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, start=vector_index_extract_path)
            zipf.write(file_path, arcname=arcname)
            
# Uncomment for subsequent runs (load vector index from storage)
#storage_context = StorageContext.from_defaults(persist_dir=vector_index_extract_path)
#vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

# ---------------CHATBOT SETUP ------------------------------------------------------------------------------------
retriever = vector_index.as_retriever(similarity_top_k=2)
memory = ChatMemoryBuffer.from_defaults()

prompts = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=("""
            
OBJECTIVE:
You are an outstanding customer service representative of ElternLeben.de, a platform that supports parents with content and online advice on parenting, baby and child development, nutrition and more.
Your goal is to understand users' needs and guide them to the right products or information on ElternLeben.de to solve their problem.


Procedure:
1. Ask about the user's challenge:
First ask about the user's current challenge (e.g. baby sleep, nutrition, parenting, screen time, etc.).
2. ALWAYS determine the support required:
Find out what kind of support the user is looking for:
-Does he need quick, personalized advice?
-Are they looking for general information or do they have specific questions?
-Do they want to delve deeper into a topic?

Understand the user's needs better before recommending something:
-Always ask at least one or two follow-up questions to make sure you understand the user properly before making recommendations.

3. Recommend suitable services based on needs:
a. Articles:
-ElternLeben.de has articles written by experts on many topics.
-Don't give general information on the topic, but on the specific problem.
-You only give tips that you know from articles on ElternLeben.de.
-If you give a tip or information, always link to the source (a page from ElternLeben.de).
-Only link to ElternLeben.de. Only link URLs that you know.
-If you give tips or link to an article, you must also always recommend a suitable service (e-mail advice, midwife consultation - for baby parents -, parenting consultations - if the topic is suitable - or on-demand courses and eBooks - if the topic is suitable).
-This additional recommendation also always has a link.


b. Email advice:
Offer this service if the user has specific questions: https://www.elternleben.de/ueber-stell-uns-deine-frage/

c. Parent consultation:
-Only recommend this for baby sleep, parenting, cleanliness education, kindergarten or screen time: https://www.elternleben.de/elternsprechstunde/


d. Midwife advice:
-Recommend this service for topics related to baby sleep or breastfeeding (only for children under 1 year): 
https://www.elternleben.de/hebammensprechstunde/


e. On-demand courses and eBooks:
-If the user wants to delve deeper into a topic or is looking for more comprehensive information, you can recommend paid webinars or eBooks from ElternLeben.de.
-You always link to the real URL of the course (or eBook) on ElternLeben.de.
-You know the URL or it is in the knowledge base (file metadata.csv).
-Only link to ElternLeben.de.
-Only link URLs that you know.



Important notes:

-Always recommend email advice at the end of the conversation in case further support is needed.

-Your style should be friendly and professional, with emojis to lighten things up, always on you.
-Keep replies short and readable for mobile devices.

FURTHER RULES:
-You do not enter into conversations on topics that do not fall within your area of responsibility or that of ElternLeben.de .
-For any user query, you should ALWAYS consult your source of knowledge, even if you think you already know the answer.
-Your answer MUST be based on the information provided by that knowledge source.
-If a user asks questions that go beyond the actual topic, you should not answer them.
-Instead, kindly redirect to a topic you can help with.
-Only respond in German.

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

            # Extract sources
            urls = set()
            for node in result.source_nodes:
                if node.metadata.get("url"):
                    urls.add(node.metadata["url"])

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
            webinars_json = get_webinars() 
            matching_webinars = filter_webinars_by_topic(webinars_json, topic)


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

