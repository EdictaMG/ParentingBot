import streamlit as st
import os
from dotenv import load_dotenv #to get secret API_TOKEN from .env file
import csv
from datetime import datetime
import pandas as pd
from newspaper import Article
import requests #for booking APIs
from rapidfuzz import fuzz  #used for approximate keyword matching (wording needs not be identical)
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

API_URL = "http://127.0.0.1:8000"  # must have API running in terminal first

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

    if st.button("🔄 Möchtest du das Gespräch neu beginnen?"):
        st.session_state["chat_history"] = []
        st.session_state["mentioned_topics"] = set()
        st.session_state["topic_counts"] = defaultdict(int)
        st.success("Der Gesprächsverlauf wurde gelöscht. Wir können neu beginnen 😊")

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
    return df.iloc[:, 0].dropna().tolist()  # Pulls from first column only, where the urls are listed


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
def get_documents_from_urls(csv_path): # Loading the URLs and downloads/returns the parsed articles (the full process of extracting and processing article content).
    urls = load_article_urls(csv_path)
    return download_articles(urls)

# ----- PATHS & ENVIRONMENT SETUP -------------------------------------------------------------------
# Define paths
base_dir = os.path.dirname(__file__)

# Paths to extract contents
data_extract_path = os.path.join(base_dir, "data")
embeddings_extract_path = os.path.join(base_dir, "embeddings")
vector_index_extract_path = os.path.join(base_dir, "vector_index")

# Define zipped resources and their corresponding Google Drive file IDs
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

# Asynchronous function to download from Google Drive
async def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    await asyncio.to_thread(gdown.download, url, dest_path, quiet=False)

# Function to unzip files if they aren't already extracted
def unzip_file(zip_path, extract_to_path, gdrive_id=None):
    if not os.path.exists(zip_path):
        print(f"{zip_path} not found, downloading from Google Drive...")
        if gdrive_id is None:
            raise ValueError(f"No Google Drive ID provided for {zip_path}")
        # Running the asynchronous download
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

# Check if running on Streamlit Cloud
if "API_TOKEN" in st.secrets:
    # When deployed on Streamlit Cloud, use API_TOKEN from secrets.toml
    API_TOKEN = st.secrets["API_TOKEN"]
else:
    # When running locally, use .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
        API_TOKEN = os.getenv("API_TOKEN")

# Check if API_TOKEN is missing
if not API_TOKEN:
    # Specific error handling based on environment
    if "API_TOKEN" not in st.secrets:
        if os.path.exists(dotenv_path):
            st.error("API token is missing in the .env file (Error Code: LOCAL-001).")
        else:
            st.error("API token is missing, and .env file is not found (Error Code: LOCAL-002).")
    else:
        st.error(f"API token is missing in secrets.toml (Error Code: CLOUD-001).")
else:
    st.success("API token loaded successfully.")

# Initialize LLM
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(
    model_name=hf_model,
    task="text-generation",
    api_key=API_TOKEN)

# Load documents from CSV in the extracted data folder
csv_path = os.path.join(data_extract_path, "metadata.csv")
documents = get_documents_from_urls(csv_path)

# Embedding model and initilaizing it
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=embeddings_extract_path
)

text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)

## First run - Create vector index
# Uncomment to run the first time (this creates the vector index and stores it in a zipped file)
# vector_index = VectorStoreIndex.from_documents(
    #documents,
    #transformations=[text_splitter],
    #embed_model=embeddings
#)
# vector_index.storage_context.persist(persist_dir=vector_index_extract_path)
# Save the vector index as a zip file for future use
# with zipfile.ZipFile(vector_index_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#     for root, dirs, files in os.walk(vector_index_extract_path):
#         for file in files:
#             zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), vector_index_extract_path))

# Uncomment for subsequent runs (load vector index from storage)
# Initialize storage context to load vector index
storage_context = StorageContext.from_defaults(persist_dir=vector_index_extract_path)
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

# ---------------CHATBOT SETUP ------------------------------------------------------------------------------------
retriever = vector_index.as_retriever(similarity_top_k=2)
memory = ChatMemoryBuffer.from_defaults()

prompts = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=("""
            
#ZIEL
Du bist ein empathischer und kompetenter Kundenservice-Mitarbeiter von **ElternLeben.de**.  
Deine Aufgabe ist es, die **aktuelle Herausforderung** des Nutzers zu verstehen und ihn gezielt zu **Inhalten oder Angeboten** von ElternLeben.de zu führen – mit dem Ziel, ihm **konkret weiterzuhelfen**.

---

# VORGEHEN – SCHRITT FÜR SCHRITT
## 1. Herausforderung verstehen
- Beginne mit einer offenen Frage wie:
  > *„Was ist gerade eure größte Herausforderung – Babyschlaf, Ernährung, Erziehung, Bildschirmzeit ...?“*
- Höre aktiv zu und erfasse das Thema genau.

## 2. Art der Unterstützung ermitteln
- Frage gezielt nach:
  - Möchtest du schnelle Hilfe oder persönliche Beratung?
  - Suchst du allgemeine Infos oder willst du tiefer eintauchen?
  - Gibt es konkrete Fragen?

## 3. Immer nachfragen
- Stelle **mindestens 1–2 Rückfragen**, bevor du etwas empfiehlst.  
  > Ziel: den Bedarf besser verstehen.

## 4. Empfehlungen geben – IMMER mit Link
Nutze ausschließlich Inhalte von **ElternLeben.de**. 

### Empfehlungen können sein:
- **Artikel** (nur spezifisch relevante Inhalte, keine allgemeinen Tipps)
- **On-Demand-Kurse oder eBooks** (wenn Nutzer vertiefen möchte)
- **Beratungsangebote**:
  - **E-Mail-Beratung** bei konkreten Fragen  
    `https://www.elternleben.de/ueber-stell-uns-deine-frage/?pilot=#####`
  - **Elternsprechstunde (Zoom)** bei Themen wie Babyschlaf, Erziehung, Bildschirmzeit  
    `https://www.elternleben.de/elternsprechstunde/?pilot=#####`
  - **Hebammensprechstunde** bei Babys < 1 Jahr (Stillen, Babyschlaf)  
    `https://www.elternleben.de/hebammensprechstunde/?pilot=#####`

### Wichtig:
- Verwende **immer echte URLs**, die du kennst oder aus der CSV-Datei (`shop_1.csv`) stammen.
- Hänge **immer** einen Tracking-Code mit `?pilot=` und **5 zufälligen Ziffern** an.
- **Kombiniere immer Inhalte + passende Leistung** (z. B. Artikel + Kurs oder Beratung).

---

# TON & STIL
- Du-Ansprache, freundlich, empathisch, professionell
- Kurz, mobilfreundlich, gut lesbar
- Emojis zur Auflockerung (sparsam einsetzen)

**Beispiel-Antwort:**
> „Verstehe ich total – Babyschlaf kann echt herausfordernd sein 😴  
Was genau ist gerade das Schwierigste bei euch – Einschlafen oder Durchschlafen?  
Ich hab da direkt was von unseren Schlafexpert*innen 👉 [Artikel-Link/?pilot=12345]  
Und wenn du tiefer einsteigen willst, passt unser On-Demand-Kurs super dazu 👉 [Kurs-Link/?pilot=67890]  
Oder du stellst direkt eine Frage – unsere Berater*innen helfen per Mail 👉 [Mail-Beratung-Link]“

---

# REGELN & EINSCHRÄNKUNGEN
- Verwende **nur Inhalte und Angebote von ElternLeben.de**
- Alle Tipps und Infos sollen aus der Wissensdatenbank stammen
- **IMMER Wissensquelle konsultieren** (Artikel oder CSV), auch wenn du die Antwort zu wissen glaubst
- Wenn ein Thema nicht abgedeckt wird:
  - Höflich ablehnen
  - Thema umleiten auf etwas Passendes
- IMMER passende Leistung zusätzlich empfehlen – mit Link:
  - **E-Mail-Beratung** → bei konkreten Fragen  
  - **Hebammensprechstunde** → Babys unter 1 Jahr (Stillen, Schlaf)  
  - **Elternsprechstunde** → Erziehung, Bildschirmzeit etc.  
  - **On-Demand-Kurse/eBooks** → bei Wunsch nach tiefergehendem Wissen

---

# STIMMUNG ERKENNEN & EMPATHISCH REAGIEREN
- Analysiere die Stimmung des Nutzers:
  - z. B. besorgt, überfordert, neugierig, frustriert
- Reagiere empathisch, z. B.:
  > *„Klingt ganz schön herausfordernd – ich helf dir gern! 💛“*

        """)
    )
]


# ----- Display chat history -----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# ----- INITIALIZING ADDITIONAL SESSION STATES ----------------------------------------------------------------------------------

if "topic_counts" not in st.session_state:
    st.session_state.topic_counts = defaultdict(int)
#alternatively code replacing the two lines above (shorter and cleaner):
#st.session_state.setdefault("topic_counts", defaultdict(int))
    
if "referral_counts" not in st.session_state:
    st.session_state.referral_counts = {} #dictionnary as would be counting potentially diff topics (i.e. emergency or consultations)   
#alternatively code:
#st.session_state.setdefault("referral_counts", {})
    
    
if "mentioned_topics" not in st.session_state:
    st.session_state.mentioned_topics = set()
#alternative code:
#st.session_state.setdefault("mentioned_topics", set())

# ----- DEFINING KEYWORDS AND TOPICS ----------------------------------------------------------------------------------

# defining topic keywords for matching user input
topic_keywords = {
    "Wutausbrüch": ["wut", "treten", "schlagen", "wutausbrüch"],
    "Schwangerschaft": ["geburtsvorbereitung", "rückbildung", "schwangerschaft"],
    "Ernährung": ["essen", "trinken", "ernährung"],
    "Elternzeit": ["elternzeit", "elternkarenz"],
    "Schlafstörung": ["schlaf", "schlafstörung"],
    "Sprachentwicklung": ["sprachen", "sprachentwicklung"],
    "Geschwister": ["geschwister"]
}


# ----- TASK FUNCTIONS ----------------------------------------------------------------------------------

def get_webinars():
    response = requests.get(f"{API_URL}/webinars")
    if response.status_code == 200:
        webinars_json = response.json()  # Store the JSON response containing webinar details
        return webinars_json
    else:
        st.error("Failed to fetch webinars.")
        return []  # Return an empty list if the request failed

def filter_webinars_by_topic(webinars_json, topic, threshold=70):
    return [w for w in webinars_json
            if fuzz.partial_ratio(topic.lower(), w["agenda"].lower()) >= threshold]

def get_consultations():
    # getting experts
    response = requests.get(f"{API_URL}/experts/")
    experts = response.json()
    experts_list = [(item['name'], item['uuid']) for item in experts]

    # formatting date-times and extracting date
    def format_slot(slot):
        start = datetime.fromisoformat(slot['start_datetime'].replace("Z", "+00:00"))
        end = datetime.fromisoformat(slot['end_datetime'].replace("Z", "+00:00")) 
        date_str = start.strftime("%A, %B %d, %Y")
        time_range = f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')}"
        
        return start, f"{date_str} — {time_range}"  # return datetime and formatted string

    # listing experts by first availabilities
    expert_slots = {}
    
    for name, expert_id in experts_list:
        slots_response = requests.get(f"{API_URL}/experts/{expert_id}/available-slots")
        
        if slots_response.status_code == 200:
            slots = slots_response.json()
            if slots:
                # sorting slots by the start datetime
                sorted_slots = sorted(slots, key=lambda slot: slot['start_datetime'])
                sorted_slots_all = format_slot(sorted_slots[0])[1]
                expert_slots[name] = sorted_slots_all
            else:
                expert_slots[name] = "Keine freien Termine."
        else:
            expert_slots[name] = "Momentan sind keine Termine verfügbar."
            
# formating and spacing
    available_consultations = "" #"Expert Availability:\n"
    for expert, slot in expert_slots.items():
        available_consultations += f"\n{expert}:\n   - {slot}\n"  # adding a newline after each expert's details for spacing

    return available_consultations #comment out if using print

    #print(available_consultations) #printing within keeps output formated

# simple: extracting relevant topics from user input based on keywords
#def extract_topics(user_input, threshold=70):  
    #input_lower = user_input.lower()
    #extracted_topics = []

    #for topic, keywords in topic_keywords.items():
        #if any(keyword in input_lower for keyword in keywords):
            #extracted_topics.append(topic)

    #return extracted_topics

# better: extracting relevant topics from user input based on keywords USING fuzzy from fuzzywuzzy
def extract_topics(user_input, threshold=70):
    input_lower = user_input.lower()
    user_words = re.split(r'\W+', input_lower)  # split into individual words
    user_words = [word for word in user_words if word]  # remove empty strings

    extracted_topics = []

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for word in user_words:
                if fuzz.partial_ratio(keyword_lower, word) >= threshold:
                    extracted_topics.append(topic)
                    break  # No need to check more keywords for this topic
            else:
                continue
            break  # Break out once a match is found

    return extracted_topics


def fuzzy_keyword_match(text, keywords):
    # Example implementation with case insensitivity
    text = text.lower()
    for keyword in keywords:
        if keyword.lower() in text:
            return True
    return False


# ----- HANDLING USER INPUT ----------------------------------------------------------------------------------
#-------- user input -------------
if user_input := st.chat_input("Womit kann ich heute helfen"):
    st.chat_message("human", avatar=os.path.join(os.path.dirname(__file__), "Images", "parent.jpg")).markdown(user_input)
    st.session_state.chat_history.append({"role": "human", "content": user_input})

#--refering to emergency services (fuzzy matching)
    emergency_keywords = ["dringend", "Notfall", "verzweifelt"]
    if fuzzy_keyword_match(user_input, emergency_keywords):
        emergency_message = "„Es scheint sich um einen Notfall zu handeln. Bitte wende dich sofort an den Notdienst!"
        st.write(emergency_message)
        st.stop()

# ----- Initialize bot -----
    @st.cache_resource
    def init_bot():
        return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)

    rag_bot = init_bot()


# ------- chatbot answers using RAG ----------
    with st.spinner("📂 Suche nach einer Antwort in den Elternartikeln..."):
        try:
            result = rag_bot.chat(user_input)
            answer = result.response

            # extract sources
            urls = set()
            for node in result.source_nodes:
                if node.metadata.get("url"):
                    urls.add(node.metadata["url"])

            # append URLs to the answer
            if urls:
                answer += "\n\n**Sources:**\n" + "\n".join(f"- [{url}]({url})" for url in urls)

        except Exception as e:
            answer = f"Entschuldigung, ich hatte Probleme bei der Bearbeitung Ihrer Frage: {e}"

# ------- show main RAG answer ----------
    with st.chat_message("assistant", avatar=os.path.join(os.path.dirname(__file__), "Images", "helping_hands.jpg")):
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ---- extract topics ----
    matched_topics = extract_topics(user_input)
#    if "mentioned_topics" not in st.session_state:
#        st.session_state.mentioned_topics = set()
    st.session_state.mentioned_topics.update(matched_topics)

# ---- Debugging output ----*****************************
    #st.write("Mentioned topics:", st.session_state.mentioned_topics)

# ---- count topics and offer webinars ----
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
 
     # Refer to consultations (fuzzy matching)
    consultation_keywords = ["Beratung", "Termin"]
    if fuzzy_keyword_match(user_input, consultation_keywords):
        consultation_message = "„Ich verstehe, dass du an einer Beratung interessiert bist.  Unten findest du unsere verfügbaren Experten und Termine.  Bitte kontaktiere uns, um einen Termin zu vereinbaren:"
        st.write(consultation_message) # Display the message to the user
        available_consultations = get_consultations() # Get the available consultations
        st.write(available_consultations)

        
# ----- FEEDBACK COLLECTION --------------------------------------------------------------------
# ----- Logging feedback -----

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
    # Check if there's enough history to access the user question and assistant answer
    if len(st.session_state.chat_history) > 1:
        user_question = st.session_state.chat_history[-2]["content"]
        last_message = st.session_state.chat_history[-1]  # Ensure last_message is defined
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
            st.session_state.feedback_submitted = False  # Reset feedback prompt only once

            feedback_key = f"feedback_radio_{len(st.session_state.chat_history)}"
            st.markdown("### War die Antwort hilfreich??")
            feedback = st.radio(" ", ("👍 Ja", "👎 Nein"), index=None, key=feedback_key)

            if feedback:
                st.write("Thank you for your feedback!")

                # Map feedback to "Yes" or "No" (removing thumbs up for feedback file)
                feedback_mapping = {
                    "👍 Ja": "Ja",
                    "👎 Nein": "Nein"
                }

                # Log feedback to CSV
                user_question = st.session_state.chat_history[-2]["content"]
                assistant_answer = last_message["content"]
                feedback_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": user_question,
                    "answer": assistant_answer,
                    "feedback": feedback_mapping.get(feedback, feedback),
                }

                log_feedback(feedback_data)
                st.session_state.feedback_submitted = True  # Mark feedback as submitted

                del st.session_state[feedback_key]  # Remove the feedback radio button key

                # Optionally, store webinar suggestions here to preserve them
                if "webinar_suggestions" in st.session_state:
                    st.session_state.webinar_suggestions = webinar_response

                # Avoid rerun to prevent state reset
                # You may update the UI here instead to display feedback acknowledgment

log_chat_history()

