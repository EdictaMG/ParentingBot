import streamlit as st
import os
import csv
from datetime import datetime
import pandas as pd
from newspaper import Article
import requests #for booking APIs
from rapidfuzz import fuzz  #used for approximate keyword matching (wording needs not be identical)
from collections import defaultdict
import re

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
st.title("Parenting Bot: Help at Your Fingertips")
st.markdown("""
Ask me anything about parenting.  My answers are based on hundreds of articles written by our professionals and available in the [Elternleben website.](https://www.elternleben.de)
""")

st.markdown("**Sample Questions:**")
st.markdown("- My child does not stop crying; how can I help him?")
st.markdown("- My baby has red marks all over his body; what do I do?")

# ----- SIDEBAR ----------------------------------------------------------------------------------------------
with st.sidebar:
    st.image(os.path.join(os.path.dirname(__file__), "Images", "elternleben_holding_hands.jpg"), use_container_width=True)
    
    st.title("Hello! I'm ParentingBot")
    st.markdown("Here to help you in your journey with parenting.")
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

    if st.button("🔄 Want to restart the conversation?"):
        st.session_state["chat_history"] = []
        st.session_state["mentioned_topics"] = set()
        st.session_state["topic_counts"] = defaultdict(int)
        st.success("Conversation history has been cleared. We can start fresh 😊")

    st.markdown("---")
    st.markdown("### Our resources:")

    st.markdown("""
        [Join a live webinar](https://www.elternleben.de/elternsprechstunde/)
    """)

    st.markdown("""
        [Get in touch with us](https://www.elternleben.de/ueber-stell-uns-deine-frage/)
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
            print(f"Failed to download {url}: {e}")
    return docs

@st.cache_resource
def get_documents_from_urls(csv_path): # Loading the URLs and downloads/returns the parsed articles (the full process of extracting and processing article content).
    urls = load_article_urls(csv_path)
    return download_articles(urls)

# ----- SEARCH ENGINE SETUP ------------------------------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "data", "metadata.csv")
documents = get_documents_from_urls(csv_path)

hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(
    model_name=hf_model,
    task="text-generation",
    token="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" #to get a free token, visit https://huggingface.co/, create an account, select "Settings" from dropdown menu, in left sidebar, within "Access Tokens" click on "New Token", select "read", and generate. Copy and save token! 
)

embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=os.path.join(os.path.dirname(__file__), "Embeddings")
)

text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)

## Uncomment for first run-------
#vector_index = VectorStoreIndex.from_documents(
    #documents,
    #transformations=[text_splitter],
    #embed_model=embeddings
#)
#vector_index.storage_context.persist(persist_dir=os.path.join(os.path.dirname(__file__), "Vector_index"))

# Uncomment these lines after first run (and comment out the first-run lines, 104-109)-------
storage_context = StorageContext.from_defaults(persist_dir=os.path.join(os.path.dirname(__file__), "Vector_index"))
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

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
            Does he need quick, personalized advice?
            Are they looking for general information or do they have specific questions?
            Do they want to delve deeper into a topic?
            
            Understand the user's needs better before recommending something:
            Always ask at least one or two follow-up questions to make sure you understand the user properly before making recommendations.
            
            3. Recommend suitable services based on needs:
            
a. Articles:
            ElternLeben.de has articles written by experts on many topics.
            Don't give general information on the topic, but on the specific problem.
            You only give tips that you know from articles on ElternLeben.de.
            If you give a tip or information, always link to the source (a page from ElternLeben.de).
            Only link to ElternLeben.de.
            Only link URLs that you know.
            If you give tips or link to an article, you must also always recommend a suitable service (e-mail advice, midwife consultation - for baby parents -, parenting consultations - if the topic is suitable - or on-demand courses and eBooks - if the topic is suitable).
            This additional recommendation also always has a link.
            
            b. Email advice:
            Offer this service if the user has specific questions.
            The user will receive an answer within 36-48 hours.
            https://www.elternleben.de/ueber-stell-uns-deine-frage/
            
            c. Parent consultation (Zoom group consultation):
            Only recommend this for baby sleep, parenting, cleanliness education, kindergarten or screen time.
            https://www.elternleben.de/elternsprechstunde/
            
            d. Midwife advice:
            Recommend this service for topics related to baby sleep or breastfeeding (only for children under 1 year).
            https://www.elternleben.de/hebammensprechstunde/
            
            
e. On-demand courses and eBooks:
            If the user wants to delve deeper into a topic or is looking for more comprehensive information, you can recommend paid courses or eBooks from ElternLeben.de.
            You always link to the real URL of the course (or eBook) on ElternLeben.de.
            You know the URL or it is in the knowledge base (file shop_1.csv).
            Only link to ElternLeben.de.
            Only link URLs that you know.
            
            
Important notes:
            
Always recommend email advice at the end of the conversation in case further support is needed.
            
Your style should be friendly and professional, with emojis to lighten things up, always on you.
            Keep replies short and readable for mobile devices.
            
Always add "?pilot=" and 5 random numbers as a tracking code at the end of links - you will always generate a new random sequence.
            
            

FURTHER RULES:
            
You do not enter into conversations on topics that do not fall within your area of responsibility or that of ElternLeben.de.
            For any user query, you should ALWAYS consult your source of knowledge, even if you think you already know the answer.
            Your answer MUST be based on the information provided by that knowledge source.
            If a user asks questions that go beyond the actual topic, you should not answer them.
            Instead, kindly redirect to a topic you can help with.
        """)
    )
]

# ----- Initialize bot -----
@st.cache_resource
def init_bot():
    return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)

rag_bot = init_bot()


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
                expert_slots[name] = "No available slots."
        else:
            expert_slots[name] = "Unable to get any slots at this moment."
            
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


# ----- HANDLING USER INPUT ----------------------------------------------------------------------------------
#-------- user input -------------
if user_input := st.chat_input("What can I help with today?"):
    st.chat_message("human", avatar=os.path.join(os.path.dirname(__file__), "Images", "parent.jpg")).markdown(user_input)
    st.session_state.chat_history.append({"role": "human", "content": user_input})

# ------- chatbot answers using RAG ----------
    with st.spinner("📂 Searching parenting articles..."):
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
            answer = f"Sorry, I had trouble processing your question: {e}"

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
    st.write("Mentioned topics:", st.session_state.mentioned_topics)

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
                    f"- [**{webinar['topic']}**]({webinar['join_url']})  \n  📅 Date: {webinar['start_time']}  \n  ⏰ Duration: {webinar['duration']}"
                    for webinar in matching_webinars
                )
                webinar_response = (
                    f"I noted your interest in **{topic}** as you mentioned the topic a few times.  \n"
                    f"Perhaps you'd be interested in a webinar that goes deeper into the subject. Here are some upcoming webinars I can highly recommend, as they are presented by our experts:\n\n"
                    f"{webinars_text}"
                )
            else:
                webinar_response = (
                    f"I noted your interest in **{topic}** and I wanted to recommend to you some potential webinars led by our experts. \n"
                    f'Unfortunately, I couldn\'t find anything relevant coming up. Check out our list of webinars by clicking on the button under "Our resources" on the left.'

                )        
        
# ---- show webinar suggestions  ----
            st.chat_message("assistant").markdown(webinar_response)        

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
            st.markdown("### Was this answer helpful?")
            feedback = st.radio(" ", ("👍 Yes", "👎 No"), index=None, key=feedback_key)

            if feedback:
                st.write("Thank you for your feedback!")

                # Map feedback to "Yes" or "No" (removing thumbs up for feedback file)
                feedback_mapping = {
                    "👍 Yes": "Yes",
                    "👎 No": "No"
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

