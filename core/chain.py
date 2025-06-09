from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from core.base_memory import PersistentConversationSummaryMemory
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# llm = ChatOllama(
#     model="mistral",
#     temperature=0.7
# )

def get_conversation_chain():
    memory = PersistentConversationSummaryMemory(
        session_id="default_session",
        llm=llm,
        max_token_limit=1000
    )
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    return chain, memory