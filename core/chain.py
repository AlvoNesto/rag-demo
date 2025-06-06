from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from core.base_memory import PersistentConversationSummaryMemory

# llm = ChatOpenAI(
#     model="gpt-4",
#     temperature=0.7,
#     openai_api_key="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"
# )

llm = ChatOllama(
    model="mistral",
    temperature=0.7
)

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