from typing import TypedDict
from langchain.chains import ConversationChain

class EstadoConversacion(TypedDict):
    chain: ConversationChain
    input: str
    output: str