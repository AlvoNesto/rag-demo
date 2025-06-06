from langchain.chains import ConversationChain
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import FastEmbedEmbeddings
from agents.base_agent import Agent
from graph.base_estado import EstadoConversacion
from core.kb import load_vector_db, get_context
from functools import lru_cache

embedding_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=100)
def cached_embedding(text: str):
    return embedding_model.embed_query(text)

def compatibility(pregunta: str) -> float:
    concepto = "quiero información sobre mis opciones de consumo de productos o servicios usando mi tarjeta bancaria"
    vector_pregunta = cached_embedding(pregunta)
    vector_concepto = cached_embedding(concepto)
    similitud = cosine_similarity([vector_pregunta], [vector_concepto])[0][0]
    return similitud

class AgentConsumo(Agent):
    name = "consumo"
    compatibility = staticmethod(compatibility)

    def run(self, state: EstadoConversacion):
        db = load_vector_db("consumo")
        retriever = db.as_retriever(search_kwargs={"k": 2})
        chain = state["chain"]
        question = state["input"]
        context = get_context(question, retriever)
        answer = answer_question(chain, question, context)
        # answer = validate_answer(chain, question, answer, context)
        state["output"] = answer
        return state

def answer_question(chain: ConversationChain, question: str, context: str) -> str:
    prompt = f"""Usa la siguiente información para responder a la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

        Contexto: {context}
        Pregunta: {question}

        Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español.
        Respuesta útil:"""
    return chain.run(prompt)

def validate_answer(chain: ConversationChain, question: str, answer: str, context: str) -> str:
    prompt = f"""Debes analizar la respuesta anterior con base en el contexto disponible. 
        Si la respuesta es incorrecta, ambigua o incompleta, debes corregirla o completarla. 
        Si es correcta, puedes devolverla tal cual, pero asegurándote de que sea clara y precisa.

        Contexto: {context}
        Pregunta: {question}
        Respuesta anterior: {answer}

        No digas si la respuesta está bien o mal. Solo entrega la respuesta final mejorada directamente y en español natural.

        Respuesta mejorada:"""
    return chain.run(prompt)
