from langchain.chains import ConversationChain
from agents.base_agent import Agent
from graph.base_estado import EstadoConversacion
from core.kb import load_vector_db, get_context

def compatibility(pregunta: str) -> float:
    return 0

class AgentPerfil(Agent):
    name = "perfil"
    compatibility = staticmethod(compatibility)

    def run(self, state: EstadoConversacion):
        db = load_vector_db("perfil")
        retriever = db.as_retriever(search_kwargs={"k": 2})
        chain = state["chain"]
        perfil_cliente = state["input"]
        context = get_context(perfil_cliente, retriever)
        answer = answer_question(chain, perfil_cliente, context)
        # answer = validate_answer(chain, answer)
        state["output"] = answer
        return state

def answer_question(chain: ConversationChain, perfil_cliente: str, context: str) -> str:
    prompt = f"""Genera un discurso de ventas para este cliente en base a los productos que manejamos definidos en el contexto.
        
        Contexto: {context}
        Cliente: {perfil_cliente}

        Lista los beneficios. Un ejemplo sería:
        'Hola, sé que te gusta comer y ahorrar, por lo que te ofrezco este producto por los siguientes beneficios:'
        y procede a listar con guiones los beneficios del producto.
        Si no hay info suficiente, muestra la más cercana."""
    return chain.run(prompt)

def validate_answer(chain: ConversationChain, discurso: str) -> str:
    prompt = f"""Revisa este discurso de ventas: '{discurso}'. 
        Si está bien estructurado, coherente y convincente, regrésalo tal cual o con mejoras sutiles. 
        Si no lo está, reformúla el contenido manteniendo la intención original."""
    return chain.run(prompt)
