from graph.base_estado import EstadoConversacion
from langgraph.graph import StateGraph, END
from agents.registry import AGENTS

def orquestador(state: EstadoConversacion) -> str:
    pregunta = state["input"]
    agent_name = "default"
    agent_score = 0
    for agent in AGENTS:
        score = agent.compatibility(pregunta)
        if score>agent_score:
            agent_name = agent.name
            agent_score = score
    return agent_name

def construir_grafo():
    builder = StateGraph(state_schema=EstadoConversacion)

    builder.add_node("entrada", lambda state: state)
    builder.add_node("default", lambda state: {
        "answer": "No entendí tu pregunta, ¿puedes reformularla?"
    })

    for agent in AGENTS:
        builder.add_node(agent.name, agent.run)

    builder.set_entry_point("entrada")

    branches = {agent.name: agent.name for agent in AGENTS}
    branches["default"] = "default"
    builder.add_conditional_edges("entrada", orquestador, branches)

    builder.add_edge("default", END)
    for agent in AGENTS:
        builder.add_edge(agent.name, END)

    return builder.compile()
