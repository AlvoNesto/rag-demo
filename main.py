from langchain.chains import ConversationChain
from graph.multiagent_graph import construir_grafo
from graph.base_estado import EstadoConversacion
from core.chain import get_conversation_chain
from agents.agent_perfil import AgentPerfil

def obtener_perfil(chain: ConversationChain) -> EstadoConversacion:
    print("Describe el perfil del cliente. Escribe 'ok' para generar el discurso:\n")
    entradas = []
    while True:
        user_input = input()
        if user_input.strip().lower() == "ok":
            break
        entradas.append(user_input)

    perfil_cliente = "\n".join(entradas)

    agente_discurso = AgentPerfil()
    respuesta = agente_discurso.run(EstadoConversacion({
        "chain": chain,
        "input": perfil_cliente,
        "output": ""
    }))

    print("\n--- Discurso ---\n")
    print(respuesta["output"])
    
    return respuesta


def main():
    print("🤖 Chat iniciado. Pulsa Ctrl+C para terminar el programa.")

    graph = construir_grafo()
    conversation_chain, memory = get_conversation_chain()

    state = obtener_perfil(conversation_chain)
    memory.save_context({"input": state["input"]}, {"output": state["output"]})
    memory.save_memory()

    try:
        while True:
            question = input("\nTú: ")

            result = graph.invoke(EstadoConversacion({
                "chain": conversation_chain,
                "input": question,
                "output": ""
            }))

            print("\n🤖 Respuesta Final:\n", result["output"])

            # Guardar conversación después de cada respuesta
            memory.save_context({"input": question}, {"output": result["output"]})
            memory.save_memory()
    except KeyboardInterrupt:
        print("🤖 Chat finalizado.")

if __name__ == "__main__":
    main()