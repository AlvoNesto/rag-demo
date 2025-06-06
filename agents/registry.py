from agents.agent_consumo import AgentConsumo
from agents.agent_tarjetas import AgentTarjetas
from agents.base_agent import Agent

AGENTS: list[Agent] = [
    AgentConsumo(),
    AgentTarjetas()
]
