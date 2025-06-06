from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
from graph.base_estado import EstadoConversacion

class Agent(ABC):
    name: str
    compatibility: Callable[[str], bool]

    @abstractmethod
    def run(self, state: EstadoConversacion) -> EstadoConversacion:
        pass
