
from abc import ABC, abstractmethod


class AgentFactory(ABC):

    @abstractmethod
    def createAgent(self):
        pass