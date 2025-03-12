
from langchain_core.embeddings.embeddings import Embeddings



class MockEmbeddings(Embeddings):

    def __init__(self):
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[hash(text)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [hash(text)]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[hash(text)] for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [hash(text)]
