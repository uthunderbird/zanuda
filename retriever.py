from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever


def produce_retriever(texts: list[str]) -> BaseRetriever:
    text_chunks = []
    for text in texts:
        text_chunks.extend(split_text(text))
    return FAISS.from_texts(text_chunks, embedding=OpenAIEmbeddings()).as_retriever()


def split_text(text: str) -> list[str]:
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=50,
    )
    return text_splitter.split_text(text)


if __name__ == '__main__':
    retriever = produce_retriever(
        [
            "harrison works in epam",
            "harrison have two children",
            "harrison have 25 years old wife",
            "harrison have a dog",
            "harrison is 27 years old",
            "harrison wrote Embedded Scrolls"
        ]
    )
    res = retriever.invoke("Does harrison have family?")
    print(res)