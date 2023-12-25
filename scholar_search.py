import os
import uuid
from pathlib import Path

import requests
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from pydantic.v1 import BaseModel as V1BaseModel
from pydantic import BaseModel
from pydantic.v1 import Field
from pydantic.fields import cached_property
from serpapi import GoogleScholarSearch

from pdf import read_file
from scihub_downloader import download as download_from_scihub
from retriever import split_text
from retriever import produce_retriever
from summarizer import summarize


def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True


class SearchResult(BaseModel):
    title: str
    link: str
    resource_link: str
    authors: str
    format: str | None = None
    summary: str

    @cached_property
    def text(self) -> str | None:
        try:
            fp = self.download()
        except:
            fp = None
        if fp is None:
            return None
        return read_file(file_path=fp)

    @property
    def retriever(self) -> BaseRetriever | None:
        if self.text is None:
            return None
        texts = split_text(self.text)
        return produce_retriever(texts)

    def summarize(self, question: str) -> str | None:
        retriever_ = self.retriever
        if retriever_:
            fragments = retriever_.invoke(question)
            return self.citation(summarize(fragments, question))

    def _download(self, url) -> Path:
        file_name = str(uuid.uuid4()) + '.pdf'
        r = requests.get(url, allow_redirects=True)
        open(file_name, 'wb').write(r.content)
        return Path(file_name)

    def download_resource(self) -> Path | None:
        if self.resource_link and is_downloadable(self.resource_link):
            return self._download(self.resource_link)

    def download_link(self) -> Path | None:
        if is_downloadable(self.link):
            return self._download(self.link)
        else:
            try:
                return download_from_scihub(self.link)
            except FileNotFoundError:
                return None

    def download(self) -> Path | None:
        result = self.download_resource()
        if result is None:
            result = self.download_link()
        return result

    @classmethod
    def parse_from_serp_dict(cls, sd: dict):
        authors = ', '.join(x['name'] for x in sd.get('publication_info',
                                                      {}).get('authors',
                                                              [])) or ''
        resource_link = sd.get('resources', [{}])[0].get('link', '')
        resource_file_format = sd.get('resources', [{}])[0].get('file_format')
        if resource_file_format != 'PDF':
            resource_link = ''
        return cls(
            title=sd['title'],
            link=sd['link'],
            summary=sd['publication_info'].get('summary', ""),
            authors=authors,
            resource_link=resource_link,
            format=sd.get('type', None)
        )

    def citation(self, text) -> str:
        return f"\"{self.title}\". {self.summary}\n\nSummary: {text}"


def search(query: str) -> list[SearchResult]:
    # for now, it will search only first page and cannot guarantee do
    # download all results
    result = GoogleScholarSearch(
        {
            'q': query,
            'serp_api_key': os.environ['SERPAPI_API_KEY'],
        },
    )
    dct = result.get_dict()
    wrapped_results = []
    if 'organic_results' not in dct:
        print(dct)
        return []
    for res in dct['organic_results']:
        wrapped_results.append(SearchResult.parse_from_serp_dict(res))
    return wrapped_results


class CitationToolSchema(V1BaseModel):
    query: str = Field(description="search query for Google Scholar. Always "
                                   "should be on English. Could be rephrased to produce best search results.")
    question: str = Field(description="a question that should be answered "
                                      "over found documents.")


results = []


@tool
def search_in_google_scholar(query: str) -> str:
    """Useful to search papers over
    Google Scholar and save them into global context. Always write queries
    on English!"""
    for result in search(query):
        results.append(result)
    return "Results found and saved in global context"


@tool
def read_papers(question: str):
    """Read papers that were found and saved by investigator to find parts
    that is relevant to the question. Always write questions
    on English!"""
    citations = []
    for result in results:
        citations.append(result.summarize(question))
    if not citations:
        return "Nothing found"
    return '\n\n\n'.join(filter(None, citations))


#     StructuredTool.from_function(
#     func=produce_citations,
#     name='search-papers',
#     args_schema=CitationToolSchema,
#     # infer_schema=True,
#     description="Useful to search whitepapers over Google Scholar and ask a "
#                 "question to its authors."
# )


if __name__ == '__main__':
    test_res = {
      "position": 3,
      "title": "Decoding ability in French as a foreign language and language learning motivation",
      "result_id": "7tNLdwrTTTcJ",
      "link": "https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-4781.2011.01238.x",
      "snippet": "\u2026 , coupled with their lack of complementarity, apparently creates difficulties for beginner English learners of French and, as we shall see, these difficulties have begun to be documented \u2026",
      "publication_info": {
        "summary": "L Erler, E Macaro - The Modern Language Journal, 2011 - Wiley Online Library"
      },
      "inline_links": {
        "serpapi_cite_link": "https://serpapi.com/search.json?engine=google_scholar_cite&hl=en&q=7tNLdwrTTTcJ",
        "cited_by": {
          "total": 102,
          "link": "https://scholar.google.com/scholar?cites=3985073287197348846&as_sdt=2005&sciodt=0,5&hl=en",
          "cites_id": "3985073287197348846",
          "serpapi_scholar_link": "https://serpapi.com/search.json?as_sdt=2005&cites=3985073287197348846&engine=google_scholar&hl=en"
        },
        "related_pages_link": "https://scholar.google.com/scholar?q=related:7tNLdwrTTTcJ:scholar.google.com/&scioq=french+language+learning&hl=en&as_sdt=0,5",
        "serpapi_related_pages_link": "https://serpapi.com/search.json?as_sdt=0%2C5&engine=google_scholar&hl=en&q=related%3A7tNLdwrTTTcJ%3Ascholar.google.com%2F",
        "versions": {
          "total": 8,
          "link": "https://scholar.google.com/scholar?cluster=3985073287197348846&hl=en&as_sdt=0,5",
          "cluster_id": "3985073287197348846",
          "serpapi_scholar_link": "https://serpapi.com/search.json?as_sdt=0%2C5&cluster=3985073287197348846&engine=google_scholar&hl=en"
        }
      }
    }

    sr = SearchResult.parse_from_serp_dict(test_res)

    print(sr.retriever.invoke("What is possible to do to find an easy way to learn French?"))
