import arxiv
import tarfile
import re
import requests
import feedparser

import numpy as np
from loguru import logger
from tqdm import tqdm
from database import CorpusDatabase
from typing import Optional
from functools import cached_property
from tempfile import TemporaryDirectory
from contextlib import ExitStack
from urllib.error import HTTPError
from requests.adapters import HTTPAdapter, Retry
from pyzotero import zotero
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime


class ArxivPaper:
    def __init__(self, paper: arxiv.Result):
        self._paper = paper
        self.title = paper.title
        self.summary = paper.summary
        self.authors = [a.name for a in paper.authors]
        self.arxiv_id = re.sub(r"v\d+$", "", paper.get_short_id())
        self.pdf_url = paper.pdf_url


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if "Feed error for query" in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        # TODO: why not just use the feed directly, compared with arxiv.Search?
        papers = []
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.arxiv_announce_type == "new"
        ]
        bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv papers")
        for i in range(0, len(all_paper_ids), 50):
            search = arxiv.Search(id_list=all_paper_ids[i : i + 50])
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(
            query="cat:cs.AI", sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers


def get_zotero_corpus(id: str, key: str, save_to_db: bool = True) -> list[dict]:
    """
    Retrieve Zotero corpus and optionally save to local database.

    Args:
        id: Zotero user ID
        key: Zotero API key
        save_to_db: Whether to save corpus to local database

    Returns:
        Filtered corpus with abstracts
    """
    zot = zotero.Zotero(id, "user", key)
    collections = zot.everything(zot.collections())
    collections = {c["key"]: c for c in collections}

    # Get all items
    # TODO: get only the items that are not in the database
    corpus = zot.everything(
        zot.items(
            itemType="conferencePaper || journalArticle || preprint || WebPage || Book"
        )
    )

    # Filter to only include items with abstracts
    corpus_with_abstracts = [c for c in corpus if c["data"]["abstractNote"] != ""]

    logger.info(
        f"Retrieved {len(corpus)} total items, {len(corpus_with_abstracts)} have abstracts"
    )
    # Add collection paths
    for c in corpus_with_abstracts:
        paths = [
            get_collection_path(collections, col) for col in c["data"]["collections"]
        ]
        c["paths"] = paths

    # Save to database if requested
    if save_to_db:
        # TODO: load only different items and save the different ones
        try:
            db = CorpusDatabase()
            stored_count = db.store_corpus(corpus_with_abstracts)
            logger.info(f"Successfully stored {stored_count} items in local database")

            # Get database statistics
            stats = db.get_corpus_stats()
            logger.info(f"Database stats: {stats}")

        except Exception as e:
            logger.error(f"Failed to save corpus to database: {e}")
    return corpus_with_abstracts


def get_collection_path(collections: dict, col_key: str) -> str:
    """Get the full path of a collection."""
    if p := collections[col_key]["data"]["parentCollection"]:
        return (
            get_collection_path(collections, p)
            + "/"
            + collections[col_key]["data"]["name"]
        )
    else:
        return collections[col_key]["data"]["name"]


def rerank_paper(
    candidate: list[ArxivPaper],
    corpus: list[dict],
    model: str = "avsolatorio/GIST-small-Embedding-v0",
) -> list[ArxivPaper]:
    # TODO: rewrite the ranker function with RAG and local zotero corpus
    encoder = SentenceTransformer(model)
    # sort corpus by date, from newest to oldest
    corpus = sorted(
        corpus,
        key=lambda x: datetime.strptime(x["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"),
        reverse=True,
    )
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper["data"]["abstractNote"] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(
        candidate_feature, corpus_feature
    )  # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10  # [n_candidate]
    for s, c in zip(scores, candidate):
        c.score = s.item()
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    return candidate
