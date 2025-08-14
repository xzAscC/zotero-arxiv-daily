import arxiv
import os
import sys
from pyzotero import zotero
from src.recommender import rerank_paper
from src.construct_email import render_email, send_email
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from src.paper import ArxivPaper
from src.llm import set_global_llm
from src.database import CorpusDatabase
import feedparser

from src.config import config
from dotenv import load_dotenv
from tqdm import tqdm
from src.logger import setup_logger
from typing import Optional, List

load_dotenv(override=True)
ZOTERO_ID = os.getenv("ZOTERO_ID")
ZOTERO_KEY = os.getenv("ZOTERO_KEY")
SENDER = os.getenv("SENDER")
RECEIVER = os.getenv("RECEIVER")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")


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


def load_corpus_from_db(limit: Optional[int] = None) -> list[dict]:
    """
    Load corpus from local database.

    Args:
        limit: Maximum number of items to return

    Returns:
        List of corpus items from database
    """
    try:
        db = CorpusDatabase()
        corpus = db.get_corpus_with_abstracts(limit=limit)
        logger.info(f"Loaded {len(corpus)} items from local database")
        return corpus
    except Exception as e:
        logger.error(f"Failed to load corpus from database: {e}")
        return []


def search_corpus_in_db(query: str, search_fields: List[str] = None) -> list[dict]:
    """
    Search corpus in local database.

    Args:
        query: Search query string
        search_fields: Fields to search in (default: title, abstract)

    Returns:
        List of matching corpus items
    """
    try:
        db = CorpusDatabase()
        results = db.search_corpus(query, search_fields)
        logger.info(f"Found {len(results)} items matching query: {query}")
        return results
    except Exception as e:
        logger.error(f"Failed to search corpus in database: {e}")
        return []


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if "Feed error for query" in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
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


if __name__ == "__main__":

    args = config()
    logger = setup_logger(args.debug)
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    
    # TODO: save different items only and load items from database
    corpus = get_zotero_corpus(ZOTERO_ID, ZOTERO_KEY)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")

    logger.info("Retrieving Arxiv papers...")
    # TODO: start from there
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info(
            "No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY."
        )
        if not args.send_empty:
            exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[: args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(
                api_key=args.openai_api_key,
                base_url=args.openai_api_base,
                model=args.model_name,
                lang=args.language,
            )
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(
        SENDER, RECEIVER, SENDER_PASSWORD, args.smtp_server, args.smtp_port, html
    )
    logger.success(
        "Email sent successfully! If you don't receive the email, please check the configuration and the junk box."
    )
