import arxiv
import os
import sys
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser

from config import config
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger

load_dotenv(override=True)
ZOTERO_ID = os.getenv("ZOTERO_ID")
ZOTERO_KEY = os.getenv("ZOTERO_KEY")
SENDER = os.getenv("SENDER")
RECEIVER = os.getenv("RECEIVER")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")


def get_zotero_corpus(id: str, key: str) -> list[dict]:
    zot = zotero.Zotero(id, "user", key)
    collections = zot.everything(zot.collections())
    collections = {c["key"]: c for c in collections}
    corpus = zot.everything(
        zot.items(itemType="conferencePaper || journalArticle || preprint")
    )
    corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

    def get_collection_path(col_key: str) -> str:
        if p := collections[col_key]["data"]["parentCollection"]:
            return get_collection_path(p) + "/" + collections[col_key]["data"]["name"]
        else:
            return collections[col_key]["data"]["name"]

    for c in corpus:
        paths = [get_collection_path(col) for col in c["data"]["collections"]]
        c["paths"] = paths
    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    _, filename = mkstemp()
    with open(filename, "w") as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir="./")
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c["paths"]]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


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


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == "" or v is None:
            return default
        return v

    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get("dest", args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        # convert env_value to the specified type
        if kwargs.get("type") == bool:
            env_value = env_value.lower() in ["true", "1"]
        else:
            env_value = kwargs.get("type")(env_value)
        parser.set_defaults(**{arg_full_name: env_value})


if __name__ == "__main__":

    args = config()

    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(ZOTERO_ID, ZOTERO_KEY)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
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
