"""
Database module for storing and managing Zotero corpus data locally.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CorpusDatabase:
    """
    SQLite database for storing Zotero corpus data with abstracts.
    """
    
    def __init__(self, db_path: str = "data/corpus.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create corpus table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corpus (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    zotero_key TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    authors TEXT,
                    item_type TEXT,
                    date_added TEXT,
                    date_modified TEXT,
                    collections TEXT,
                    paths TEXT,
                    url TEXT,
                    doi TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create collections table for better organization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_key TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    parent_collection TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster searches
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_corpus_abstract 
                ON corpus(abstract)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_corpus_title 
                ON corpus(title)
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def store_corpus(self, corpus: List[Dict[str, Any]]) -> int:
        """
        Store the filtered corpus in the database.
        
        Args:
            corpus: List of corpus items with abstracts
            
        Returns:
            Number of items successfully stored
        """
        stored_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for item in corpus:
                try:
                    # Extract basic information
                    data = item.get("data", {})
                    zotero_key = item.get("key", "")
                    title = data.get("title", "")
                    abstract = data.get("abstractNote", "")
                    authors = json.dumps(data.get("creators", []))
                    item_type = data.get("itemType", "")
                    date_added = data.get("dateAdded", "")
                    date_modified = data.get("dateModified", "")
                    url = data.get("url", "")
                    doi = data.get("doi", "")
                    
                    # Handle collections and paths
                    collections = json.dumps(item.get("collections", []))
                    paths = json.dumps(item.get("paths", []))
                    
                    # Insert or update corpus item
                    cursor.execute("""
                        INSERT OR REPLACE INTO corpus (
                            zotero_key, title, abstract, authors, item_type,
                            date_added, date_modified, collections, paths, url, doi
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        zotero_key, title, abstract, authors, item_type,
                        date_added, date_modified, collections, paths, url, doi
                    ))
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing item {item.get('key', 'unknown')}: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"Successfully stored {stored_count} corpus items")
        return stored_count
    
    def get_corpus_with_abstracts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve corpus items that have abstracts.
        
        Args:
            limit: Maximum number of items to return (None for all)
            
        Returns:
            List of corpus items with abstracts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM corpus WHERE abstract != '' AND abstract IS NOT NULL"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            corpus = []
            for row in rows:
                item = dict(row)
                # Parse JSON fields back to Python objects
                item["authors"] = json.loads(item["authors"]) if item["authors"] else []
                item["collections"] = json.loads(item["collections"]) if item["collections"] else []
                item["paths"] = json.loads(item["paths"]) if item["paths"] else []
                corpus.append(item)
            
            return corpus
    
    def search_corpus(self, query: str, search_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search corpus by text query.
        
        Args:
            query: Search query string
            search_fields: Fields to search in (default: title, abstract)
            
        Returns:
            List of matching corpus items
        """
        if search_fields is None:
            search_fields = ["title", "abstract"]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build search query
            search_conditions = []
            params = []
            
            for field in search_fields:
                search_conditions.append(f"{field} LIKE ?")
                params.append(f"%{query}%")
            
            where_clause = " OR ".join(search_conditions)
            sql_query = f"""
                SELECT * FROM corpus 
                WHERE ({where_clause}) 
                AND abstract != '' AND abstract IS NOT NULL
                ORDER BY updated_at DESC
            """
            
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            corpus = []
            for row in rows:
                item = dict(row)
                item["authors"] = json.loads(item["authors"]) if item["authors"] else []
                item["collections"] = json.loads(item["collections"]) if item["collections"] else []
                item["paths"] = json.loads(item["paths"]) if item["paths"] else []
                corpus.append(item)
            
            return corpus
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total items
            cursor.execute("SELECT COUNT(*) FROM corpus")
            total_items = cursor.fetchone()[0]
            
            # Items with abstracts
            cursor.execute("SELECT COUNT(*) FROM corpus WHERE abstract != '' AND abstract IS NOT NULL")
            items_with_abstracts = cursor.fetchone()[0]
            
            # Items without abstracts
            cursor.execute("SELECT COUNT(*) FROM corpus WHERE abstract = '' OR abstract IS NULL")
            items_without_abstracts = cursor.fetchone()[0]
            
            # Item types distribution
            cursor.execute("""
                SELECT item_type, COUNT(*) 
                FROM corpus 
                GROUP BY item_type 
                ORDER BY COUNT(*) DESC
            """)
            item_types = dict(cursor.fetchall())
            
            # Latest update
            cursor.execute("SELECT MAX(updated_at) FROM corpus")
            latest_update = cursor.fetchone()[0]
            
            return {
                "total_items": total_items,
                "items_with_abstracts": items_with_abstracts,
                "items_without_abstracts": items_without_abstracts,
                "item_types": item_types,
                "latest_update": latest_update
            }
    
    def clear_corpus(self) -> None:
        """Clear all corpus data from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM corpus")
            conn.commit()
            logger.info("Corpus database cleared")
    
    def close(self) -> None:
        """Close the database connection."""
        pass  # SQLite connections are automatically closed 