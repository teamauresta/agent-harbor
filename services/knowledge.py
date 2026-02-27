"""
Harbor — Multi-tenant RAG knowledge base using pgvector.
Each client's knowledge is namespaced by client_id.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import structlog
from sqlalchemy import Column, String, Text, Index, text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.future import select
from pgvector.sqlalchemy import Vector

from services.embeddings import EMBEDDING_DIM, get_embedding, get_embeddings_batch

log = structlog.get_logger()


class Base(DeclarativeBase):
    pass


class KnowledgeChunk(Base):
    """Knowledge base chunks with vector embeddings."""

    __tablename__ = "knowledge_chunks"

    id = Column(String(64), primary_key=True)
    client_id = Column(String(100), nullable=False, index=True)  # multi-tenant key
    source_type = Column(String(50), index=True)  # product, faq, policy, bundle
    source_id = Column(String(255), index=True)  # product handle, faq id, etc.

    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    chunk_metadata = Column("metadata", Text, nullable=True)  # JSON

    embedding = Column(Vector(EMBEDDING_DIM))

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_knowledge_client_type", "client_id", "source_type"),
        Index(
            "ix_knowledge_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


@dataclass
class SearchResult:
    id: str
    client_id: str
    source_type: str
    source_id: str
    title: Optional[str]
    content: str
    metadata: Dict
    score: float


class KnowledgeService:
    """Multi-tenant RAG knowledge base."""

    def __init__(self, database_url: str):
        if "asyncpg" not in database_url:
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        self._engine = create_async_engine(database_url, echo=False, pool_size=5)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """Create tables + pgvector extension."""
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        log.info("harbor.knowledge.initialized")

    async def close(self):
        if self._engine:
            await self._engine.dispose()

    def _chunk_id(self, client_id: str, content: str, source_type: str, source_id: str) -> str:
        data = f"{client_id}:{source_type}:{source_id}:{content}"
        return hashlib.sha256(data.encode()).hexdigest()[:64]

    async def upsert_chunk(
        self,
        client_id: str,
        content: str,
        source_type: str,
        source_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        chunk_id = self._chunk_id(client_id, content, source_type, source_id)
        embedding = await get_embedding(content)

        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id)
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.content = content
                existing.title = title
                existing.chunk_metadata = json.dumps(metadata) if metadata else None
                existing.embedding = embedding
                existing.updated_at = datetime.now(timezone.utc)
            else:
                session.add(
                    KnowledgeChunk(
                        id=chunk_id,
                        client_id=client_id,
                        content=content,
                        source_type=source_type,
                        source_id=source_id,
                        title=title,
                        chunk_metadata=json.dumps(metadata) if metadata else None,
                        embedding=embedding,
                    )
                )
            await session.commit()
        return chunk_id

    async def upsert_batch(
        self, client_id: str, chunks: List[Dict[str, Any]]
    ) -> int:
        """Batch upsert chunks. Each dict: content, source_type, source_id, title?, metadata?"""
        if not chunks:
            return 0

        contents = [c["content"] for c in chunks]
        embeddings = await get_embeddings_batch(contents)

        count = 0
        async with self._session_factory() as session:
            for i, c in enumerate(chunks):
                chunk_id = self._chunk_id(
                    client_id, c["content"], c["source_type"], c["source_id"]
                )
                result = await session.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    existing.content = c["content"]
                    existing.title = c.get("title")
                    existing.chunk_metadata = (
                        json.dumps(c.get("metadata")) if c.get("metadata") else None
                    )
                    existing.embedding = embeddings[i]
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    session.add(
                        KnowledgeChunk(
                            id=chunk_id,
                            client_id=client_id,
                            content=c["content"],
                            source_type=c["source_type"],
                            source_id=c["source_id"],
                            title=c.get("title"),
                            chunk_metadata=(
                                json.dumps(c.get("metadata"))
                                if c.get("metadata")
                                else None
                            ),
                            embedding=embeddings[i],
                        )
                    )
                count += 1
            await session.commit()

        log.info("harbor.knowledge.batch_upsert", client_id=client_id, count=count)
        return count

    async def search(
        self,
        client_id: str,
        query: str,
        top_k: int = 5,
        source_types: Optional[List[str]] = None,
        min_score: float = 0.4,
    ) -> List[SearchResult]:
        """Semantic search scoped to a single client."""
        query_embedding = await get_embedding(query)
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = f"""
            SELECT id, client_id, source_type, source_id, title, content, metadata,
                   1 - (embedding <=> '{emb_str}'::vector) AS similarity
            FROM knowledge_chunks
            WHERE client_id = :client_id
              AND 1 - (embedding <=> '{emb_str}'::vector) > :min_score
        """
        params: dict = {"client_id": client_id, "min_score": min_score}

        if source_types:
            placeholders = ", ".join(f":st{i}" for i in range(len(source_types)))
            sql += f" AND source_type IN ({placeholders})"
            for i, st in enumerate(source_types):
                params[f"st{i}"] = st

        sql += f" ORDER BY similarity DESC LIMIT :top_k"
        params["top_k"] = top_k

        async with self._session_factory() as session:
            result = await session.execute(text(sql), params)
            rows = result.fetchall()

        return [
            SearchResult(
                id=r[0],
                client_id=r[1],
                source_type=r[2],
                source_id=r[3],
                title=r[4],
                content=r[5],
                metadata=json.loads(r[6]) if r[6] else {},
                score=r[7],
            )
            for r in rows
        ]

    async def get_context(
        self, client_id: str, query: str, max_chars: int = 3000
    ) -> str:
        """Get formatted product/knowledge context for the LLM prompt."""
        results = await self.search(client_id, query)
        if not results:
            return ""

        parts = []
        total = 0
        for r in results:
            part = f"[{r.source_type.upper()}] "
            if r.title:
                part += f"{r.title}: "
            part += r.content
            # Append real URL from metadata if available
            url = r.metadata.get("url") if r.metadata else None
            if url:
                part += f" | URL: {url}"
            if total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)

        return "\n\n".join(parts)

    async def delete_client(self, client_id: str) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                text("DELETE FROM knowledge_chunks WHERE client_id = :cid"),
                {"cid": client_id},
            )
            await session.commit()
            return result.rowcount

    async def stats(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        async with self._session_factory() as session:
            if client_id:
                result = await session.execute(
                    text(
                        "SELECT source_type, COUNT(*) FROM knowledge_chunks "
                        "WHERE client_id = :cid GROUP BY source_type"
                    ),
                    {"cid": client_id},
                )
            else:
                result = await session.execute(
                    text(
                        "SELECT client_id, source_type, COUNT(*) FROM knowledge_chunks "
                        "GROUP BY client_id, source_type ORDER BY client_id"
                    )
                )
            return {"rows": [dict(zip(result.keys(), r)) for r in result.fetchall()]}
