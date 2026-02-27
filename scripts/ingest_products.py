#!/usr/bin/env python3
"""
Ingest WillBBQ product catalog into Harbor's knowledge base.
Parses the products.md file and creates one chunk per product.

Usage:
    python scripts/ingest_products.py [--client-id willbbq] [--products-file PATH]
"""
import asyncio
import argparse
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.knowledge import KnowledgeService


DEFAULT_PRODUCTS = (
    Path(__file__).parent.parent.parent.parent
    / ".openclaw/workspace/projects/agent-smokey/knowledge/products.md"
)
DEFAULT_DB = "postgresql://sotastack:sotastack-local-2026@postgres.data.svc.cluster.local:5432/harbor"
# For local dev outside K3s:
LOCAL_DB = "postgresql://sotastack:sotastack-local-2026@localhost:5432/harbor"


def parse_products_md(text: str) -> list[dict]:
    """Parse products.md into structured chunks."""
    chunks = []
    # Extract store info section
    store_info_match = re.search(r"## Store Info\n(.*?)(?=\n## )", text, re.DOTALL)
    if store_info_match:
        chunks.append({
            "content": store_info_match.group(1).strip(),
            "source_type": "policy",
            "source_id": "store-info",
            "title": "WILLBBQ Store Info",
        })

    # Split by ### (each product)
    sections = re.split(r"\n### ", text)
    for section in sections[1:]:  # skip preamble
        lines = section.strip().split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        # Extract metadata
        price_match = re.search(r"\*\*Price:\*\*\s*\$([\d,.]+)", body)
        was_match = re.search(r"\(was \$([\d,.]+)\)", body)
        handle_match = re.search(r"\*\*Handle:\*\*\s*(\S+)", body)
        tags_match = re.search(r"\*\*Tags:\*\*\s*(.+)", body)

        price = price_match.group(1) if price_match else None
        was_price = was_match.group(1) if was_match else None
        handle = handle_match.group(1) if handle_match else None
        tags = [t.strip() for t in tags_match.group(1).split(",")] if tags_match else []

        # Build a rich content string for embedding
        # Include price + description so semantic search catches price queries too
        desc_lines = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("- **") or line.startswith("**"):
                continue  # skip metadata lines
            if line.startswith("- "):
                desc_lines.append(line[2:])
            elif line:
                desc_lines.append(line)
        description = " ".join(desc_lines).strip()

        content = f"{title}. "
        if price:
            content += f"Price: ${price}"
            if was_price:
                content += f" (was ${was_price})"
            content += ". "
        if tags:
            content += f"Tags: {', '.join(tags)}. "
        if description:
            content += description

        metadata = {
            "price": price,
            "was_price": was_price,
            "handle": handle,
            "tags": tags,
            "url": f"https://willbbq.com.au/products/{handle}" if handle else None,
        }

        chunks.append({
            "content": content,
            "source_type": "product",
            "source_id": handle or title.lower().replace(" ", "-")[:100],
            "title": title,
            "metadata": metadata,
        })

    return chunks


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", default="willbbq")
    parser.add_argument("--products-file", default=str(DEFAULT_PRODUCTS))
    parser.add_argument("--db", default=LOCAL_DB)
    parser.add_argument("--clear", action="store_true", help="Clear existing chunks first")
    args = parser.parse_args()

    products_path = Path(args.products_file)
    if not products_path.exists():
        print(f"❌ Products file not found: {products_path}")
        sys.exit(1)

    text = products_path.read_text()
    chunks = parse_products_md(text)
    print(f"📦 Parsed {len(chunks)} chunks from {products_path.name}")

    ks = KnowledgeService(args.db)
    await ks.initialize()

    if args.clear:
        deleted = await ks.delete_client(args.client_id)
        print(f"🗑️  Cleared {deleted} existing chunks for {args.client_id}")

    count = await ks.upsert_batch(args.client_id, chunks)
    print(f"✅ Ingested {count} chunks for client '{args.client_id}'")

    stats = await ks.stats(args.client_id)
    print(f"📊 Stats: {stats}")

    # Quick test search
    results = await ks.search(args.client_id, "best grill for family BBQ under $200")
    print(f"\n🔍 Test search: 'best grill for family BBQ under $200'")
    for r in results[:3]:
        print(f"  [{r.score:.3f}] {r.title}")

    await ks.close()


if __name__ == "__main__":
    asyncio.run(main())
