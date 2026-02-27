#!/usr/bin/env python3
"""
Sync products from a Shopify store's public /products.json endpoint
into Harbor's pgvector knowledge base.

No API key needed — uses the public storefront JSON.

Usage:
    python scripts/sync_shopify.py --store www.willbbq.com.au --client-id willbbq
    python scripts/sync_shopify.py --store www.willbbq.com.au --client-id willbbq --dry-run
"""
import asyncio
import argparse
import sys
from pathlib import Path

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.knowledge import KnowledgeService

log = structlog.get_logger()

DEFAULT_DB = "postgresql://sotastack:sotastack-local-2026@10.43.163.110:5432/harbor"


async def fetch_products(store: str, limit: int = 250) -> list[dict]:
    """Fetch all products from Shopify public JSON endpoint."""
    products = []
    page = 1
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            url = f"https://{store}/products.json?limit={limit}&page={page}"
            resp = await client.get(url)
            resp.raise_for_status()
            batch = resp.json().get("products", [])
            if not batch:
                break
            products.extend(batch)
            if len(batch) < limit:
                break
            page += 1
    return products


def product_to_chunk(product: dict, store: str) -> dict:
    """Convert a Shopify product JSON to a knowledge chunk."""
    handle = product["handle"]
    title = product["title"]
    product_type = product.get("product_type", "")
    tags = product.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    body_html = product.get("body_html", "")
    
    # Get variants for pricing
    variants = product.get("variants", [])
    prices = sorted(set(v["price"] for v in variants if v.get("price")))
    compare_prices = sorted(set(
        v["compare_at_price"] for v in variants 
        if v.get("compare_at_price") and v["compare_at_price"] != v["price"]
    ))
    
    # Check availability
    available = any(v.get("available", True) for v in variants)
    
    # Build variant info
    variant_info = ""
    if len(variants) > 1:
        variant_names = [v.get("title", "") for v in variants if v.get("title") and v["title"] != "Default Title"]
        if variant_names:
            variant_info = f" Available in: {', '.join(variant_names)}."
    
    # Strip HTML from description
    import re
    description = re.sub(r"<[^>]+>", " ", body_html or "").strip()
    description = re.sub(r"\s+", " ", description)
    # Truncate long descriptions
    if len(description) > 500:
        description = description[:500] + "..."
    
    # Build rich content for embedding
    content = f"{title}."
    if product_type:
        content += f" Category: {product_type}."
    if prices:
        if len(prices) == 1:
            content += f" Price: ${prices[0]}"
        else:
            content += f" Price: ${prices[0]} - ${prices[-1]}"
        if compare_prices:
            content += f" (was ${compare_prices[-1]})"
        content += "."
    if not available:
        content += " SOLD OUT."
    if tags:
        content += f" Tags: {', '.join(tags)}."
    if variant_info:
        content += variant_info
    if description:
        content += f" {description}"
    
    url = f"https://{store}/products/{handle}"
    
    metadata = {
        "price": prices[0] if prices else None,
        "compare_at_price": compare_prices[-1] if compare_prices else None,
        "handle": handle,
        "tags": tags,
        "product_type": product_type,
        "url": url,
        "available": available,
        "variants": len(variants),
        "shopify_id": product.get("id"),
        "updated_at": product.get("updated_at"),
    }
    
    return {
        "content": content,
        "source_type": "product",
        "source_id": handle,
        "title": title,
        "metadata": metadata,
    }


def store_info_chunk(store: str) -> dict:
    """Static store info chunk — update manually if store policies change."""
    return {
        "content": (
            f"WILLBBQ — Australia's home of premium charcoal grills and BBQ accessories. "
            f"Website: https://{store}. "
            f"Pickup: 2/56 Smith Rd, Springvale VIC 3171 (weekdays 9am-4pm). "
            f"Shipping: Free on grills and orders over $50 within Australia. Standard 3-7 business days. "
            f"Returns: 30-day returns on unused items in original packaging. "
            f"Warranty: 12-month warranty on all WillBBQ products. "
            f"Brands: WillBBQ (grills, pizza ovens, BBQ), Willkon (camping, outdoor)."
        ),
        "source_type": "policy",
        "source_id": "store-info",
        "title": "WILLBBQ Store Info",
    }


async def main():
    parser = argparse.ArgumentParser(description="Sync Shopify products to Harbor knowledge base")
    parser.add_argument("--store", default="www.willbbq.com.au")
    parser.add_argument("--client-id", default="willbbq")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse but don't write to DB")
    args = parser.parse_args()

    print(f"🛒 Fetching products from {args.store}...")
    products = await fetch_products(args.store)
    print(f"📦 Fetched {len(products)} products")

    chunks = [store_info_chunk(args.store)]
    for p in products:
        chunks.append(product_to_chunk(p, args.store))
    
    print(f"🔧 Built {len(chunks)} chunks ({len(chunks)-1} products + 1 store info)")

    if args.dry_run:
        print("\n🔍 Dry run — sample chunks:")
        for c in chunks[:3]:
            print(f"\n  [{c['source_type']}] {c['title']}")
            print(f"  URL: {c.get('metadata', {}).get('url', 'N/A')}")
            print(f"  Content preview: {c['content'][:150]}...")
        print(f"\n✅ Dry run complete. {len(chunks)} chunks would be ingested.")
        return

    ks = KnowledgeService(args.db)
    await ks.initialize()

    # Clear and re-ingest (atomic swap)
    deleted = await ks.delete_client(args.client_id)
    print(f"🗑️  Cleared {deleted} existing chunks")

    count = await ks.upsert_batch(args.client_id, chunks)
    print(f"✅ Ingested {count} chunks for '{args.client_id}'")

    stats = await ks.stats(args.client_id)
    print(f"📊 Stats: {stats}")

    # Quick sanity check
    results = await ks.search(args.client_id, "best portable grill for camping")
    print(f"\n🔍 Test search: 'best portable grill for camping'")
    for r in results[:3]:
        url = r.metadata.get("url", "N/A")
        available = "✅" if r.metadata.get("available", True) else "❌ SOLD OUT"
        print(f"  [{r.score:.3f}] {r.title} {available}")
        print(f"           {url}")

    await ks.close()
    print("\n🎉 Sync complete!")


if __name__ == "__main__":
    asyncio.run(main())
