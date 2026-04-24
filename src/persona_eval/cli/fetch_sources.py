"""`persona-eval fetch-sources` — pre-fetch OpenAlex abstracts and titles."""

from __future__ import annotations

from persona_eval.annotations import load_annotations
from persona_eval.cli._args import add_data_args
from persona_eval.openalex import OpenAlexClient


def register(subparsers):
    sp = subparsers.add_parser("fetch-sources", help="Fetch OpenAlex source documents")
    add_data_args(sp)
    sp.set_defaults(func=run)


def run(args):
    _, entries = load_annotations(args.annotations)
    client = OpenAlexClient(cache_dir=args.cache_dir, email=args.email)
    source_texts, reference_texts = client.get_texts_batch(entries)
    print(f"Fetched texts for {len(source_texts)} unique queries")
    n_empty_src = sum(1 for t in source_texts.values() if not t)
    n_empty_ref = sum(1 for t in reference_texts.values() if not t)
    if n_empty_src:
        print(f"  ({n_empty_src} queries had no abstracts available)")
    if n_empty_ref:
        print(f"  ({n_empty_ref} queries had no titles available)")
