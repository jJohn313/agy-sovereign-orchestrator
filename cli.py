#!/usr/bin/env python3
"""
CLI interface for AGY Jev Orchestrator.
Usage:
    python3 cli.py "Check git status and inspect recent commits"
"""

import sys
import argparse
import json
import logging
from orchestrator import AgyOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="AGY Jev System-1 Orchestrator")
    parser.add_argument("prompt", help="User prompt to process")
    parser.add_argument("--json", action="store_true", help="Output full JSON lifecycle report")
    parser.add_argument("--user", default="john", help="User namespace for memory (default: john)")
    args = parser.parse_args()

    orchestrator = AgyOrchestrator()
    result = orchestrator.process_turn(args.prompt)

    if args.json:
        # Convert non-serializable objects if any
        clean_result = {
            "pre_scores": result.get("pre_scores"),
            "active_tools": result.get("active_tools"),
            "final_content": result.get("final_content"),
            "post_result": result.get("post_result"),
            "codebase_matches": len(result.get("retrieval_context", {}).get("codebase_matches", [])),
            "mem0_matches": len(result.get("retrieval_context", {}).get("mem0_matches", [])),
        }
        print(json.dumps(clean_result, indent=2))
    else:
        print("\n=== AGY JEV ORCHESTRATOR TURN REPORT ===")
        print(f"Prompt: {args.prompt}")
        print(f"Jev Pre-Gating: {result.get('pre_scores')}")
        print(f"Active Tools Mounted: {result.get('active_tools')}")
        mem_count = len(result.get("retrieval_context", {}).get("mem0_matches", []))
        cb_count = len(result.get("retrieval_context", {}).get("codebase_matches", []))
        print(f"Vector Retrieval: {cb_count} codebase, {mem_count} memory matches (>= 0.75)")
        print(f"Output: {result.get('final_content')}")
        print(f"Mem0 Write Gate: {result.get('post_result')}")


if __name__ == "__main__":
    main()
