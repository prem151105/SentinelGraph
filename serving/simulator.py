"""
asyncio-based transaction stream simulator.
Replays the IBM AML dataset as a live transaction stream.

Replaces Kafka with a lightweight asyncio queue + HTTP calls to the scoring API.
Same conceptual pattern: producer → topic/queue → consumer → alerts.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import aiofiles
import pandas as pd
import httpx

logger = logging.getLogger(__name__)

SCORING_API_URL = "http://localhost:8001/score"
ALERTS_LOG_PATH = Path("serving/alerts_stream.jsonl")


async def produce_transactions(
    csv_path: str,
    queue: asyncio.Queue,
    tps: int = 10,
    max_transactions: int = 10_000,
):
    """
    Read transactions from CSV and push them to the queue at `tps` rate.
    Simulates a live transaction stream.
    """
    logger.info(f"Producer starting | TPS: {tps} | Max: {max_transactions:,}")
    df = pd.read_csv(csv_path, nrows=max_transactions)

    delay = 1.0 / tps
    for i, (_, row) in enumerate(df.iterrows()):
        tx = _row_to_transaction(row, i)
        await queue.put(tx)
        await asyncio.sleep(delay)

    await queue.put(None)  # Sentinel: producer done
    logger.info(f"Producer finished: {i+1:,} transactions queued")


async def consume_and_score(
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
) -> dict:
    """
    Consume transactions from the queue, score each via the API, log alerts.

    Returns:
        Summary statistics after queue is drained.
    """
    stats = {
        "total": 0,
        "flagged": 0,
        "errors": 0,
        "latencies_ms": [],
    }

    ALERTS_LOG_PATH.parent.mkdir(exist_ok=True)

    async with aiofiles.open(ALERTS_LOG_PATH, "a") as alert_log:
        while True:
            tx = await queue.get()
            if tx is None:
                queue.task_done()
                break

            t_start = time.perf_counter()
            try:
                resp = await client.post(SCORING_API_URL, json=tx, timeout=5.0)
                latency_ms = (time.perf_counter() - t_start) * 1000
                stats["latencies_ms"].append(latency_ms)
                stats["total"] += 1

                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("is_fraud"):
                        stats["flagged"] += 1
                        alert = {
                            **result,
                            "original_tx": tx,
                            "alerted_at": datetime.utcnow().isoformat() + "Z",
                        }
                        await alert_log.write(json.dumps(alert) + "\n")

                        if stats["total"] % 100 == 0:
                            logger.info(
                                f"  Scored: {stats['total']:,} | "
                                f"Flagged: {stats['flagged']} | "
                                f"P99 latency: {_p99(stats['latencies_ms']):.1f}ms"
                            )
                else:
                    stats["errors"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"Scoring failed: {e}")

            queue.task_done()

    return stats


async def run_simulation(
    csv_path: str,
    tps: int = 10,
    max_transactions: int = 10_000,
):
    """
    Run the full producer-consumer simulation.
    """
    logger.info("="*50)
    logger.info(f"Starting FraudGraph stream simulation")
    logger.info(f"Data: {csv_path} | TPS: {tps} | Max: {max_transactions:,}")
    logger.info("="*50)

    queue: asyncio.Queue = asyncio.Queue(maxsize=tps * 10)

    async with httpx.AsyncClient() as client:
        # Check API is up
        try:
            health = await client.get("http://localhost:8001/health", timeout=3.0)
            logger.info(f"API health: {health.json()}")
        except Exception:
            logger.error("Scoring API not reachable. Start it with: uvicorn serving.main:app --port 8001")
            return

        t_start = time.monotonic()
        producer_task = asyncio.create_task(
            produce_transactions(csv_path, queue, tps, max_transactions)
        )
        consumer_task = asyncio.create_task(
            consume_and_score(queue, client)
        )

        await producer_task
        stats = await consumer_task
        elapsed = time.monotonic() - t_start

    # Print summary
    n = stats["total"]
    latencies = stats["latencies_ms"]
    print(f"\n{'='*50}")
    print(f"  Stream Simulation Complete")
    print(f"{'='*50}")
    print(f"  Transactions scored:  {n:,}")
    print(f"  Fraud alerts fired:   {stats['flagged']} ({stats['flagged']/max(n,1)*100:.2f}%)")
    print(f"  Errors:               {stats['errors']}")
    print(f"  Wall-clock time:      {elapsed:.1f}s")
    print(f"  Effective TPS:        {n/max(elapsed, 1):.1f}")
    if latencies:
        print(f"  Avg API latency:      {sum(latencies)/len(latencies):.1f}ms")
        print(f"  P99 API latency:      {_p99(latencies):.1f}ms")
    print(f"  Alerts log:           {ALERTS_LOG_PATH}")
    print(f"{'='*50}\n")
    return stats


def _row_to_transaction(row: pd.Series, idx: int) -> dict:
    """Convert a DataFrame row to a scoring API request dict."""
    return {
        "transaction_id": f"sim_{idx:08d}",
        "sender_account": str(row.get("Account", f"ACC_{idx}")),
        "receiver_account": str(row.get("Account.1", f"ACC_{idx+1}")),
        "amount_paid": float(row.get("Amount Paid", row.get("amount_paid", 1.0)) or 1.0),
        "amount_received": float(row.get("Amount Received", row.get("amount_received", 1.0)) or 1.0),
        "payment_currency": str(row.get("Payment Currency", "USD") or "USD"),
        "receiving_currency": str(row.get("Receiving Currency", "USD") or "USD"),
        "payment_format": str(row.get("Payment Format", "Wire Transfer") or "Wire Transfer"),
        "sender_bank": str(row.get("From Bank", "") or ""),
        "receiver_bank": str(row.get("To Bank", "") or ""),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _p99(values: list) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.99)
    return sorted_v[min(idx, len(sorted_v) - 1)]


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "./data/raw/HI-Small_Trans.csv"
    asyncio.run(run_simulation(csv_path))
