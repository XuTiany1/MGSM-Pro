"""
Claude Batch API utilities for math tasks.

This module provides functions to:
1. Create JSONL batch request files
2. Submit batch jobs
3. Monitor batch job status
4. Retrieve and process results
"""

import json
import time

from anthropic import Anthropic
from typing import Dict, Optional, Tuple

from pathlib import Path
from datetime import datetime


class ClaudeBatchApiManager:

    def __init__(
            self,
            client: Anthropic,
            base_dir: str,
            deployment: str = "claude-sonnet-4-0"
    ):
        self.client = client
        self.base_dir = Path(base_dir)
        self.deployment = deployment

        self.requests_dir = self.base_dir / "requests"
        self.results_dir = self.base_dir / "results"
        self.status_dir = self.base_dir / "status"

        for dir_path in [self.requests_dir, self.results_dir, self.status_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _save_status(
        self,
        *,
        batch_id: str,
        processing_status: Optional[str],
        results_url: Optional[str],
        created_at_iso: Optional[str],
        timestamp: str,
        extra: Optional[Dict] = None,
    ) -> Path:
        """Save batch status JSON under status/<timestamp>/batch_meta.json. Returns the saved path."""
        ts_dir = self.status_dir / timestamp
        ts_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "batch_id": batch_id,
            "created_at": created_at_iso,
            "processing_status": processing_status,
            "results_url": results_url,
        }
        if extra:
            payload.update(extra)

        meta_path = ts_dir / "batch_meta.json"
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def create_batch_request_file(
        self,
        custom_id_to_texts: Dict[str, str],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timestamp: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """Create a JSONL batch request file.

        Args:
            custom_id_to_texts: Dict mapping custom IDs to prompt texts
            temperature: Model temperature parameter
            max_tokens: Maximum tokens for response
            timestamp: Optional timestamp string (auto-generated if None)

        Returns:
            Tuple of (request_file_path, timestamp)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        timestamp_dir = self.requests_dir / timestamp
        timestamp_dir.mkdir(parents=True, exist_ok=True)

        request_file = timestamp_dir / "request.jsonl"

        with request_file.open("w", encoding="utf-8") as f:
            for cid, text in custom_id_to_texts.items():
                entry = {
                    "custom_id": cid,
                    "params": {
                        "model": self.deployment,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": text}]}
                        ],
                    },
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return request_file, timestamp

    def submit_batch_file(self, request_file: Path) -> str:
        """Submits a JSONL request file (from create_batch_request_file) to Claude. Returns the batch_id."""
        timestamp = request_file.parent.name

        requests = []
        with request_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "custom_id" not in obj or "params" not in obj:
                        raise ValueError("missing 'custom_id' or 'params'")
                    requests.append(obj)
                except Exception as e:
                    raise ValueError(f"Bad JSONL on line {i}: {e}")

        if not requests:
            raise ValueError("No requests found in the JSONL file.")

        batch = self.client.messages.batches.create(requests=requests)

        created_at = getattr(batch, "created_at", None)
        created_at_iso = created_at.isoformat() if created_at else None
        processing_status = getattr(batch, "processing_status", None)
        results_url = getattr(batch, "results_url", None)

        meta_path = self._save_status(
            batch_id=batch.id,
            created_at_iso=created_at_iso,
            processing_status=processing_status,
            results_url=results_url,
            timestamp=timestamp,
            extra={"source_request_file": str(request_file)},
        )

        print(f"[ClaudeBatchApiManager] Created batch: {batch.id}")
        print(f"[ClaudeBatchApiManager] Status saved -> {meta_path}")
        return batch.id

    def monitor_batch_job(
        self,
        batch_id: str,
        timestamp: str,
        poll_interval: int = 15,
        max_wait_seconds: Optional[int] = None,
        verbose: bool = True,
        write_history: bool = True,
    ) -> Dict[str, object]:
        """Polls a Claude Message Batch until it ends (processing_status == 'ended') or times out.

        Args:
            batch_id: The batch id returned from submit_batch_file.
            timestamp: The timestamp folder name under which to save status.
            poll_interval: Seconds to wait between polls.
            max_wait_seconds: If set, raise TimeoutError after this many seconds.
            verbose: If True, print status updates.
            write_history: If True, append each poll snapshot to status_history.jsonl.

        Returns:
            A dict summary of the final batch state (processing_status, counts, urls, etc.).
        """
        start = time.time()
        history_path = self.status_dir / timestamp / "status_history.jsonl"
        (self.status_dir / timestamp).mkdir(parents=True, exist_ok=True)

        def _dt_iso(dt):
            try:
                return dt.isoformat() if dt else None
            except Exception:
                return None

        while True:
            try:
                b = self.client.messages.batches.retrieve(batch_id)
            except Exception as e:
                if verbose:
                    print(f"[monitor] retrieve error: {e!r}; retrying in {poll_interval}s")
                time.sleep(poll_interval)
                continue

            counts = getattr(b, "request_counts", None)
            counts_dict = {
                "succeeded": getattr(counts, "succeeded", None),
                "errored": getattr(counts, "errored", None),
                "canceled": getattr(counts, "canceled", None),
                "expired": getattr(counts, "expired", None),
                "processing": getattr(counts, "processing", None),
            } if counts else {}

            snapshot = {
                "batch_id": b.id,
                "processing_status": getattr(b, "processing_status", None),
                "created_at": _dt_iso(getattr(b, "created_at", None)),
                "ended_at": _dt_iso(getattr(b, "ended_at", None)),
                "archived_at": _dt_iso(getattr(b, "archived_at", None)),
                "cancel_initiated_at": _dt_iso(getattr(b, "cancel_initiated_at", None)),
                "expires_at": _dt_iso(getattr(b, "expires_at", None)),
                "results_url": getattr(b, "results_url", None),
                "request_counts": counts_dict,
            }

            self._save_status(
                batch_id=b.id,
                created_at_iso=snapshot["created_at"],
                processing_status=snapshot["processing_status"],
                results_url=snapshot["results_url"],
                timestamp=timestamp,
                extra={
                    "ended_at": snapshot["ended_at"],
                    "archived_at": snapshot["archived_at"],
                    "cancel_initiated_at": snapshot["cancel_initiated_at"],
                    "expires_at": snapshot["expires_at"],
                    "request_counts": counts_dict,
                },
            )

            if write_history:
                with history_path.open("a", encoding="utf-8") as hf:
                    hf.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

            if verbose:
                c = counts_dict
                print(
                    f"[monitor] {snapshot['processing_status']}"
                    f" | succ={c.get('succeeded')} err={c.get('errored')} proc={c.get('processing')}"
                )

            if snapshot["processing_status"] == "ended":
                return snapshot

            if max_wait_seconds is not None and (time.time() - start) > max_wait_seconds:
                raise TimeoutError(f"Batch {batch_id} did not finish within {max_wait_seconds}s")

            time.sleep(poll_interval)

    def retrieve_batch_results(
        self,
        batch_id: str,
        timestamp: str,
        save_to_file: bool = True,
    ) -> Dict[str, Dict]:
        """Retrieves and parses results from a completed batch job.

        Returns:
            Dict mapping custom_id to a result dict with "type" ("succeeded"/"errored"),
            plus "message" or "error" depending on type.
        """
        try:
            batch = self.client.messages.batches.retrieve(batch_id)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve batch {batch_id}: {e}")

        processing_status = getattr(batch, "processing_status", None)
        if processing_status != "ended":
            raise ValueError(f"Batch {batch_id} has not ended yet (status: {processing_status})")

        results_url = getattr(batch, "results_url", None)
        if not results_url:
            raise ValueError(f"No results_url available for batch {batch_id}")

        results_dict = {}
        results_list = []

        try:
            for result in self.client.messages.batches.results(batch_id):
                custom_id = result.custom_id
                result_type = result.result.type

                result_entry = {
                    "custom_id": custom_id,
                    "type": result_type,
                }

                if result_type == "succeeded":
                    message = result.result.message
                    result_entry["message"] = {
                        "id": message.id,
                        "type": message.type,
                        "role": message.role,
                        "content": [
                            {
                                "type": c.type,
                                "text": c.text if hasattr(c, "text") else None
                            }
                            for c in message.content
                        ],
                        "model": message.model,
                        "stop_reason": message.stop_reason,
                        "usage": {
                            "input_tokens": message.usage.input_tokens,
                            "output_tokens": message.usage.output_tokens,
                        } if message.usage else None,
                    }
                elif result_type == "errored":
                    error = result.result.error
                    result_entry["error"] = {
                        "type": error.type,
                        "message": error.message if hasattr(error, "message") else str(error),
                    }

                results_dict[custom_id] = result_entry
                results_list.append(result_entry)

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve results for batch {batch_id}: {e}")

        if save_to_file:
            timestamp_dir = self.results_dir / timestamp
            timestamp_dir.mkdir(parents=True, exist_ok=True)

            results_file = timestamp_dir / "results.jsonl"
            with results_file.open("w", encoding="utf-8") as f:
                for result in results_list:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"[ClaudeBatchApiManager] Results saved -> {results_file}")

        return results_dict

    def compute_batch(
        self,
        custom_id_to_texts: Dict[str, str],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timestamp: Optional[str] = None,
        poll_interval: int = 15,
        max_wait_seconds: Optional[int] = None,
        verbose: bool = True,
        write_history: bool = True,
        save_results: bool = True,
    ) -> Tuple[str, Dict[str, Dict]]:
        """End-to-end batch workflow: create, submit, monitor, and retrieve results.

        Returns:
            Tuple of (batch_id, results_dict) where results_dict maps custom_id to result data.

        Example:
            >>> manager = ClaudeBatchApiManager(client, "./batch_data")
            >>> prompts = {"task_1": "What is 2+2?"}
            >>> batch_id, results = manager.compute_batch(prompts, verbose=True)
            >>> print(results["task_1"]["message"]["content"][0]["text"])
        """
        if not custom_id_to_texts:
            raise ValueError("custom_id_to_texts cannot be empty")

        if verbose:
            print("\n[compute_batch] Step 1/4: Creating batch request file...")
            print(f"[compute_batch] Number of requests: {len(custom_id_to_texts)}")

        request_file, timestamp = self.create_batch_request_file(
            custom_id_to_texts=custom_id_to_texts,
            temperature=temperature,
            max_tokens=max_tokens,
            timestamp=timestamp,
        )

        if verbose:
            print(f"[compute_batch] Request file created: {request_file}")
            print("\n[compute_batch] Step 2/4: Submitting batch to Claude API...")

        batch_id = self.submit_batch_file(request_file)

        if verbose:
            print(f"[compute_batch] Batch submitted successfully: {batch_id}")
            print("\n[compute_batch] Step 3/4: Monitoring batch job...")

        final_status = self.monitor_batch_job(
            batch_id=batch_id,
            timestamp=timestamp,
            poll_interval=poll_interval,
            max_wait_seconds=max_wait_seconds,
            verbose=verbose,
            write_history=write_history,
        )

        if verbose:
            counts = final_status.get("request_counts", {})
            print("\n[compute_batch] Batch completed!")
            print("[compute_batch] Summary:")
            print(f"  - Succeeded: {counts.get('succeeded', 0)}")
            print(f"  - Errored: {counts.get('errored', 0)}")
            print(f"  - Canceled: {counts.get('canceled', 0)}")
            print(f"  - Expired: {counts.get('expired', 0)}")
            print("\n[compute_batch] Step 4/4: Retrieving results...")

        results_dict = self.retrieve_batch_results(
            batch_id=batch_id,
            timestamp=timestamp,
            save_to_file=save_results,
        )

        if verbose:
            print(f"[compute_batch] Retrieved {len(results_dict)} results")
            print("\n[compute_batch] Workflow complete!")

        return batch_id, results_dict
