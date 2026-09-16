"""
Azure OpenAI Batch API utilities for math tasks.

This module provides functions to:
1. Create JSONL batch request files
2. Upload files to Azure OpenAI
3. Submit batch jobs
4. Monitor batch job status
5. Retrieve and process results
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from openai import AzureOpenAI, BadRequestError


class BatchMathManager:
    """Manages Azure OpenAI Batch API operations for math evaluation tasks."""

    def __init__(
        self,
        client: AzureOpenAI,
        base_dir: str,
        deployment: str,
    ):
        """
        Args:
            client: Azure OpenAI client instance
            base_dir: Base directory for batch operations (requests/results/status subfolders)
            deployment: Azure OpenAI deployment name
        """
        self.client = client
        self.base_dir = Path(base_dir)
        self.deployment = deployment

        self.requests_dir = self.base_dir / "requests"
        self.results_dir = self.base_dir / "results"
        self.status_dir = self.base_dir / "status"

        for dir_path in [self.requests_dir, self.results_dir, self.status_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_batch_request_file(
        self,
        texts: List[str],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timestamp: Optional[str] = None,
    ) -> Tuple[Path, str]:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        timestamp_dir = self.requests_dir / timestamp
        timestamp_dir.mkdir(parents=True, exist_ok=True)

        request_file = timestamp_dir / "request.jsonl"

        with open(request_file, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                request = {
                    "custom_id": f"{self.deployment}_{i}",
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": self.deployment,
                        "messages": [{"role": "user", "content": text}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                }
                f.write(json.dumps(request, ensure_ascii=False) + '\n')

        print(f"Created batch request file: {request_file}")
        print(f"  - Model : {self.deployment}")
        print(f"  - Number of requests: {len(texts)}")

        return request_file, timestamp

    def upload_batch_file(self, request_file: Path, expires_in_days: int = 30) -> str:
        """Uploads a batch request file. Returns the Azure OpenAI file ID."""
        print(f"Uploading file: {request_file}")

        with open(request_file, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose="batch",
                extra_body={
                    "expires_after": {
                        "seconds": expires_in_days * 86400,
                        "anchor": "created_at",
                    }
                },
            )

        print(f"  - File ID: {file_response.id}")
        print(f"  - Status: {file_response.status}")
        print(f"  - Expires at: {datetime.fromtimestamp(file_response.expires_at) if file_response.expires_at else 'Not set'}")

        return file_response.id

    def submit_batch_job(
        self,
        file_id: str,
        request_file: Path,
        max_retries: int = 10,
        initial_delay: int = 120,
        output_expires_in_days: int = 30,
    ) -> str:
        """Submits a batch job, retrying with exponential backoff on token-limit errors."""
        retries = 0
        delay = initial_delay

        while True:
            try:
                batch_response = self.client.batches.create(
                    input_file_id=file_id,
                    endpoint="/chat/completions",
                    completion_window="24h",
                    extra_body={
                        "output_expires_after": {
                            "seconds": output_expires_in_days * 86400,
                            "anchor": "created_at",
                        }
                    },
                )

                print(f"Batch job created successfully after {retries} retries")
                print(f"  - Batch ID: {batch_response.id}")
                print(f"  - Status: {batch_response.status}")

                self._save_status(request_file, batch_response.id, batch_response)
                return batch_response.id

            except BadRequestError as e:
                error_message = str(e)
                if 'token_limit_exceeded' not in error_message:
                    print(f"Encountered non-token-limit error: {error_message}")
                    raise

                retries += 1
                if retries >= max_retries:
                    print(f"Maximum retries ({max_retries}) reached. Giving up.")
                    raise

                print(f"Token limit exceeded. Waiting {delay}s before retry {retries}/{max_retries}...")
                time.sleep(delay)
                delay *= 2

    def monitor_batch_job(
        self,
        batch_id: str,
        request_file: Path,
        check_interval: int = 5,
        quiet: bool = False,
    ) -> Dict:
        """Polls a batch job until it reaches a terminal state. Returns the final batch response."""
        status = "validating"

        while status not in ("completed", "failed", "canceled", "cancelled", "expired"):
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status

            self._save_status(request_file, batch_id, batch_response)

            if not quiet:
                print(f"{datetime.now()} Batch ID: {batch_id}, Status: {status}")
                if batch_response.request_counts:
                    counts = batch_response.request_counts
                    print(f"  - Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")

            time.sleep(check_interval)

        if batch_response.status == "failed":
            print("Batch job failed!")
            if batch_response.errors:
                for error in batch_response.errors.data:
                    print(f"  - Error code: {error.code}")
                    print(f"  - Message: {error.message}")

        return batch_response

    def retrieve_batch_results(self, batch_response, request_file) -> List[str]:
        """Retrieves batch results, preserving request order (empty string for failures)."""
        output_file_id = batch_response.output_file_id or batch_response.error_file_id
        if not output_file_id:
            raise ValueError("No output file available for this batch job")

        request_file_path = Path(request_file)
        total_requests = sum(1 for line in open(request_file_path, "r", encoding="utf-8") if line.strip())

        translations = [""] * total_requests
        failed_indices = []

        result_file = self.client.files.content(batch_response.output_file_id)
        result_str = result_file.read().decode("utf-8")

        for line in result_str.strip().split("\n"):
            if not line.strip():
                continue

            result_obj = json.loads(line)
            custom_id = result_obj.get("custom_id", "")

            try:
                index = int(custom_id.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                print(f"Could not parse index from custom_id: {custom_id}")
                continue

            if result_obj.get("error"):
                failed_indices.append((custom_id, result_obj["error"].get("message", "Unknown error")))
                continue

            body = result_obj.get("response", {}).get("body", {})
            choices = body.get("choices")
            if not choices:
                failed_indices.append((custom_id, "No choices in response"))
                continue

            choice = choices[0]
            if choice.get("finish_reason") == "content_filter":
                filtered = [
                    cat for cat, res in choice.get("content_filter_results", {}).items()
                    if isinstance(res, dict) and res.get("filtered", False)
                ]
                failed_indices.append((custom_id, f"Content filtered due to policy violation: {filtered}"))
                continue

            content = choice.get("message", {}).get("content", "").strip()
            if content and index < total_requests:
                translations[index] = content

        if failed_indices:
            print(f"{len(failed_indices)} request errors occurred:")
            for custom_id, error_msg in failed_indices[:5]:
                print(f"  - {custom_id}: {error_msg}")

        print(f"Successfully retrieved {len(translations)} results")
        return translations

    def _save_status(self, request_file: Path, batch_id: str, batch_response):
        relative_path = request_file.relative_to(self.requests_dir)
        status_dir = self.status_dir / relative_path.parent
        status_dir.mkdir(parents=True, exist_ok=True)

        status_file = status_dir / f"{request_file.stem}.json"
        counts = batch_response.request_counts
        status_data = {
            "batch_id": batch_id,
            "request_file": str(request_file),
            "status": batch_response.status,
            "created_at": batch_response.created_at,
            "last_updated": datetime.now().isoformat(),
            "request_counts": {
                "total": counts.total,
                "completed": counts.completed,
                "failed": counts.failed,
            } if counts else {},
            "input_file_id": batch_response.input_file_id,
            "output_file_id": batch_response.output_file_id or "",
            "error_file_id": batch_response.error_file_id or "",
        }

        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)

    def compute_math_batch(
        self,
        texts: List[str],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timestamp: Optional[str] = None,
        check_interval: int = 5,
        quiet: bool = False,
    ) -> Tuple[List[str], Dict]:
        """End-to-end batch workflow: create, upload, submit, monitor, retrieve.

        Returns:
            Tuple of (outputs, usage_stats).
        """
        request_file, timestamp = self.create_batch_request_file(
            texts=texts, temperature=temperature, max_tokens=max_tokens, timestamp=timestamp,
        )
        file_id = self.upload_batch_file(request_file)
        batch_id = self.submit_batch_job(file_id, request_file)
        batch_response = self.monitor_batch_job(
            batch_id=batch_id, request_file=request_file, check_interval=check_interval, quiet=quiet,
        )
        results = self.retrieve_batch_results(batch_response=batch_response, request_file=request_file)

        counts = batch_response.request_counts
        usage_stats = {
            "total_requests": counts.total if counts else 0,
            "completed_requests": counts.completed if counts else 0,
            "failed_requests": counts.failed if counts else 0,
        }
        return results, usage_stats

    def get_pending_jobs(self) -> List[Dict]:
        """Returns status dicts for all batch jobs that haven't reached a terminal state."""
        pending_jobs = []
        for status_file in self.status_dir.rglob("*.json"):
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            if status_data['status'] not in ('completed', 'failed', 'cancelled', 'expired'):
                pending_jobs.append(status_data)
        return pending_jobs

    def resume_pending_jobs(self, check_interval: int = 5, quiet: bool = False) -> Dict[str, List[str]]:
        """Resumes monitoring of all pending batch jobs to completion. Returns batch_id -> outputs."""
        pending_jobs = self.get_pending_jobs()
        if not pending_jobs:
            print("No pending batch jobs found.")
            return {}

        print(f"Found {len(pending_jobs)} pending batch jobs. Resuming...")
        results = {}

        for job in pending_jobs:
            batch_id = job['batch_id']
            request_file = Path(job['request_file'])
            print(f"\nResuming batch job: {batch_id}")

            batch_response = self.monitor_batch_job(
                batch_id=batch_id, request_file=request_file, check_interval=check_interval, quiet=quiet,
            )

            if batch_response.status == 'completed':
                results[batch_id] = self.retrieve_batch_results(
                    batch_response=batch_response, request_file=request_file,
                )

        return results

    def list_files(self, purpose: str = "batch", limit: int = 100) -> List[Dict]:
        """Lists files uploaded to Azure OpenAI, filtered by purpose."""
        files = []
        for file in self.client.files.list(purpose=purpose):
            files.append({
                "id": file.id,
                "filename": file.filename,
                "bytes": file.bytes,
                "created_at": datetime.fromtimestamp(file.created_at),
                "expires_at": datetime.fromtimestamp(file.expires_at) if file.expires_at else None,
                "purpose": file.purpose,
                "status": file.status,
            })
            if len(files) >= limit:
                break
        return files

    def delete_file(self, file_id: str) -> bool:
        try:
            self.client.files.delete(file_id)
            print(f"Deleted file: {file_id}")
            return True
        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
            return False

    def delete_all_files(self, purpose: str = "batch", confirm: bool = True) -> Dict[str, int]:
        files = self.list_files(purpose=purpose, limit=10000)
        if not files:
            print("No files found to delete.")
            return {"deleted": 0, "failed": 0}

        print(f"\nWARNING: About to delete {len(files)} files!")
        if confirm and input(f"Are you sure you want to delete {len(files)} files? (yes/no): ").lower() not in ('yes', 'y'):
            print("Deletion cancelled.")
            return {"deleted": 0, "failed": 0}

        deleted = sum(self.delete_file(f["id"]) for f in files)
        failed = len(files) - deleted
        print(f"\nDeleted: {deleted}, Failed: {failed}")
        return {"deleted": deleted, "failed": failed}

    def delete_expired_files(self, dry_run: bool = True) -> Dict[str, int]:
        files = self.list_files(purpose="batch", limit=10000)
        now = datetime.now()
        expired_files = [f for f in files if f["expires_at"] and f["expires_at"] < now]

        if not expired_files:
            print("No expired files found.")
            return {"would_delete": 0, "deleted": 0}

        if dry_run:
            print(f"\n[DRY RUN] Would delete {len(expired_files)} expired files:")
            for f in expired_files[:10]:
                print(f"  - {f['id']}: {f['filename']} (expired: {f['expires_at']})")
            if len(expired_files) > 10:
                print(f"  ... and {len(expired_files) - 10} more")
            return {"would_delete": len(expired_files), "deleted": 0}

        deleted = sum(self.delete_file(f["id"]) for f in expired_files)
        print(f"\nDeleted {deleted} expired files.")
        return {"deleted": deleted, "failed": len(expired_files) - deleted}

    def cancel_all_pending_batches(self, confirm: bool = True) -> Dict[str, int]:
        pending_jobs = self.get_pending_jobs()
        if not pending_jobs:
            print("No pending batch jobs found.")
            return {"cancelled": 0, "failed": 0}

        print(f"\nWARNING: About to cancel {len(pending_jobs)} pending batch jobs!")
        if confirm and input(f"Are you sure you want to cancel {len(pending_jobs)} batch jobs? (yes/no): ").lower() not in ('yes', 'y'):
            print("Cancellation aborted.")
            return {"cancelled": 0, "failed": 0}

        cancelled, failed = 0, 0
        for job in pending_jobs:
            try:
                self.client.batches.cancel(job['batch_id'])
                print(f"Cancelled batch: {job['batch_id']}")
                cancelled += 1
            except Exception as e:
                print(f"Error cancelling batch {job['batch_id']}: {e}")
                failed += 1

        print(f"\nCancelled: {cancelled}, Failed: {failed}")
        return {"cancelled": cancelled, "failed": failed}
