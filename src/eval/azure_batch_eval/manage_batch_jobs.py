"""
Inspect and manage Azure OpenAI Batch API jobs and uploaded files for a given
batch base directory (the same --base-dir / result directory used by
submit_batch_eval.py).

Usage (run from src/eval/):
    python -m azure_batch_eval.manage_batch_jobs list-pending --base-dir /path/to/result/...
    python -m azure_batch_eval.manage_batch_jobs resume --base-dir /path/to/result/...
    python -m azure_batch_eval.manage_batch_jobs status batch_abc123 --base-dir ...
    python -m azure_batch_eval.manage_batch_jobs cancel batch_abc123 --base-dir ...
    python -m azure_batch_eval.manage_batch_jobs list-all --base-dir ...
    python -m azure_batch_eval.manage_batch_jobs structure --base-dir ...
    python -m azure_batch_eval.manage_batch_jobs list-files --base-dir ...
    python -m azure_batch_eval.manage_batch_jobs delete-files --base-dir ... [--force]
    python -m azure_batch_eval.manage_batch_jobs delete-expired --base-dir ... [--execute]
    python -m azure_batch_eval.manage_batch_jobs cancel-all --base-dir ... [--force]

Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.
"""

import os
import sys
import argparse
from datetime import datetime

from openai import AzureOpenAI

from azure_batch_eval.batch_manager import BatchMathManager


def list_pending_jobs(batch_manager: BatchMathManager):
    pending = batch_manager.get_pending_jobs()
    if not pending:
        print("No pending batch jobs found.")
        return

    print(f"\n{len(pending)} pending batch job(s):\n")
    for job in pending:
        counts = job.get("request_counts", {})
        print(f"Batch ID: {job['batch_id']}")
        print(f"  Status: {job['status']}")
        print(f"  Request file: {job['request_file']}")
        print(f"  Progress: {counts.get('completed', 0)}/{counts.get('total', 0)} completed, "
              f"{counts.get('failed', 0)} failed")
        print(f"  Last updated: {job['last_updated']}")
        print()


def resume_pending_jobs(batch_manager: BatchMathManager, check_interval: int = 5):
    print("\nResuming monitoring of pending batch jobs...")
    results = batch_manager.resume_pending_jobs(check_interval=check_interval, quiet=False)

    print(f"\n{'='*60}\nCOMPLETED MONITORING\n{'='*60}\n")
    if results:
        print(f"Successfully completed {len(results)} batch jobs.")
        for batch_id, outputs in results.items():
            print(f"  - {batch_id}: {len(outputs)} outputs")
    else:
        print("No jobs completed.")


def check_job_status(batch_manager: BatchMathManager, batch_id: str):
    try:
        b = batch_manager.client.batches.retrieve(batch_id)

        print(f"\n{'='*60}\nBATCH JOB STATUS\n{'='*60}\n")
        print(f"Batch ID: {b.id}")
        print(f"Status: {b.status}")
        print(f"Created at: {datetime.fromtimestamp(b.created_at)}")
        if b.completed_at:
            print(f"Completed at: {datetime.fromtimestamp(b.completed_at)}")

        if b.request_counts:
            c = b.request_counts
            print(f"\nProgress:\n  Total: {c.total}\n  Completed: {c.completed}\n  Failed: {c.failed}")
            if c.total:
                print(f"  Success rate: {(c.completed / c.total * 100):.2f}%")

        print(f"\nInput file ID: {b.input_file_id}")
        print(f"Output file ID: {b.output_file_id or 'N/A'}")
        print(f"Error file ID: {b.error_file_id or 'N/A'}")

        if b.errors and b.errors.data:
            print("\nErrors:")
            for error in b.errors.data:
                print(f"  - {error.code}: {error.message}")

    except Exception as e:
        print(f"Error retrieving batch job status: {e}")


def cancel_job(batch_manager: BatchMathManager, batch_id: str):
    try:
        response = batch_manager.client.batches.cancel(batch_id)
        print(f"\nBatch job {batch_id} is being cancelled. Status: {response.status}")
    except Exception as e:
        print(f"Error cancelling batch job: {e}")


def list_all_jobs(batch_manager: BatchMathManager, limit: int = 20):
    print(f"\n{'='*60}\nALL BATCH JOBS (Most Recent First)\n{'='*60}\n")

    all_jobs = sorted(batch_manager.client.batches.list(limit=limit), key=lambda j: j.created_at)[-limit:]
    if not all_jobs:
        print("No batch jobs found.")
        return

    for i, job in enumerate(all_jobs, 1):
        print(f"{i}. Batch ID: {job.id}")
        print(f"   Status: {job.status}")
        print(f"   Created: {datetime.fromtimestamp(job.created_at)}")
        if job.completed_at:
            print(f"   Completed: {datetime.fromtimestamp(job.completed_at)}")
        if job.request_counts and job.request_counts.total > 0:
            c = job.request_counts
            print(f"   Progress: {c.completed}/{c.total} completed, {c.failed} failed")
        if job.errors and job.errors.data:
            for error in job.errors.data:
                print(f"   Error: {error.code} - {error.message}")
        print()


def show_directory_structure(batch_manager: BatchMathManager):
    print(f"\n{'='*60}\nBATCH OPERATIONS DIRECTORY STRUCTURE\n{'='*60}\n")
    print(f"Base directory: {batch_manager.base_dir}\n")

    requests_count = len(list(batch_manager.requests_dir.rglob("*.jsonl")))
    results_count = len(list(batch_manager.results_dir.rglob("*.jsonl")))
    status_count = len(list(batch_manager.status_dir.rglob("*.json")))

    print(f"requests/  ({requests_count} files)")
    print(f"results/   ({results_count} files)")
    print(f"status/    ({status_count} files)")

    timestamp_dirs = sorted(d for d in batch_manager.requests_dir.iterdir() if d.is_dir())
    if timestamp_dirs:
        print("\nTimestamp directories:")
        for ts_dir in timestamp_dirs:
            files = list(ts_dir.glob("*.jsonl"))
            print(f"  {ts_dir.name}/ ({len(files)} request file(s))")


def list_uploaded_files(batch_manager: BatchMathManager, limit: int = 20):
    print(f"\n{'='*60}\nUPLOADED FILES\n{'='*60}\n")

    files = batch_manager.list_files(purpose="batch", limit=limit)
    if not files:
        print("No files found.")
        return

    for i, f in enumerate(files, 1):
        print(f"{i}. File ID: {f['id']}")
        print(f"   Filename: {f['filename']}")
        print(f"   Size: {f['bytes']:,} bytes")
        print(f"   Created: {f['created_at']}")
        print(f"   Expires: {f['expires_at'] or 'Never'}")
        print(f"   Status: {f['status']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Manage Azure OpenAI Batch API jobs and files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        choices=["list-pending", "resume", "status", "cancel", "list-all", "structure",
                 "list-files", "delete-files", "delete-expired", "cancel-all"],
        help="Command to execute",
    )
    parser.add_argument("batch_id", nargs="?", help="Batch ID (required for 'status' and 'cancel')")
    parser.add_argument("--base-dir", type=str, required=True, help="Batch base directory (requests/results/status)")
    parser.add_argument("--deployment", type=str, default=os.getenv("AZURE_DEPLOYMENT", "gpt-4.1"),
                        help="Azure OpenAI deployment name")
    parser.add_argument("--check-interval", type=int, default=60, help="Seconds between status checks")
    parser.add_argument("--limit", type=int, default=20, help="Limit for listing jobs/files")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--execute", action="store_true", help="Actually delete (for delete-expired)")

    args = parser.parse_args()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        print("Error: Missing required environment variable(s): AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
        sys.exit(1)

    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    batch_manager = BatchMathManager(client=client, base_dir=args.base_dir, deployment=args.deployment)

    if args.command in ("status", "cancel") and not args.batch_id:
        print(f"Error: batch_id is required for '{args.command}' command")
        sys.exit(1)

    {
        "list-pending": lambda: list_pending_jobs(batch_manager),
        "resume": lambda: resume_pending_jobs(batch_manager, check_interval=args.check_interval),
        "status": lambda: check_job_status(batch_manager, args.batch_id),
        "cancel": lambda: cancel_job(batch_manager, args.batch_id),
        "list-all": lambda: list_all_jobs(batch_manager, limit=args.limit),
        "structure": lambda: show_directory_structure(batch_manager),
        "list-files": lambda: list_uploaded_files(batch_manager, limit=args.limit),
        "delete-files": lambda: batch_manager.delete_all_files(purpose="batch", confirm=not args.force),
        "delete-expired": lambda: batch_manager.delete_expired_files(dry_run=not args.execute),
        "cancel-all": lambda: batch_manager.cancel_all_pending_batches(confirm=not args.force),
    }[args.command]()


if __name__ == "__main__":
    main()
