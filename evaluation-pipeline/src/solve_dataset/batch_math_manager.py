"""
Azure OpenAI Batch API utilities for math tasks.

This module provides functions to:
1. Create JSONL batch request files
2. Upload files to Azure OpenAI
3. Submit batch jobs
4. Monitor batch job status
5. Retrieve and process results
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from openai import AzureOpenAI, BadRequestError

class BatchMathManager:
    """Manages Azure OpenAI Batch translation operations."""
    
    def __init__(
        self,
        client: AzureOpenAI,
        base_dir: str = "/project/aip-davlan/shared/LanguagePretrain/data/datasets/openai-batch",
        deployment: str = "gpt-4.1",
    ):
        """
        Initialize the batch translation manager.
        
        Args:
            client: Azure OpenAI client instance
            base_dir: Base directory for batch operations
            deployment: Azure OpenAI deployment name
        """
        self.client = client
        self.base_dir = Path(base_dir)
        self.deployment = deployment
        
        # Create directory structure
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
                
        # Create timestamp directory structure
        timestamp_dir = self.requests_dir / timestamp 
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create request file
        request_file = timestamp_dir / f"result.jsonl"

        with open(request_file, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                request = {
                    "custom_id": f"{self.deployment}_{i}",
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": self.deployment,
                        "messages": [
                            {"role": "user", "content": text}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        print(f"Created batch request file: {request_file}")
        print(f"  - Model : {self.deployment}")
        print(f"  - Number of requests: {len(texts)}")
        
        return request_file, timestamp
    
    def upload_batch_file(
        self,
        request_file: Path,
        expires_in_days: int = 30,
    ) -> str:
        """
        Upload a batch request file to Azure OpenAI.
        
        Args:
            request_file: Path to the JSONL request file
            expires_in_days: Number of days before file expires (14-30, default: 30)
        
        Returns:
            File ID from Azure OpenAI
        """
        print(f"Uploading file: {request_file}")
        
        with open(request_file, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose="batch",
                extra_body={
                    "expires_after": {
                        "seconds": expires_in_days * 86400,  # days to seconds
                        "anchor": "created_at"
                    }
                },
                extra_headers={'x-policy-id': 'CustomContentFilter333'}
            )
        
        file_id = file_response.id
        print(f"  - File ID: {file_id}")
        print(f"  - Status: {file_response.status}")
        print(f"  - Expires at: {datetime.fromtimestamp(file_response.expires_at) if file_response.expires_at else 'Not set'}")
        
        return file_id
    
    def submit_batch_job(
        self,
        file_id: str,
        request_file: Path,
        max_retries: int = 10,
        initial_delay: int = 120,
        output_expires_in_days: int = 30,
    ) -> str:
        """
        Submit a batch job with retry logic for token limit errors.
        
        Args:
            file_id: Azure OpenAI file ID
            request_file: Path to the request file (for status tracking)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds for exponential backoff
            output_expires_in_days: Number of days before output expires (14-30, default: 30)
        
        Returns:
            Batch job ID
        """
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
                            "anchor": "created_at"
                        }
                    }
                )
                
                batch_id = batch_response.id
                print(f"✅ Batch job created successfully after {retries} retries")
                print(f"  - Batch ID: {batch_id}")
                print(f"  - Status: {batch_response.status}")
                
                # Save initial status
                self._save_status(request_file, batch_id, batch_response)
                
                return batch_id
                
            except BadRequestError as e:
                error_message = str(e)
                
                if 'token_limit_exceeded' in error_message:
                    retries += 1
                    if retries >= max_retries:
                        print(f"❌ Maximum retries ({max_retries}) reached. Giving up.")
                        raise
                    
                    print(f"⏳ Token limit exceeded. Waiting {delay} seconds before retry {retries}/{max_retries}...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Encountered non-token limit error: {error_message}")
                    raise
    
    def monitor_batch_job(
        self,
        batch_id: str,
        request_file: Path,
        check_interval: int = 5,
        quiet: bool = False,
    ) -> Dict:
        """
        Monitor a batch job until completion.
        
        Args:
            batch_id: Batch job ID
            request_file: Path to the request file (for status tracking)
            check_interval: Seconds between status checks
            quiet: If True, suppress progress output
        
        Returns:
            Final batch response dict
        """
        status = "validating"
        
        while status not in ("completed", "failed", "canceled", "cancelled", "expired"):
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status
            
            # Save status update
            self._save_status(request_file, batch_id, batch_response)
            
            if not quiet:
                print(f"{datetime.now()} Batch ID: {batch_id}, Status: {status}")
                if batch_response.request_counts:
                    counts = batch_response.request_counts
                    print(f"  - Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")

            time.sleep(check_interval)
        
        if batch_response.status == "failed":
            print(f"❌ Batch job failed!")
            if batch_response.errors:
                for error in batch_response.errors.data:
                    print(f"  - Error code: {error.code}")
                    print(f"  - Message: {error.message}")
        
        return batch_response
    
    def retrieve_batch_results(
        self,
        batch_response,
        request_file: str | Path
    ) -> list[str]:
        """
        Retrieve and parse batch translation results, maintaining order with empty strings for failures.
        
        Args:
            batch_response: Batch response object from Azure OpenAI
            request_file: Path to the original request file (for counting)
            target_lang: Target language code
        
        Returns:
            List of translated texts in the same order as requests, with empty strings for failures
        """
        output_file_id = batch_response.output_file_id
        
        if not output_file_id:
            output_file_id = batch_response.error_file_id
            if not output_file_id:
                raise ValueError("No output file available for this batch job")
        
        # Count total requests to determine output size
        request_file_path = Path(request_file)
        total_requests = 0
        with open(request_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total_requests += 1
        
        # Initialize result list with empty strings
        translations = [""] * total_requests
        failed_indices = []
        
        # Download and parse results
        result_file = self.client.files.content(batch_response.output_file_id)
        result_bytes = result_file.read()
        result_str = result_bytes.decode("utf-8")
        
        for line in result_str.strip().split("\n"):
            if not line.strip():
                continue
            
            result_obj = json.loads(line)
            custom_id = result_obj.get("custom_id", "")
            
            # Extract index from custom_id (format: "DomainName_LanguageName_INDEX")
            try:
                index = int(custom_id.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                print(f"⚠️ Could not parse index from custom_id: {custom_id}")
                continue
            
            # Check for errors
            if result_obj.get("error"):
                error_info = result_obj["error"]
                failed_indices.append((custom_id, error_info.get("message", "Unknown error")))
                # translations[index] already initialized to ""
                continue
            
            # Extract translation
            response = result_obj.get("response", {})
            body = response.get("body", {})
            
            # Check for content filtering
            if body.get("choices") and len(body["choices"]) > 0:
                choice = body["choices"][0]
                finish_reason = choice.get("finish_reason")
                
                if finish_reason == "content_filter":
                    content_filter_results = choice.get("content_filter_results", {})
                    filtered_categories = [
                        cat for cat, result in content_filter_results.items()
                        if isinstance(result, dict) and result.get("filtered", False)
                    ]
                    failed_indices.append((custom_id, f"Content filtered due to policy violation: {filtered_categories}"))
                    # translations[index] already initialized to ""
                    continue
                
                message = choice.get("message", {})
                translation = message.get("content", "").strip()
                
                if translation and index < total_requests:
                    translations[index] = translation
            else:
                failed_indices.append((custom_id, "No choices in response"))
        
        # Report any errors encountered during processing
        if failed_indices:
            print(f"⚠️ {len(failed_indices)} translation errors occurred:")
            for custom_id, error_msg in failed_indices[:5]:  # Show first 5 errors
                print(f"  - {custom_id}: {error_msg}")
        
        print(f"Successfully retrieved {len(translations)} translations")
        
        return translations
    
    def _save_status(
        self,
        request_file: Path,
        batch_id: str,
        batch_response,
    ):
        """
        Save batch job status to a JSON file.
        
        Args:
            request_file: Path to the request file
            batch_id: Batch job ID
            batch_response: Batch response object
        """
        relative_path = request_file.relative_to(self.requests_dir)
        status_dir = self.status_dir / relative_path.parent
        status_dir.mkdir(parents=True, exist_ok=True)
        
        status_file = status_dir / f"{request_file.stem}.json"
        
        status_data = {
            "batch_id": batch_id,
            "request_file": str(request_file),
            "status": batch_response.status,
            "created_at": batch_response.created_at,
            "last_updated": datetime.now().isoformat(),
            "request_counts": {
                "total": batch_response.request_counts.total if batch_response.request_counts else 0,
                "completed": batch_response.request_counts.completed if batch_response.request_counts else 0,
                "failed": batch_response.request_counts.failed if batch_response.request_counts else 0,
            } if batch_response.request_counts else {},
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
        """
        Complete batch translation workflow: create, upload, submit, monitor, retrieve.

        Returns:
            Tuple of (translations, usage_stats)
        """

        # Step 1: Create batch request file
        request_file, timestamp = self.create_batch_request_file(
            texts=texts,
            temperature=temperature,
            max_tokens=max_tokens,
            timestamp=timestamp,
        )
        
        # Step 2: Upload file
        file_id = self.upload_batch_file(request_file)
        
        # Step 3: Submit batch job
        batch_id = self.submit_batch_job(file_id, request_file)
        
        # Step 4: Monitor batch job
        batch_response = self.monitor_batch_job(
            batch_id=batch_id,
            request_file=request_file,
            check_interval=check_interval,
            quiet=quiet,
        )
        
        # Step 5: Retrieve results
        results = self.retrieve_batch_results(
            batch_response=batch_response,
            request_file=request_file,
        )
        
        # Prepare usage stats (if available)
        usage_stats = {
            "total_requests": batch_response.request_counts.total if batch_response.request_counts else 0,
            "completed_requests": batch_response.request_counts.completed if batch_response.request_counts else 0,
            "failed_requests": batch_response.request_counts.failed if batch_response.request_counts else 0,
        }
        
        return results, usage_stats
    
    def get_pending_jobs(self) -> List[Dict]:
        """
        Get all pending batch jobs from status files.
        
        Returns:
            List of status dictionaries for pending jobs
        """
        pending_jobs = []
        
        for status_file in self.status_dir.rglob("*.json"):
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            
            if status_data['status'] not in ('completed', 'failed', 'cancelled', 'expired'):
                pending_jobs.append(status_data)
        
        return pending_jobs
    
    def resume_pending_jobs(
        self,
        check_interval: int = 5,
        quiet: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Resume monitoring of all pending batch jobs.
        
        Args:
            check_interval: Seconds between status checks
            quiet: If True, suppress progress output
        
        Returns:
            Dictionary mapping batch_id to translations
        """
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
            
            # Monitor to completion
            batch_response = self.monitor_batch_job(
                batch_id=batch_id,
                request_file=request_file,
                check_interval=check_interval,
                quiet=quiet,
            )
            
            if batch_response.status == 'completed':
                # Extract target language from filename
                target_lang = request_file.stem
                
                translations = self.retrieve_batch_results(
                    batch_response=batch_response,
                    request_file=request_file,
                    target_lang=target_lang,
                )
                
                results[batch_id] = translations
        
        return results
    
    def list_files(self, purpose: str = "batch", limit: int = 100) -> List[Dict]:
        """
        List all files uploaded to Azure OpenAI.
        
        Args:
            purpose: Filter by file purpose (default: "batch")
            limit: Maximum number of files to retrieve
        
        Returns:
            List of file metadata dictionaries
        """
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
        """
        Delete a specific file from Azure OpenAI.
        
        Args:
            file_id: File ID to delete
        
        Returns:
            True if successful
        """
        try:
            self.client.files.delete(file_id)
            print(f"✅ Deleted file: {file_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting file {file_id}: {e}")
            return False
    
    def delete_all_files(self, purpose: str = "batch", confirm: bool = True) -> Dict[str, int]:
        """
        Delete all uploaded files.
        
        Args:
            purpose: Filter by file purpose (default: "batch")
            confirm: If True, requires confirmation before deletion
        
        Returns:
            Dictionary with counts of deleted and failed files
        """
        files = self.list_files(purpose=purpose, limit=10000)
        
        if not files:
            print("No files found to delete.")
            return {"deleted": 0, "failed": 0}
        
        print(f"\n{'='*80}")
        print(f"WARNING: About to delete {len(files)} files!")
        print(f"{'='*80}")
        
        if confirm:
            response = input(f"\nAre you sure you want to delete {len(files)} files? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Deletion cancelled.")
                return {"deleted": 0, "failed": 0}
        
        deleted = 0
        failed = 0
        
        print(f"\nDeleting files...")
        for file_info in files:
            if self.delete_file(file_info["id"]):
                deleted += 1
            else:
                failed += 1
        
        print(f"\n{'='*80}")
        print(f"DELETION SUMMARY")
        print(f"{'='*80}")
        print(f"Deleted: {deleted}")
        print(f"Failed: {failed}")
        
        return {"deleted": deleted, "failed": failed}
    
    def delete_expired_files(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Delete files that have already expired.
        
        Args:
            dry_run: If True, only show what would be deleted without deleting
        
        Returns:
            Dictionary with counts of deleted files
        """
        from datetime import datetime
        
        files = self.list_files(purpose="batch", limit=10000)
        now = datetime.now()
        
        expired_files = []
        for file_info in files:
            if file_info["expires_at"] and file_info["expires_at"] < now:
                expired_files.append(file_info)
        
        if not expired_files:
            print("No expired files found.")
            return {"would_delete": 0, "deleted": 0}
        
        print(f"\nFound {len(expired_files)} expired files.")
        
        if dry_run:
            print("\n[DRY RUN MODE - No files will be deleted]")
            for file_info in expired_files[:10]:
                print(f"  - {file_info['id']}: {file_info['filename']} (expired: {file_info['expires_at']})")
            if len(expired_files) > 10:
                print(f"  ... and {len(expired_files) - 10} more")
            return {"would_delete": len(expired_files), "deleted": 0}
        
        deleted = 0
        failed = 0
        
        print("\nDeleting expired files...")
        for file_info in expired_files:
            if self.delete_file(file_info["id"]):
                deleted += 1
            else:
                failed += 1
        
        print(f"\nDeleted {deleted} expired files.")
        return {"deleted": deleted, "failed": failed}
    
    def cancel_all_pending_batches(self, confirm: bool = True) -> Dict[str, int]:
        """
        Cancel all pending batch jobs.
        
        Args:
            confirm: If True, requires confirmation before cancellation
        
        Returns:
            Dictionary with counts of cancelled and failed batches
        """
        pending_jobs = self.get_pending_jobs()
        
        if not pending_jobs:
            print("No pending batch jobs found.")
            return {"cancelled": 0, "failed": 0}
        
        print(f"\n{'='*80}")
        print(f"WARNING: About to cancel {len(pending_jobs)} pending batch jobs!")
        print(f"{'='*80}")
        
        if confirm:
            response = input(f"\nAre you sure you want to cancel {len(pending_jobs)} batch jobs? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Cancellation aborted.")
                return {"cancelled": 0, "failed": 0}
        
        cancelled = 0
        failed = 0
        
        print(f"\nCancelling batch jobs...")
        for job in pending_jobs:
            try:
                self.client.batches.cancel(job['batch_id'])
                print(f"✅ Cancelled batch: {job['batch_id']}")
                cancelled += 1
            except Exception as e:
                print(f"❌ Error cancelling batch {job['batch_id']}: {e}")
                failed += 1
        
        print(f"\n{'='*80}")
        print(f"CANCELLATION SUMMARY")
        print(f"{'='*80}")
        print(f"Cancelled: {cancelled}")
        print(f"Failed: {failed}")
        
        return {"cancelled": cancelled, "failed": failed}
        print(f"{i}. Batch ID: {job['batch_id']}")
        print(f"   Status: {job['status']}")
        print(f"   Timestamp: {timestamp_dir}")
        print(f"   Domain: {domain}")
        print(f"   Language: {language}")
        print(f"   Progress: {job['request_counts']['completed']}/{job['request_counts']['total']} completed, "
              f"{job['request_counts']['failed']} failed")
        print(f"   Last updated: {job['last_updated']}")
        print(f"   Request file: {request_file}")
        print()


def resume_pending_jobs(batch_manager: BatchMathManager, check_interval: int = 5):
    """Resume monitoring of all pending batch jobs."""
    print("\nResuming monitoring of pending batch jobs...")
    results = batch_manager.resume_pending_jobs(check_interval=check_interval, quiet=False)
    
    print(f"\n{'='*100}")
    print(f"COMPLETED MONITORING")
    print(f"{'='*100}\n")
    
    if results:
        print(f"Successfully completed {len(results)} batch jobs.")
        for batch_id, translations in results.items():
            print(f"  - {batch_id}: {len(translations)} translations")
    else:
        print("No jobs completed.")


def check_job_status(batch_manager: BatchMathManager, batch_id: str):
    """Check the status of a specific batch job."""
    try:
        batch_response = batch_manager.client.batches.retrieve(batch_id)
        
        print(f"\n{'='*100}")
        print(f"BATCH JOB STATUS")
        print(f"{'='*100}\n")
        
        print(f"Batch ID: {batch_response.id}")
        print(f"Status: {batch_response.status}")
        print(f"Created at: {datetime.fromtimestamp(batch_response.created_at)}")
        
        if batch_response.completed_at:
            print(f"Completed at: {datetime.fromtimestamp(batch_response.completed_at)}")
        
        if batch_response.request_counts:
            counts = batch_response.request_counts
            print(f"\nProgress:")
            print(f"  Total: {counts.total}")
            print(f"  Completed: {counts.completed}")
            print(f"  Failed: {counts.failed}")
            print(f"  Success rate: {(counts.completed/counts.total*100):.2f}%")
        
        print(f"\nInput file ID: {batch_response.input_file_id}")
        print(f"Output file ID: {batch_response.output_file_id or 'N/A'}")
        print(f"Error file ID: {batch_response.error_file_id or 'N/A'}")

        if batch_response.errors and batch_response.errors.data:
            print(f"\nErrors:")
            for error in batch_response.errors.data:
                print(f"  - Code: {error.code}")
                print(f"    Message: {error.message}")
        
    except Exception as e:
        print(f"Error retrieving batch job status: {e}")


def cancel_job(batch_manager: BatchMathManager, batch_id: str):
    """Cancel a specific batch job."""
    try:
        response = batch_manager.client.batches.cancel(batch_id)
        print(f"\n✅ Batch job {batch_id} is being cancelled.")
        print(f"Status: {response.status}")
    except Exception as e:
        print(f"Error cancelling batch job: {e}")


def list_all_jobs(batch_manager: BatchMathManager, limit: int = 20):
    print(f"\n{'='*100}")
    print(f"ALL BATCH JOBS (Most Recent First)")
    print(f"{'='*100}\n")
    
    all_jobs = []
    for job in batch_manager.client.batches.list(limit=limit):
        all_jobs.append(job)
    
    if not all_jobs:
        print("No batch jobs found.")
        return
    
    all_jobs.sort(key=lambda x: x.created_at)
    all_jobs = all_jobs[-limit:]
    for i, job in enumerate(all_jobs, 1):
        print(f"{i}. Batch ID: {job.id}")
        print(f"   Status: {job.status}")
        print(f"   Created: {datetime.fromtimestamp(job.created_at)}")
        if job.completed_at:
            print(f"   Completed: {datetime.fromtimestamp(job.completed_at)}")
        if job.request_counts and job.request_counts.total > 0:
            counts = job.request_counts
            print(f"   Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")
        if job.errors and job.errors.data:
            for error in job.errors.data:
                print(f"   Error: {error.code} - {error.message}")
        print()

def show_directory_structure(batch_manager: BatchMathManager):
    """Show the directory structure of batch operations."""
    print(f"\n{'='*100}")
    print(f"BATCH OPERATIONS DIRECTORY STRUCTURE")
    print(f"{'='*100}\n")
    
    base_dir = batch_manager.base_dir
    print(f"Base directory: {base_dir}\n")
    
    # Count files in each directory
    requests_count = len(list(batch_manager.requests_dir.rglob("*.jsonl")))
    results_count = len(list(batch_manager.results_dir.rglob("*.jsonl")))
    status_count = len(list(batch_manager.status_dir.rglob("*.json")))
    
    print(f"📁 requests/  ({requests_count} files)")
    print(f"📁 results/   ({results_count} files)")
    print(f"📁 status/    ({status_count} files)")
    
    # Show timestamp directories
    timestamp_dirs = sorted([d for d in batch_manager.requests_dir.iterdir() if d.is_dir()])
    
    if timestamp_dirs:
        print(f"\nTimestamp directories:")
        for ts_dir in timestamp_dirs:
            domain_dirs = sorted([d for d in ts_dir.iterdir() if d.is_dir()])
            print(f"  📅 {ts_dir.name}/")
            for domain_dir in domain_dirs:
                files = list(domain_dir.glob("*.jsonl"))
                print(f"      📂 {domain_dir.name}/ ({len(files)} languages)")


def list_uploaded_files(batch_manager: BatchMathManager, limit: int = 20):
    """List all uploaded files in Azure OpenAI."""
    print(f"\n{'='*100}")
    print(f"UPLOADED FILES IN AZURE OPENAI")
    print(f"{'='*100}\n")
    
    files = batch_manager.list_files(purpose="batch", limit=limit)
    
    if not files:
        print("No files found.")
        return
    
    for i, file_info in enumerate(files, 1):
        print(f"{i}. File ID: {file_info['id']}")
        print(f"   Filename: {file_info['filename']}")
        print(f"   Size: {file_info['bytes']:,} bytes")
        print(f"   Created: {file_info['created_at']}")
        print(f"   Expires: {file_info['expires_at'] or 'Never'}")
        print(f"   Status: {file_info['status']}")
        print()


def delete_all_files(batch_manager: BatchMathManager, force: bool = False):
    """Delete all uploaded files."""
    batch_manager.delete_all_files(purpose="batch", confirm=not force)


def delete_expired_files(batch_manager: BatchMathManager, dry_run: bool = True):
    """Delete expired files."""
    batch_manager.delete_expired_files(dry_run=dry_run)


def cancel_all_pending(batch_manager: BatchMathManager, force: bool = False):
    """Cancel all pending batch jobs."""
    batch_manager.cancel_all_pending_batches(confirm=not force)


def main():
    import sys
    import argparse
    from dotenv import load_dotenv

    load_dotenv('/project/aip-davlan/shared/LanguagePretrain/code/src/data/.env')

    parser = argparse.ArgumentParser(
        description="Manage Azure OpenAI Batch translation jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all pending jobs
  python manage_batch_jobs.py list-pending
  
  # Resume monitoring of pending jobs
  python manage_batch_jobs.py resume
  
  # Check status of a specific job
  python manage_batch_jobs.py status batch_abc123
  
  # Cancel a specific job
  python manage_batch_jobs.py cancel batch_abc123
  
  # List all jobs (including completed)
  python manage_batch_jobs.py list-all
  
  # Show directory structure
  python manage_batch_jobs.py structure
  
  # List uploaded files
  python manage_batch_jobs.py list-files
  
  # Delete all uploaded files (with confirmation)
  python manage_batch_jobs.py delete-files
  
  # Delete all uploaded files (no confirmation)
  python manage_batch_jobs.py delete-files --force
  
  # Delete expired files (dry run)
  python manage_batch_jobs.py delete-expired
  
  # Delete expired files (actually delete)
  python manage_batch_jobs.py delete-expired --execute
  
  # Cancel all pending batch jobs
  python manage_batch_jobs.py cancel-all
        """
    )
    
    parser.add_argument(
        "command",
        choices=["list-pending", "resume", "status", "cancel", "list-all", "structure", 
                 "list-files", "delete-files", "delete-expired", "cancel-all"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "batch_id",
        nargs="?",
        help="Batch ID (required for 'status' and 'cancel' commands)"
    )
    
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between status checks (default: 60)"
    )
    
    parser.add_argument(
        "--deployment",
        type=str,
        default=os.getenv("AZURE_DEPLOYMENT", "gpt-4.1"),
        help="Azure OpenAI deployment name"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit for listing jobs/files (default: 20)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts for destructive operations"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute deletion (for delete-expired command)"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = AzureOpenAI(
        api_version="2025-03-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://mila-multilingual.cognitiveservices.azure.com/"),
        api_key="9lRwsl3JjzVJ7TxgQXKnKAUPoce6rASWQsZNqyEJsfAPkTcBWXP4JQQJ99BIACYeBjFXJ3w3AAAAACOG2gHM"
    )
    
    # Initialize batch manager
    batch_manager = BatchMathManager(
        client=client,
        base_dir="/home/mila/x/xut/github/playground/azure_api_key/batch_submission/openai-batch",
        deployment=args.deployment,
    )
    
    # Execute command
    if args.command == "list-pending":
        list_pending_jobs(batch_manager)
    
    elif args.command == "resume":
        resume_pending_jobs(batch_manager, check_interval=args.check_interval)
    
    elif args.command == "status":
        if not args.batch_id:
            print("Error: batch_id is required for 'status' command")
            sys.exit(1)
        check_job_status(batch_manager, args.batch_id)
    
    elif args.command == "cancel":
        if not args.batch_id:
            print("Error: batch_id is required for 'cancel' command")
            sys.exit(1)
        cancel_job(batch_manager, args.batch_id)
    
    elif args.command == "list-all":
        list_all_jobs(batch_manager, limit=args.limit)
    
    elif args.command == "structure":
        show_directory_structure(batch_manager)
    
    elif args.command == "list-files":
        list_uploaded_files(batch_manager, limit=args.limit)
    
    elif args.command == "delete-files":
        delete_all_files(batch_manager, force=args.force)
    
    elif args.command == "delete-expired":
        delete_expired_files(batch_manager, dry_run=not args.execute)
    
    elif args.command == "cancel-all":
        cancel_all_pending(batch_manager, force=args.force)


if __name__ == "__main__":
    main()
