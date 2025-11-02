"""
SEC EDGAR data ingestion module
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sec_edgar_downloader import Downloader
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType

from ..utils.config import (
    RAW_DATA_PATH, BRONZE_DOCS_TABLE, BRONZE_METADATA_TABLE,
    SEC_TICKER, SEC_FILING_TYPES, SEC_FILING_DATE_RANGE
)
from ..utils.logger import logger
from ..utils.delta_helpers import get_spark_session, log_audit_event


class SECIngestor:
    """Class to handle SEC EDGAR data ingestion"""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_spark_session()
        self.downloader = Downloader()
        self.raw_data_path = Path(RAW_DATA_PATH)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
    def download_filings(
        self,
        ticker: str = SEC_TICKER,
        filing_types: List[str] = None,
        date_range: tuple = None
    ) -> List[dict]:
        """
        Download SEC filings for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types (e.g., ["10-K", "10-Q"])
            date_range: Tuple of (start_date, end_date) as strings
            
        Returns:
            List of downloaded file metadata
        """
        if filing_types is None:
            filing_types = SEC_FILING_TYPES
        if date_range is None:
            date_range = SEC_FILING_DATE_RANGE
            
        logger.info(f"Downloading SEC filings for {ticker}: {filing_types}")
        
        downloaded_files = []
        start_date, end_date = date_range
        
        try:
            for filing_type in filing_types:
                logger.info(f"Downloading {filing_type} filings...")
                self.downloader.get(
                    filing_type=filing_type,
                    ticker=ticker,
                    after=start_date,
                    before=end_date,
                    download_details=True
                )
                
                # Get downloaded files
                ticker_path = self.raw_data_path / ticker.lower() / filing_type
                if ticker_path.exists():
                    for file_path in ticker_path.rglob("*.txt"):
                        file_stat = file_path.stat()
                        downloaded_files.append({
                            "filename": file_path.name,
                            "file_path": str(file_path),
                            "file_type": filing_type,
                            "file_size_bytes": file_stat.st_size,
                            "ticker": ticker,
                            "source": "SEC EDGAR"
                        })
                        
            logger.info(f"Downloaded {len(downloaded_files)} files")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading SEC filings: {str(e)}")
            raise
    
    def ingest_to_bronze(self, files: List[dict], database: str = "default") -> int:
        """
        Ingest downloaded files into Bronze Delta table
        
        Args:
            files: List of file metadata dictionaries
            database: Database name
            
        Returns:
            Number of files ingested
        """
        logger.info(f"Ingesting {len(files)} files into Bronze layer...")
        
        run_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            ingested_count = 0
            
            for file_info in files:
                doc_id = str(uuid.uuid4())
                
                # Read file content (for small files, store reference for large files)
                file_path = file_info["file_path"]
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {str(e)}")
                    content = None
                
                # Create row for bronze_docs table
                doc_row = {
                    "doc_id": doc_id,
                    "filename": file_info["filename"],
                    "file_path": file_path,
                    "file_type": file_info["file_type"],
                    "file_size_bytes": file_info["file_size_bytes"],
                    "ingestion_timestamp": current_timestamp(),
                    "source": file_info.get("source", "SEC EDGAR"),
                    "raw_content": content[:100000] if content else None  # Limit size
                }
                
                # Create row for metadata table
                metadata_rows = [
                    {"doc_id": doc_id, "metadata_key": "ticker", "metadata_value": file_info.get("ticker"), "created_at": current_timestamp()},
                    {"doc_id": doc_id, "metadata_key": "filing_type", "metadata_value": file_info.get("file_type"), "created_at": current_timestamp()}
                ]
                
                # Write to Delta tables
                docs_df = self.spark.createDataFrame([doc_row])
                docs_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
                    f"{database}.{BRONZE_DOCS_TABLE}"
                )
                
                metadata_df = self.spark.createDataFrame(metadata_rows)
                metadata_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
                    f"{database}.{BRONZE_METADATA_TABLE}"
                )
                
                ingested_count += 1
                
            end_time = datetime.now()
            
            # Log audit event
            log_audit_event(
                self.spark,
                run_id=run_id,
                workflow_name="ingestion",
                step_name="sec_bronze_ingest",
                status="SUCCESS",
                start_time=start_time,
                end_time=end_time,
                records_processed=ingested_count,
                metadata={"source": "SEC EDGAR", "ticker": files[0].get("ticker") if files else None}
            )
            
            logger.info(f"Successfully ingested {ingested_count} files to Bronze layer")
            return ingested_count
            
        except Exception as e:
            end_time = datetime.now()
            log_audit_event(
                self.spark,
                run_id=run_id,
                workflow_name="ingestion",
                step_name="sec_bronze_ingest",
                status="FAILED",
                start_time=start_time,
                end_time=end_time,
                records_processed=0,
                records_failed=len(files),
                error_message=str(e)
            )
            logger.error(f"Error ingesting files: {str(e)}")
            raise
    
    def run_full_ingestion(
        self,
        ticker: str = SEC_TICKER,
        filing_types: List[str] = None,
        date_range: tuple = None,
        database: str = "default"
    ) -> int:
        """
        Run complete ingestion pipeline: download + ingest
        
        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types
            date_range: Date range tuple
            database: Database name
            
        Returns:
            Number of files ingested
        """
        logger.info("Starting full SEC ingestion pipeline...")
        
        # Download files
        files = self.download_filings(ticker, filing_types, date_range)
        
        if not files:
            logger.warning("No files downloaded")
            return 0
        
        # Ingest to Bronze
        ingested = self.ingest_to_bronze(files, database)
        
        return ingested


if __name__ == "__main__":
    # Example usage
    ingestor = SECIngestor()
    count = ingestor.run_full_ingestion(ticker="AAPL", filing_types=["10-K"], date_range=("2023-01-01", "2023-12-31"))
    print(f"Ingested {count} files")

