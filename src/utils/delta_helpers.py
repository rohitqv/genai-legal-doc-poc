"""
Delta Lake helper functions for Legal Document Analysis PoC
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType
from .config import (
    BRONZE_DOCS_TABLE, BRONZE_METADATA_TABLE, SILVER_TEXTS_TABLE,
    GOLD_EXTRACTIONS_TABLE, AUDIT_LOG_TABLE
)
from .logger import logger


def get_spark_session():
    """Get or create Spark session"""
    return SparkSession.builder \
        .appName("LegalDocAnalysis") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()


def create_bronze_docs_table(spark: SparkSession, database: str = "default"):
    """Create Bronze layer table for raw documents"""
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("filename", StringType(), False),
        StructField("file_path", StringType(), False),
        StructField("file_type", StringType(), False),
        StructField("file_size_bytes", IntegerType(), False),
        StructField("ingestion_timestamp", TimestampType(), False),
        StructField("source", StringType(), True),
        StructField("raw_content", StringType(), True)  # Base64 encoded or path reference
    ])
    
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {database}.{BRONZE_DOCS_TABLE} (
            doc_id STRING NOT NULL,
            filename STRING NOT NULL,
            file_path STRING NOT NULL,
            file_type STRING NOT NULL,
            file_size_bytes INT NOT NULL,
            ingestion_timestamp TIMESTAMP NOT NULL,
            source STRING,
            raw_content STRING
        ) USING DELTA
        TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
    """)
    logger.info(f"Created/verified table: {database}.{BRONZE_DOCS_TABLE}")


def create_bronze_metadata_table(spark: SparkSession, database: str = "default"):
    """Create Bronze layer metadata table"""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {database}.{BRONZE_METADATA_TABLE} (
            doc_id STRING NOT NULL,
            metadata_key STRING NOT NULL,
            metadata_value STRING,
            created_at TIMESTAMP NOT NULL
        ) USING DELTA
        PARTITIONED BY (doc_id)
    """)
    logger.info(f"Created/verified table: {database}.{BRONZE_METADATA_TABLE}")


def create_silver_texts_table(spark: SparkSession, database: str = "default"):
    """Create Silver layer table for processed text"""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {database}.{SILVER_TEXTS_TABLE} (
            doc_id STRING NOT NULL,
            chunk_id STRING NOT NULL,
            chunk_index INT NOT NULL,
            text_content STRING NOT NULL,
            language STRING,
            translation_applied BOOLEAN,
            processing_timestamp TIMESTAMP NOT NULL,
            metadata MAP<STRING, STRING>
        ) USING DELTA
        PARTITIONED BY (doc_id)
        TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true'
        )
    """)
    logger.info(f"Created/verified table: {database}.{SILVER_TEXTS_TABLE}")


def create_gold_extractions_table(spark: SparkSession, database: str = "default"):
    """Create Gold layer table for extracted attributes"""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {database}.{GOLD_EXTRACTIONS_TABLE} (
            doc_id STRING NOT NULL,
            extraction_id STRING NOT NULL,
            attribute_name STRING NOT NULL,
            attribute_value STRING,
            confidence_score DOUBLE,
            model_used STRING,
            ensemble_agreement INT,
            extraction_timestamp TIMESTAMP NOT NULL
        ) USING DELTA
        PARTITIONED BY (doc_id)
    """)
    logger.info(f"Created/verified table: {database}.{GOLD_EXTRACTIONS_TABLE}")


def create_audit_log_table(spark: SparkSession, database: str = "default"):
    """Create audit log table"""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {database}.{AUDIT_LOG_TABLE} (
            run_id STRING NOT NULL,
            workflow_name STRING NOT NULL,
            step_name STRING NOT NULL,
            status STRING NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            records_processed INT,
            records_failed INT,
            error_message STRING,
            metadata MAP<STRING, STRING>
        ) USING DELTA
        PARTITIONED BY (workflow_name, status)
        TBLPROPERTIES (
            'delta.logRetentionDuration' = 'interval 90 days',
            'delta.deletedFileRetentionDuration' = 'interval 30 days'
        )
    """)
    logger.info(f"Created/verified table: {database}.{AUDIT_LOG_TABLE}")


def log_audit_event(
    spark: SparkSession,
    run_id: str,
    workflow_name: str,
    step_name: str,
    status: str,
    start_time,
    end_time=None,
    records_processed: int = 0,
    records_failed: int = 0,
    error_message: str = None,
    metadata: dict = None,
    database: str = "default"
):
    """Log an audit event"""
    from pyspark.sql import Row
    
    audit_row = Row(
        run_id=run_id,
        workflow_name=workflow_name,
        step_name=step_name,
        status=status,
        start_time=start_time,
        end_time=end_time,
        records_processed=records_processed,
        records_failed=records_failed,
        error_message=error_message,
        metadata=metadata or {}
    )
    
    df = spark.createDataFrame([audit_row])
    df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
        f"{database}.{AUDIT_LOG_TABLE}"
    )
    logger.info(f"Audit event logged: {workflow_name}.{step_name} - {status}")


def initialize_all_tables(spark: SparkSession, database: str = "default"):
    """Initialize all Delta tables"""
    logger.info("Initializing all Delta tables...")
    create_bronze_docs_table(spark, database)
    create_bronze_metadata_table(spark, database)
    create_silver_texts_table(spark, database)
    create_gold_extractions_table(spark, database)
    create_audit_log_table(spark, database)
    logger.info("All Delta tables initialized successfully")

