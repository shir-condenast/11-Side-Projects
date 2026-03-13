# Databricks notebook source
"""
Fetch article data from Delta table and save as JSON for RAG pipeline.

This script:
1. Connects to the Delta table: gold_us_prod.content.gld_cross_brand_live
2. Fetches Architectural Digest articles (limited to 100 for now)
3. Transforms the data to match the expected schema
4. Saves to data/articles.json for the Python RAG pipeline

Run this on Databricks cluster: DDA Cluster
"""

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace, concat_ws
import json
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Fetch Data from Delta Table

# COMMAND ----------

# Query the Delta table
query = """
SELECT 
    id,
    hed,
    body,
    full_url
FROM gold_us_prod.content.gld_cross_brand_live
WHERE brand = 'Architectural Digest'
    AND body IS NOT NULL
    AND hed IS NOT NULL
    AND full_url IS NOT NULL
    AND published_date >= '2024-01-01'
"""

print("Fetching data from Delta table...")
df = spark.sql(query)

# Show sample
print(f"\nFetched {df.count()} articles")
df.show(5, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transform Data to Match RAG Schema

# COMMAND ----------

# Transform to match expected schema:
# - id: keep as is (convert to string)
# - hed -> title
# - body -> content
# - full_url -> url

transformed_df = df.select(
    col("id").cast("string").alias("id"),
    trim(col("hed")).alias("title"),
    # Clean body text: remove excessive whitespace, newlines
    regexp_replace(trim(col("body")), r'\s+', ' ').alias("content"),
    trim(col("full_url")).alias("url")
)

# Filter out any rows with null/empty values after transformation
transformed_df = transformed_df.filter(
    (col("id").isNotNull()) & 
    (col("title") != "") & 
    (col("content") != "") & 
    (col("url") != "")
)

print(f"\nTransformed {transformed_df.count()} articles")
transformed_df.show(5, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Convert to JSON Format

# COMMAND ----------

# Collect data to driver (safe for 100 articles)
articles_list = transformed_df.collect()

# Convert to list of dictionaries
articles_json = [
    {
        "id": row.id,
        "title": row.title,
        "content": row.content,
        "url": row.url
    }
    for row in articles_list
]

print(f"\nConverted {len(articles_json)} articles to JSON format")
print("\nSample article:")
print(json.dumps(articles_json[0], indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save to DBFS and Local File System

# COMMAND ----------

# Save to DBFS first
dbfs_path = "/FileStore/rag_articles/articles.json"
local_dbfs_path = "/dbfs" + dbfs_path

# Create directory if it doesn't exist
dbutils.fs.mkdirs("/FileStore/rag_articles/")

# Write JSON file
with open(local_dbfs_path, 'w', encoding='utf-8') as f:
    json.dump(articles_json, f, indent=2, ensure_ascii=False)

print(f"✅ Saved to DBFS: {dbfs_path}")
print(f"   Access via: /dbfs{dbfs_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Display Summary Statistics

# COMMAND ----------

# Calculate some statistics
from pyspark.sql.functions import length, avg, min as spark_min, max as spark_max

stats_df = transformed_df.select(
    length(col("content")).alias("content_length")
).agg(
    avg("content_length").alias("avg_length"),
    spark_min("content_length").alias("min_length"),
    spark_max("content_length").alias("max_length")
)

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total articles fetched: {len(articles_json)}")
print(f"Source: gold_us_prod.content.gld_cross_brand_live")
print(f"Brand: Architectural Digest")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nContent Length Statistics:")
stats_df.show()
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Download Instructions
# MAGIC 
# MAGIC To download the file to your local machine:
# MAGIC 
# MAGIC **Option 1: Using Databricks CLI**
# MAGIC ```bash
# MAGIC databricks fs cp dbfs:/FileStore/rag_articles/articles.json ./11-Side-Projects/01\ RAG\ For\ Articles/data/articles.json
# MAGIC ```
# MAGIC 
# MAGIC **Option 2: Using the UI**
# MAGIC 1. Go to Data > DBFS > FileStore > rag_articles
# MAGIC 2. Click on articles.json
# MAGIC 3. Download the file
# MAGIC 4. Move it to: `11-Side-Projects/01 RAG For Articles/data/articles.json`
# MAGIC 
# MAGIC **Option 3: Using Python script (see download_articles.py)**

print("\n✅ Data fetch complete!")
print(f"📁 File location: {dbfs_path}")
print("\n📥 Next steps:")
print("1. Download the file using one of the methods above")
print("2. Place it in: 11-Side-Projects/01 RAG For Articles/data/articles.json")
print("3. Run your Streamlit app: streamlit run app.py")

