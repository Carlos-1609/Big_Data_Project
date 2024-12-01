from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

# Step 1: Initialize Spark session
spark = SparkSession.builder \
    .appName("NYHousingAnalysis") \
    .getOrCreate()

# Step 2: Load the dataset from HDFS
hdfs_path = "hdfs://localhost:9000/ny_housing/NY_dataset.csv"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# Step 3: Preview the data and schema
df.show(10)
df.printSchema()

# Step 4: Handle missing values and filter invalid data
df_cleaned = df.dropna().filter((df["PRICE"] > 0) & (df["PROPERTYSQFT"] > 0))

# Step 5: Add calculated columns (price per square foot and price category)
df_cleaned = df_cleaned.withColumn("price_per_sqft", df_cleaned["PRICE"] / df_cleaned["PROPERTYSQFT"])
df_cleaned = df_cleaned.withColumn(
    "price_category",
    when(df_cleaned["PRICE"] < 500000, "Low")
    .when(df_cleaned["PRICE"] < 1000000, "Medium")
    .otherwise("High")
)

# Step 6: Perform analysis and queries
# 6.1 Average price by borough (or SUBLOCALITY)
avg_price = df_cleaned.groupBy("SUBLOCALITY").avg("PRICE")
avg_price.show()

# 6.2 Top 10 most expensive properties
top_properties = df_cleaned.orderBy(df_cleaned["PRICE"].desc()).limit(10)
top_properties.show()

# 6.3 Count of properties by type
property_count = df_cleaned.groupBy("TYPE").count()
property_count.show()

# 6.4 Filter properties with 3+ bedrooms under $500,000
affordable_homes = df_cleaned.filter((df_cleaned["BEDS"] >= 3) & (df_cleaned["PRICE"] < 500000))
affordable_homes.show()

# Step 7: Save results back to HDFS
# Step 7.1 Save cleaned data 
df_cleaned.write.mode("overwrite").csv("hdfs://localhost:9000/ny_housing/cleaned_data.csv", header=True)

# Step 7.2 Save average price by borough 
avg_price.write.mode("overwrite").csv("hdfs://localhost:9000/ny_housing/avg_price_by_borough.csv", header=True)

# Step 8: Improved visualization (requires pandas and matplotlib)
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns

    # Convert the average price DataFrame to Pandas
    pandas_df = avg_price.toPandas()

    # Create a bar chart with improved formatting
    ax = pandas_df.plot(kind="bar", x="SUBLOCALITY", y="avg(PRICE)", legend=False)
    ax.set_title("Average Price by Sublocality")
    ax.set_ylabel("Price ($)")
    
    # Format Y-axis as standard currency
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    
    # Rotate X-axis labels for better visibility
    plt.xticks(rotation=45, ha="right")
    
    # Add gridlines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit labels
    plt.tight_layout()
    plt.show()
    
    # Count properties by price category
    price_category_count = df_cleaned.groupBy("price_category").count().toPandas()

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        price_category_count["count"],
        labels=price_category_count["price_category"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#ff9999", "#66b3ff", "#99ff99"],
    )
    plt.title("Proportion of Price Categories", fontsize=16)
    plt.tight_layout()
    plt.show()


except ImportError:
    print("Pandas or Matplotlib not installed. Skipping visualization.")

# Stop the Spark session
spark.stop()
