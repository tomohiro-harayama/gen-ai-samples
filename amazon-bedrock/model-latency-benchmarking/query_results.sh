#!/bin/bash

# Define the SQLite database file
DB_FILE="benchmark_results.db"

# Check if the database file exists
if [ ! -f "$DB_FILE" ]; then
    echo "Database file $DB_FILE does not exist. Please run the data loading script first."
    exit 1
fi

# Get the list of tables in the database
tables=$(sqlite3 "$DB_FILE" ".tables")

# Iterate over each table and execute the query
for table in $tables; do
    echo "Querying table: $table"
    sqlite3 "$DB_FILE" <<EOF
SELECT 
    AVG(time_to_first_byte) AS avg_time_to_first_byte,
    AVG(time_to_last_byte) AS avg_time_to_last_byte,
    region,
    model
FROM $table
GROUP BY region, model
ORDER BY avg_time_to_first_byte;
EOF
    echo "----------------------------------------"
done

echo "All tables have been queried."