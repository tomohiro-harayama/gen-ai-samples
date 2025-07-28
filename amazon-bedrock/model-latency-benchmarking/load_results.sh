#!/bin/bash

# Define the SQLite database file
DB_FILE="benchmark_results.db"

# Create the SQLite database if it doesn't exist
if [ ! -f "$DB_FILE" ]; then
    touch "$DB_FILE"
fi

# Iterate over all CSV files in the results folder
for csv_file in results/*.csv; do
    # Extract the table name from the CSV file name
    table_name=$(basename "$csv_file" .csv)

    # Define the schema for the table
    schema="CREATE TABLE IF NOT EXISTS $table_name (
        time_to_first_byte DECIMAL NOT NULL, 
        time_to_last_byte DECIMAL NOT NULL, 
        job_timestamp_iso TIMESTAMP, 
        configured_output_tokens_for_request DECIMAL NOT NULL, 
        model_input_tokens DECIMAL NOT NULL, 
        model_output_tokens DECIMAL NOT NULL, 
        model VARCHAR NOT NULL, 
        region VARCHAR NOT NULL, 
        invocation_id DECIMAL NOT NULL, 
        api_call_status VARCHAR NOT NULL, 
        full_error_message VARCHAR NOT NULL, 
        temperature BOOLEAN NOT NULL, 
        top_p BOOLEAN NOT NULL, 
        top_k DECIMAL NOT NULL, 
        experiment_name VARCHAR NOT NULL, 
        task_type VARCHAR NOT NULL, 
        performance_config VARCHAR, 
        timestamp TIMESTAMP, 
        run_count DECIMAL NOT NULL
    );"

    # Create the table in the SQLite database
    sqlite3 "$DB_FILE" "$schema"

    # Load the CSV data into the table
    sqlite3 "$DB_FILE" <<EOF
.mode csv
.import --skip 1 $csv_file $table_name
EOF

    echo "Data from $csv_file has been loaded into table $table_name."
done

echo "All CSV files have been processed."