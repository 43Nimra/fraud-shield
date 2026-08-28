import sys
sys.path.insert(0, '.')

import pandas as pd
from src.data.loader import load_full_dataset, get_dataset_info
from src.data.database import get_connection
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sample_to_db():
    """
    Pehle 10K rows PostgreSQL mein load karo
    Poora 590K baad mein — abhi SQL sikhne ke liye sample kafi hai
    """
    logger.info("Loading sample data to PostgreSQL...")
    df_t, df_i = load_full_dataset()

    # Sample — 5K fraud + 5K legitimate (balanced sample for exploration)
    fraud_sample = df_t[df_t['isFraud'] == 1].sample(n=5000, random_state=42)
    legit_sample = df_t[df_t['isFraud'] == 0].sample(n=5000, random_state=42)
    sample = pd.concat([fraud_sample, legit_sample]).reset_index(drop=True)

    conn = get_connection()
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM transactions")

    inserted = 0
    for _, row in sample.iterrows():
        cursor.execute("""
            INSERT INTO transactions 
                (transaction_id, amount, sender_id, receiver_id, transaction_type, is_fraud)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO NOTHING
        """, (
            str(row['TransactionID']),
            float(row['TransactionAmt']),
            str(row.get('card1', 'unknown')),
            str(row.get('card2', 'unknown')),
            str(row.get('ProductCD', 'unknown')),
            int(row['isFraud'])
        ))
        inserted += 1

    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"{inserted:,} rows inserted into PostgreSQL")
    return sample

def run_sql_analysis(sample_df: pd.DataFrame):
    """
    SQL se fraud patterns explore karo
    """
    conn = get_connection()
    cursor = conn.cursor()

    print("\n" + "="*50)
    print("FRAUD PATTERN ANALYSIS — SQL")
    print("="*50)

    # Query 1 — Basic count
    cursor.execute("""
        SELECT 
            is_fraud,
            COUNT(*) as total,
            ROUND(AVG(amount)::numeric, 2) as avg_amount,
            ROUND(MIN(amount)::numeric, 2) as min_amount,
            ROUND(MAX(amount)::numeric, 2) as max_amount
        FROM transactions
        GROUP BY is_fraud
        ORDER BY is_fraud
    """)
    print("\nQuery 1: Fraud vs Legitimate — Amount Comparison")
    print(f"{'Type':<12} {'Count':>8} {'Avg Amount':>12} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for row in cursor.fetchall():
        label = "FRAUD" if row[0] == 1 else "Legitimate"
        print(f"{label:<12} {row[1]:>8,} {row[2]:>12} {row[3]:>10} {row[4]:>10}")

    # Query 2 — Window Function: Running total fraud per product
    cursor.execute("""
        SELECT 
            transaction_type,
            is_fraud,
            COUNT(*) as count,
            ROUND(AVG(amount)::numeric, 2) as avg_amount,
            SUM(COUNT(*)) OVER (PARTITION BY transaction_type) as total_in_type,
            ROUND(
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY transaction_type),
                2
            ) as fraud_rate_pct
        FROM transactions
        GROUP BY transaction_type, is_fraud
        ORDER BY transaction_type, is_fraud DESC
    """)
    print("\nQuery 2: Window Function — Fraud Rate by Product Type")
    print(f"{'Product':<10} {'Fraud?':>7} {'Count':>7} {'Avg Amt':>10} {'Total':>7} {'Rate%':>7}")
    print("-" * 55)
    for row in cursor.fetchall():
        label = "YES" if row[1] == 1 else "no"
        print(f"{str(row[0]):<10} {label:>7} {row[2]:>7,} {row[3]:>10} {row[4]:>7,} {row[5]:>7}")

    # Query 3 — CTE: High value fraud transactions
    cursor.execute("""
        WITH fraud_transactions AS (
            SELECT 
                transaction_id,
                amount,
                transaction_type,
                is_fraud
            FROM transactions
            WHERE is_fraud = 1
        ),
        high_value_fraud AS (
            SELECT *
            FROM fraud_transactions
            WHERE amount > 500
        )
        SELECT 
            COUNT(*) as high_value_count,
            ROUND(AVG(amount)::numeric, 2) as avg_amount,
            ROUND(MAX(amount)::numeric, 2) as max_amount
        FROM high_value_fraud
    """)
    print("\nQuery 3: CTE — High Value Fraud (>$500)")
    row = cursor.fetchone()
    print(f"  High value fraud transactions : {row[0]:,}")
    print(f"  Average amount               : ${row[1]}")
    print(f"  Maximum amount               : ${row[2]}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    sample = load_sample_to_db()
    run_sql_analysis(sample)
    print("\nDay 2 EDA complete!")
