import sys
sys.path.insert(0, '.')

import psycopg2
from configs.config import config

def get_connection():
    """
    PostgreSQL connection return karta hai.
    Context manager pattern — use with 'with' statement.
    """
    return psycopg2.connect(
        host=config.database.host,
        port=config.database.port,
        dbname=config.database.name,
        user=config.database.user,
        password=config.database.password
    )

def initialize_database():
    """
    FraudShield ke liye 3 tables banata hai.
    IF NOT EXISTS — safe to run multiple times.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1 — Raw transactions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              SERIAL PRIMARY KEY,
            transaction_id  TEXT UNIQUE NOT NULL,
            amount          FLOAT,
            sender_id       TEXT,
            receiver_id     TEXT,
            transaction_type TEXT,
            is_fraud        INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index — fraud queries fast hongi
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_is_fraud
        ON transactions(is_fraud)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sender
        ON transactions(sender_id)
    """)

    # Table 2 — Model predictions log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              SERIAL PRIMARY KEY,
            transaction_id  TEXT REFERENCES transactions(transaction_id),
            fraud_probability FLOAT,
            is_fraud_predicted INTEGER,
            model_version   TEXT,
            latency_ms      FLOAT,
            predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 3 — Model versions track karna
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id              SERIAL PRIMARY KEY,
            version         TEXT UNIQUE NOT NULL,
            algorithm       TEXT,
            auc_score       FLOAT,
            precision_score FLOAT,
            recall_score    FLOAT,
            is_active       BOOLEAN DEFAULT FALSE,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database tables created successfully!")

if __name__ == "__main__":
    initialize_database()
