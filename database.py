import sqlite3


DB_PATH = "data/business.db"


def create_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            product TEXT NOT NULL,
            category TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            revenue REAL NOT NULL
        )
    """)

    cursor.execute("SELECT COUNT(*) FROM sales")

    count = cursor.fetchone()[0]

    if count == 0:

        sample_data = [
            ("Laptop", "Electronics", 15, 900000),
            ("Phone", "Electronics", 30, 750000),
            ("Headphones", "Accessories", 50, 125000),
            ("Monitor", "Electronics", 20, 300000),
            ("Keyboard", "Accessories", 40, 80000),
            ("Mouse", "Accessories", 60, 60000),
        ]

        cursor.executemany(
            """
            INSERT INTO sales
            (product, category, quantity, revenue)
            VALUES (?, ?, ?, ?)
            """,
            sample_data
        )

    connection.commit()
    connection.close()


def execute_query(query: str):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(query)

    rows = cursor.fetchall()

    columns = [
        description[0]
        for description in cursor.description
    ]

    connection.close()

    return columns, rows


if __name__ == "__main__":

    create_database()

    columns, rows = execute_query(
        """
        SELECT product, revenue
        FROM sales
        ORDER BY revenue DESC
        """
    )

    print(columns)

    for row in rows:
        print(row)