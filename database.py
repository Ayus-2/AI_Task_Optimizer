import sqlite3

conn = sqlite3.connect("employee.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS moods")
cursor.execute("""
CREATE TABLE IF NOT EXISTS moods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    emotion TEXT,
    employee TEXT DEFAULT 'You',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
