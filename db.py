import sqlite3
from datetime import datetime
import cv2
import os


def init_db():
    conn = sqlite3.connect('illegal_vehicle.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS illegal_vehicles (
        track_id INTEGER PRIMARY KEY,
        timestamp TEXT,
        class TEXT,
        x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
        image_path TEXT
    )""")
    conn.commit()

    return conn, cursor


def is_already_saved(cursor, track_id):
    cursor.execute("SELECT 1 FROM illegal_vehicles WHERE track_id=?", (track_id,))
    
    return cursor.fetchone() is not None


def save_illegal_vehicle(frame, box, track_id, cursor, conn):
    x1, y1, x2, y2, _ = map(int, box)
    roi = frame[y1:y2, x1:x2]
    save_path = f"saved_illegal/illegal_{track_id}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, roi)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO illegal_vehicles (track_id, timestamp, class, x1, y1, x2, y2, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (track_id, timestamp, 'illegal', x1, y1, x2, y2, save_path))
    conn.commit()