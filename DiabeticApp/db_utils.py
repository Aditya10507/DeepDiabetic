import os
import sqlite3

import pymysql

from .app_config import APP_DB_BACKEND
from .app_config import MYSQL_CONFIG
from .app_config import SQLITE_DB_PATH


def get_connection():
    if APP_DB_BACKEND == "mysql":
        return pymysql.connect(**MYSQL_CONFIG)
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    connection = sqlite3.connect(SQLITE_DB_PATH)
    ensure_sqlite_schema(connection)
    return connection


def ensure_sqlite_schema(connection):
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS register (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            contact_no TEXT,
            email TEXT,
            address TEXT
        )
        """
    )
    connection.commit()


def _placeholder():
    return "%s" if APP_DB_BACKEND == "mysql" else "?"


def _fetchall(query, params=()):
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
    finally:
        connection.close()


def _execute(query, params=()):
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(query, params)
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def validate_user(username, password):
    placeholder = _placeholder()
    rows = _fetchall(
        f"SELECT username, password FROM register WHERE username = {placeholder} AND password = {placeholder}",
        (username, password),
    )
    return len(rows) > 0


def username_exists(username):
    placeholder = _placeholder()
    rows = _fetchall(f"SELECT username FROM register WHERE username = {placeholder}", (username,))
    return len(rows) > 0


def create_user(username, password, contact, email, address):
    placeholder = _placeholder()
    return _execute(
        "INSERT INTO register (username, password, contact_no, email, address) "
        f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
        (username, password, contact, email, address),
    )
