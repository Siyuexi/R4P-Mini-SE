import os
import sqlite3
from dataclasses import dataclass

@dataclass
class FunctionEntry:
    name: str
    file_path: str
    body: str
    start_line: int
    end_line: int
    parent_function: "FunctionEntry | None" = None
    parent_class: "ClassEntry | None" = None

@dataclass
class ClassEntry:
    name: str
    file_path: str
    body: str
    fields: list[str]
    methods: list[str]
    start_line: int
    end_line: int

class DataBase:
    def __init__(self, root_dir, instance_id):
        self.root = root_dir
        self.path = f'{root_dir}/codegraph/{instance_id}.db'
        
        if not os.path.exists(f'{root_dir}/codegraph'):
            os.mkdir(f'{root_dir}/codegraph')
        self.db_connection = sqlite3.connect(self.path)

    def init_db(self):
        for sql in [FUNCTION_SQL, CLASS_SQL, CLASS_METHOD_SQL, DIRECTORY_SQL, FILE_SQL]:
            self.db_connection.execute(sql)
        self.db_connection.commit()

    def disconnect(self):
        self.db_connection.close()

    def __del__(self):
        self.disconnect()
        
    def insert_entry(self, entry: FunctionEntry | ClassEntry) -> None:
        match entry:
            case FunctionEntry():
                self._insert_function_handler(entry)
            case ClassEntry():
                self._insert_class_handler(entry)

        self.db_connection.commit()

    def insert_directory(self, path: str, files: str) -> None:
        """Insert directory information into database."""
        self.db_connection.execute(
            """
            INSERT OR REPLACE INTO directories (path, files)
            VALUES (?, ?)
            """,
            (path, files)
        )
        self.db_connection.commit()
        
    def insert_file(self, path: str, entities: str) -> None:
        """Insert file information into database."""
        self.db_connection.execute(
            """
            INSERT OR REPLACE INTO files (path, entities)
            VALUES (?, ?)
            """,
            (path, entities)
        )
        self.db_connection.commit()

    def _insert_function_handler(self, entry: FunctionEntry) -> None:
        if entry.parent_class:
            self.db_connection.execute(
                """
                    INSERT INTO class_methods (name, class_name, file_path, body, start_line, end_line)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.name,
                    entry.parent_class.name,
                    entry.file_path,
                    entry.body,
                    entry.start_line,
                    entry.end_line,
                ),
            )
        else:
            # no parent class, so we need to insert a function
            self.db_connection.execute(
                """
                    INSERT INTO functions (name, file_path, body, start_line, end_line)
                    VALUES (?, ?, ?, ?, ?)
                """,
                (entry.name, entry.file_path, entry.body, entry.start_line, entry.end_line),
            )

    def _insert_class_handler(self, entry: ClassEntry) -> None:
        class_fields: str = "\n".join(entry.fields)
        class_methods: str = "\n".join(entry.methods)
        self.db_connection.execute(
            """
                INSERT INTO classes (name, file_path, body, fields, methods, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.name,
                entry.file_path,
                entry.body,
                class_fields,
                class_methods,
                entry.start_line,
                entry.end_line,
            ),
        )


FUNCTION_SQL = """
    CREATE TABLE IF NOT EXISTS functions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        body TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL
    )"""

CLASS_SQL = """
    CREATE TABLE IF NOT EXISTS classes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        body TEXT NOT NULL,
        fields TEXT NOT NULL,
        methods TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL
    )"""

CLASS_METHOD_SQL = """
    CREATE TABLE IF NOT EXISTS class_methods (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        class_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        body TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL
    )"""

DIRECTORY_SQL = """
    CREATE TABLE IF NOT EXISTS directories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        files TEXT NOT NULL
    )"""

FILE_SQL = """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        entities TEXT NOT NULL
    )"""