#!/usr/bin/env python3
"""
Create a fresh database with the new schema for Ollama Telegram Bot
This script creates a new database without importing data from the old one.
"""

import sqlite3
import os
from datetime import datetime

def create_fresh_database(db_path='../db/users.db'):
    """Create a fresh database with the new schema."""

    # Remove if exists
    if os.path.exists(db_path):
        response = input(f"⚠️  {db_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
        os.remove(db_path)

    print(f"🔨 Creating fresh database: {db_path}")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    try:
        # Create all tables with the new schema
        print("📊 Creating tables...")

        # Users table
        c.execute('''CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            preferred_model TEXT,
            language TEXT DEFAULT 'en'
        )''')
        print("  ✅ Created table: users")

        # Chats table
        c.execute('''CREATE TABLE chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model_used TEXT,
            tokens_used INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )''')
        print("  ✅ Created table: chats")

        # System prompts table
        c.execute('''CREATE TABLE system_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt TEXT NOT NULL,
            is_global BOOLEAN DEFAULT 0,
            name TEXT,
            usage_count INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )''')
        print("  ✅ Created table: system_prompts")

        # Conversation templates table
        c.execute('''CREATE TABLE conversation_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            description TEXT,
            template TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            usage_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )''')
        print("  ✅ Created table: conversation_templates")

        # User feedback table
        c.execute('''CREATE TABLE user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message_id INTEGER,
            rating INTEGER NOT NULL,
            feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )''')
        print("  ✅ Created table: user_feedback")

        # Create indexes
        print("🚀 Creating indexes...")

        indexes = [
            ("idx_chats_user_timestamp", "chats(user_id, timestamp)"),
            ("idx_chats_user_role", "chats(user_id, role)"),
            ("idx_prompts_user", "system_prompts(user_id)"),
            ("idx_prompts_global", "system_prompts(is_global)"),
            ("idx_templates_user", "conversation_templates(user_id)"),
            ("idx_feedback_user", "user_feedback(user_id)"),
            ("idx_feedback_rating", "user_feedback(rating)")
        ]

        for index_name, index_def in indexes:
            c.execute(f"CREATE INDEX {index_name} ON {index_def}")
            print(f"  ✅ Created index: {index_name}")

        # Add default templates
        print("📝 Adding default templates...")

        default_templates = [
            ("Code Review", "Template for code review requests",
             "Please review this code and provide feedback on:\n1. Code quality\n2. Potential bugs\n3. Performance issues\n4. Best practices\n\nCode:\n```\n[INSERT CODE HERE]\n```"),

            ("Brainstorming", "Template for brainstorming sessions",
             "I need help brainstorming ideas for: [TOPIC]\n\nContext: [PROVIDE CONTEXT]\n\nConstraints: [LIST ANY CONSTRAINTS]\n\nPlease provide creative and practical suggestions."),

            ("Learning Plan", "Template for creating learning plans",
             "I want to learn: [SUBJECT/SKILL]\n\nCurrent level: [BEGINNER/INTERMEDIATE/ADVANCED]\n\nTime available: [HOURS PER WEEK]\n\nGoal: [WHAT YOU WANT TO ACHIEVE]\n\nPlease create a structured learning plan."),

            ("Debug Helper", "Template for debugging assistance",
             "I'm experiencing this issue: [DESCRIBE THE PROBLEM]\n\nError message: ```\n[ERROR MESSAGE]\n```\n\nWhat I've tried: [LIST ATTEMPTS]\n\nEnvironment: [DESCRIBE ENVIRONMENT]\n\nPlease help me debug this."),

            ("Explain Like I'm 5", "Simple explanations for complex topics",
             "Please explain [TOPIC/CONCEPT] in simple terms that anyone can understand.\n\nAssume I have no prior knowledge of this subject."),

            ("Meeting Summary", "Template for summarizing meetings",
             "Meeting: [MEETING NAME]\nDate: [DATE]\nAttendees: [LIST ATTENDEES]\n\nKey points discussed:\n[PASTE MEETING NOTES/TRANSCRIPT]\n\nPlease provide:\n1. Executive summary\n2. Key decisions made\n3. Action items with owners\n4. Next steps"),
        ]

        for name, desc, template in default_templates:
            c.execute(
                "INSERT INTO conversation_templates (user_id, name, description, template) VALUES (?, ?, ?, ?)",
                (None, name, desc, template)
            )
        print(f"  ✅ Added {len(default_templates)} default templates")

        # Commit changes
        conn.commit()
        print(f"\n✅ Fresh database created successfully: {db_path}")

        return True

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error creating database: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main function."""
    print("🤖 Ollama Telegram Bot - Fresh Database Creator")
    print("=" * 50)

    # Create fresh database
    if not create_fresh_database():
        return

    print("\n✅ Fresh database created!")

if __name__ == "__main__":
    main()
