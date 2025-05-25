# >> interactions
import logging
import os
import aiohttp
import json
import sqlite3
import aiosqlite
from aiogram import types
from aiohttp import ClientTimeout
from asyncio import Lock
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import asyncio

load_dotenv()

# Environment variables
token = os.getenv("TOKEN")
allowed_ids = list(map(int, os.getenv("USER_IDS", "").split(","))) if os.getenv("USER_IDS") else []
admin_ids = list(map(int, os.getenv("ADMIN_IDS", "").split(","))) if os.getenv("ADMIN_IDS") else []
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "localhost")
ollama_port = os.getenv("OLLAMA_PORT", "11434")
log_level_str = os.getenv("LOG_LEVEL", "INFO")
allow_all_users_in_groups = bool(int(os.getenv("ALLOW_ALL_USERS_IN_GROUPS", "0")))
timeout = os.getenv("TIMEOUT", "3000")

# Logging setup
log_levels = list(logging._levelToName.values())
if log_level_str not in log_levels:
    log_level = logging.DEBUG
else:
    log_level = logging.getLevelName(log_level_str)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Database connection pool
class DatabasePool:
    def __init__(self, db_path: str = 'users.db'):
        self.db_path = db_path
        self._pool = None
        
    async def get_connection(self) -> aiosqlite.Connection:
        if not self._pool:
            self._pool = await aiosqlite.connect(self.db_path)
            self._pool.row_factory = sqlite3.Row
        return self._pool
    
    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

db_pool = DatabasePool()

# Synchronous database functions (for backward compatibility)
def add_system_prompt(user_id: int, prompt: str, is_global: bool, name: str = None) -> int:
    """Add a system prompt synchronously (backward compatibility)."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    name = name or prompt[:50]  # Use first 50 chars as name if not provided
    c.execute(
        "INSERT INTO system_prompts (user_id, prompt, is_global, name) VALUES (?, ?, ?, ?)",
        (user_id, prompt, is_global, name)
    )
    prompt_id = c.lastrowid
    conn.commit()
    conn.close()
    return prompt_id

def get_system_prompts(user_id: Optional[int] = None, is_global: Optional[bool] = None) -> List[Tuple]:
    """Get system prompts synchronously (backward compatibility)."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    query = "SELECT * FROM system_prompts WHERE 1=1"
    params = []

    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)

    if is_global is not None:
        query += " AND is_global = ?"
        params.append(is_global)

    c.execute(query, params)
    prompts = c.fetchall()
    conn.close()
    return prompts

def delete_system_prompt(prompt_id: int) -> bool:
    """Delete a system prompt (fixed typo in function name)."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM system_prompts WHERE id = ?", (prompt_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# Alias for backward compatibility (fixing the typo)
delete_ystem_prompt = delete_system_prompt

def load_allowed_ids_from_db() -> List[int]:
    """Load allowed user IDs from database."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users")
    user_ids = [row[0] for row in c.fetchall()]
    conn.close()
    logging.info(f"Loaded {len(user_ids)} allowed users from database")
    return user_ids

def get_all_users_from_db() -> List[Tuple[int, str]]:
    """Get all users from database."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM users")
    users = c.fetchall()
    conn.close()
    return users

def remove_user_from_db(user_id: int) -> bool:
    """Remove a user from database."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    removed = c.rowcount > 0
    conn.commit()
    conn.close()
    if removed:
        global allowed_ids
        allowed_ids = [uid for uid in allowed_ids if uid != user_id]
    return removed

# Async database functions (new functionality)
async def save_chat_message(user_id: int, role: str, content: str, model: str = None, tokens: int = 0):
    """Save chat message with model and token information."""
    conn = await db_pool.get_connection()
    await conn.execute(
        "INSERT INTO chats (user_id, role, content, model_used, tokens_used) VALUES (?, ?, ?, ?, ?)",
        (user_id, role, content, model, tokens)
    )
    await conn.commit()

async def save_feedback(user_id: int, message_id: int, rating: int, feedback_text: str = None):
    """Save user feedback for a message."""
    conn = await db_pool.get_connection()
    await conn.execute(
        "INSERT INTO user_feedback (user_id, message_id, rating, feedback) VALUES (?, ?, ?, ?)",
        (user_id, message_id, rating, feedback_text)
    )
    await conn.commit()

async def get_system_prompt_for_user(user_id: int) -> Optional[str]:
    """Get the active system prompt for a user."""
    conn = await db_pool.get_connection()
    
    # First check for user-specific prompts
    cursor = await conn.execute(
        """SELECT prompt FROM system_prompts 
           WHERE user_id = ? AND is_global = 0 
           ORDER BY timestamp DESC LIMIT 1""",
        (user_id,)
    )
    row = await cursor.fetchone()
    if row:
        return row[0]
    
    # Then check for global prompts
    cursor = await conn.execute(
        """SELECT prompt FROM system_prompts 
           WHERE is_global = 1 
           ORDER BY timestamp DESC LIMIT 1"""
    )
    row = await cursor.fetchone()
    if row:
        return row[0]
    
    return None

async def update_prompt_usage(prompt_id: int):
    """Update usage count for a prompt."""
    conn = await db_pool.get_connection()
    await conn.execute(
        "UPDATE system_prompts SET usage_count = usage_count + 1 WHERE id = ?",
        (prompt_id,)
    )
    await conn.commit()

async def get_user_preferences(user_id: int) -> Dict[str, Any]:
    """Get user preferences."""
    conn = await db_pool.get_connection()
    cursor = await conn.execute(
        "SELECT preferred_model, language FROM users WHERE id = ?",
        (user_id,)
    )
    row = await cursor.fetchone()
    if row:
        return {
            'preferred_model': row[0],
            'language': row[1]
        }
    return {'preferred_model': None, 'language': 'en'}

async def update_user_preference(user_id: int, preference: str, value: str):
    """Update a user preference."""
    conn = await db_pool.get_connection()
    if preference == 'model':
        await conn.execute(
            "UPDATE users SET preferred_model = ? WHERE id = ?",
            (value, user_id)
        )
    elif preference == 'language':
        await conn.execute(
            "UPDATE users SET language = ? WHERE id = ?",
            (value, user_id)
        )
    await conn.commit()

async def create_conversation_template(user_id: int, name: str, description: str, template: str) -> int:
    """Create a conversation template."""
    conn = await db_pool.get_connection()
    cursor = await conn.execute(
        """INSERT INTO conversation_templates (user_id, name, description, template) 
           VALUES (?, ?, ?, ?)""",
        (user_id, name, description, template)
    )
    template_id = cursor.lastrowid
    await conn.commit()
    return template_id

async def get_conversation_templates(user_id: int) -> List[Dict[str, Any]]:
    """Get conversation templates for a user."""
    conn = await db_pool.get_connection()
    cursor = await conn.execute(
        """SELECT id, name, description, template, usage_count 
           FROM conversation_templates 
           WHERE user_id = ? 
           ORDER BY usage_count DESC, created_at DESC""",
        (user_id,)
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]

async def update_template_usage(template_id: int):
    """Update usage count for a template."""
    conn = await db_pool.get_connection()
    await conn.execute(
        "UPDATE conversation_templates SET usage_count = usage_count + 1 WHERE id = ?",
        (template_id,)
    )
    await conn.commit()

# Ollama API functions
async def manage_model(action: str, model_name: str) -> aiohttp.ClientResponse:
    """Manage Ollama models (pull/delete)."""
    async with aiohttp.ClientSession() as session:
        url = f"http://{ollama_base_url}:{ollama_port}/api/{action}"
        
        if action == "pull":
            data = json.dumps({"name": model_name})
            headers = {'Content-Type': 'application/json'}
            logging.info(f"Pulling model: {model_name}")
            
            async with session.post(url, data=data, headers=headers) as response:
                logging.info(f"Pull model response status: {response.status}")
                if response.status != 200:
                    response_text = await response.text()
                    logging.error(f"Pull model error: {response_text}")
                return response
                
        elif action == "delete":
            data = json.dumps({"name": model_name})
            headers = {'Content-Type': 'application/json'}
            
            async with session.delete(url, data=data, headers=headers) as response:
                logging.info(f"Delete model response status: {response.status}")
                return response
        else:
            logging.error(f"Unsupported model management action: {action}")
            return None

async def model_list() -> List[Dict[str, Any]]:
    """Get list of available Ollama models."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://{ollama_base_url}:{ollama_port}/api/tags"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    logging.error(f"Failed to get model list: {response.status}")
                    return []
    except Exception as e:
        logging.error(f"Error getting model list: {e}")
        return []

async def generate(payload: dict, modelname: str, prompt: str):
    """Generate response from Ollama API with improved error handling."""
    client_timeout = ClientTimeout(total=int(timeout))
    retry_count = 3
    retry_delay = 1
    
    for attempt in range(retry_count):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                url = f"http://{ollama_base_url}:{ollama_port}/api/chat"
                
                # Prepare the payload
                ollama_payload = {
                    "model": modelname,
                    "messages": payload.get("messages", []),
                    "stream": payload.get("stream", True)
                }
                
                # Add options if needed
                if "options" in payload:
                    ollama_payload["options"] = payload["options"]
                
                logging.debug(f"Ollama API request: {json.dumps(ollama_payload, indent=2)}")
                
                async with session.post(url, json=ollama_payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"API Error: {response.status} - {error_text}")
                        
                        # Retry on certain errors
                        if response.status in [502, 503, 504] and attempt < retry_count - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"API Error: {error_text}"
                        )
                    
                    # Stream the response
                    buffer = b""
                    async for chunk in response.content.iter_any():
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    yield data
                                except json.JSONDecodeError as e:
                                    logging.error(f"JSON Decode Error: {e}")
                                    logging.error(f"Problematic line: {line}")
                    break  # Success, exit retry loop
                    
        except aiohttp.ClientError as e:
            logging.error(f"Client Error during request (attempt {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                raise
        except Exception as e:
            logging.error(f"Unexpected error during generation: {e}")
            raise

# Permission decorators
def perms_allowed(func):
    """Decorator to check if user is allowed to use the bot."""
    @wraps(func)
    async def wrapper(message: types.Message = None, query: types.CallbackQuery = None):
        # Get user_id and chat info
        if message:
            user_id = message.from_user.id
            chat = message.chat
        elif query:
            user_id = query.from_user.id
            chat = query.message.chat if query.message else None
        else:
            return
        
        # Check permissions
        if user_id in admin_ids or user_id in allowed_ids:
            # User is explicitly allowed
            if message:
                return await func(message)
            elif query:
                return await func(query)
        else:
            # Check group permissions
            if chat and chat.type in ["supergroup", "group"] and allow_all_users_in_groups:
                if message:
                    return await func(message)
                elif query:
                    return await func(query)
            else:
                # Access denied
                response_text = "⛔ Access Denied. Please contact an administrator."
                if message:
                    await message.answer(response_text)
                    logging.warning(
                        f"Access denied for {message.from_user.full_name} ({user_id})"
                    )
                elif query:
                    await query.answer(response_text, show_alert=True)
                    logging.warning(
                        f"Access denied for {query.from_user.full_name} ({user_id})"
                    )
    return wrapper

def perms_admins(func):
    """Decorator to check if user is an admin."""
    @wraps(func)
    async def wrapper(message: types.Message = None, query: types.CallbackQuery = None):
        # Get user_id
        if message:
            user_id = message.from_user.id
        elif query:
            user_id = query.from_user.id
        else:
            return
        
        # Check admin permissions
        if user_id in admin_ids:
            if message:
                return await func(message)
            elif query:
                return await func(query)
        else:
            # Access denied
            response_text = "⛔ Admin access required."
            if message:
                await message.answer(response_text)
                logging.warning(
                    f"Admin access denied for {message.from_user.full_name} ({user_id})"
                )
            elif query:
                await query.answer(response_text, show_alert=True)
                logging.warning(
                    f"Admin access denied for {query.from_user.full_name} ({user_id})"
                )
    return wrapper

# Context lock class
class contextLock:
    """Async context manager for thread-safe operations."""
    def __init__(self):
        self.lock = Lock()

    async def __aenter__(self):
        await self.lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        self.lock.release()

# Utility functions
async def check_ollama_connection() -> bool:
    """Check if Ollama API is accessible."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://{ollama_base_url}:{ollama_port}/api/version"
            async with session.get(url, timeout=ClientTimeout(total=5)) as response:
                return response.status == 200
    except:
        return False

async def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://{ollama_base_url}:{ollama_port}/api/show"
            data = json.dumps({"name": model_name})
            headers = {'Content-Type': 'application/json'}
            
            async with session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                return None
    except Exception as e:
        logging.error(f"Error getting model info: {e}")
        return None

# Cleanup function
async def cleanup():
    """Cleanup resources on shutdown."""
    await db_pool.close()
    logging.info("Database connections closed")

# Initialize allowed_ids on module load
if not allowed_ids:
    try:
        allowed_ids = load_allowed_ids_from_db()
    except Exception as e:
        logging.error(f"Failed to load allowed IDs from database: {e}")
        allowed_ids = []
