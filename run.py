#!/usr/bin/env python3
"""
Enhanced Ollama bot with conversation compression and clean-slate options
"""

import logging
import os
import sys

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters.command import Command, CommandStart
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from func.interactions import *

# Add extensions to path and import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add current directory to Python path

try:
    from extensions import (
        conversation_manager, 
        get_conversation_mode_keyboard,
        handle_mode_selection,
        show_compressed_history,
        conversation_stats,
        ConversationMode
    )
    EXTENSIONS_AVAILABLE = True
    logging.info("Extensions module loaded successfully")
except ImportError as e:
    logging.warning(f"Extensions module not available: {e}")
    logging.info(f"Current directory: {current_dir}")
    logging.info(f"Files in directory: {os.listdir(current_dir)}")
    EXTENSIONS_AVAILABLE = False

import asyncio
import traceback
import io
import base64
import sqlite3
import re

bot = Bot(token=token)
dp = Dispatcher()

# Global variables
ACTIVE_CHATS = {}
ACTIVE_CHATS_LOCK = contextLock()
modelname = os.getenv("INITMODEL", "llama2")
mention = None
selected_prompt_id = None
CHAT_TYPE_GROUP = "group"
CHAT_TYPE_SUPERGROUP = "supergroup"

# Model selection mappings (to handle long model names)
model_choices = {}
delete_model_choices = {}

# Utility functions
def split_into_chunks(text, max_length=4000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, name TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  role TEXT,
                  content TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS system_prompts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  prompt TEXT,
                  is_global BOOLEAN,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

def register_user(user_id, user_name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?)", (user_id, user_name))
    conn.commit()
    conn.close()

def save_chat_message(user_id, role, content):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_id, role, content) VALUES (?, ?, ?)",
              (user_id, role, content))
    conn.commit()
    conn.close()

def get_context_key(message: types.Message):
    if message.chat.type == "private":
        return message.from_user.id
    else:
        return message.chat.id

def escape_markdown(text: str) -> str:
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(special_chars), r'\\\1', text)

async def get_bot_info():
    global mention
    if mention is None:
        get = await bot.get_me()
        mention = f"@{get.username}"
    return mention

# Command handlers
@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    # Create keyboard
    start_kb = InlineKeyboardBuilder()
    start_kb.row(
        types.InlineKeyboardButton(text="ℹ️ About", callback_data="about"),
        types.InlineKeyboardButton(text="⚙️ Settings", callback_data="settings"),
        types.InlineKeyboardButton(text="📝 Register", callback_data="register"),
    )
    
    start_message = f"Welcome, <b>{message.from_user.full_name}</b>! 🤖\n\nI'm your AI assistant powered by Ollama."
    
    await message.answer(
        start_message,
        parse_mode=ParseMode.HTML,
        reply_markup=start_kb.as_markup(),
        disable_web_page_preview=True,
    )

@dp.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    help_text = """📚 <b>Available Commands:</b>

/start - Initialize the bot
/reset - Clear current conversation
/history - View recent messages
/pullmodel <name> - Download new model
/addglobalprompt <text> - Add global prompt
/addprivateprompt <text> - Add private prompt

<b>Current model:</b> <code>{}</code>

<b>💡 Tips:</b>
• Send images for visual analysis
• Use @mention in groups
• Messages are automatically saved""".format(modelname)
    
    await message.answer(help_text, parse_mode=ParseMode.HTML)

# Keep all your existing command handlers...
@dp.message(Command("reset"))
async def command_reset_handler(message: Message) -> None:
    user_id = message.from_user.id
    if user_id in allowed_ids:
        context_key = get_context_key(message)
        async with ACTIVE_CHATS_LOCK:
            if context_key in ACTIVE_CHATS:
                ACTIVE_CHATS.pop(context_key)
        logging.info(f"Chat has been reset for {message.from_user.first_name}")
        await bot.send_message(
            chat_id=message.chat.id,
            text="✅ Chat has been reset",
        )

@dp.message(Command("history"))
async def command_get_context_handler(message: Message) -> None:
    user_id = message.from_user.id
    context_key = get_context_key(message)

    if user_id in allowed_ids:
        if context_key in ACTIVE_CHATS:
            messages = ACTIVE_CHATS.get(context_key)["messages"]
            context = ""
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                # Skip system messages in history display
                if role.lower() != "system":
                    context += f"*{role}*: {content}\n\n"

            if context:
                chunks = split_into_chunks(context, 4000)
                for chunk in chunks:
                    await safe_send_message(message.chat.id, chunk, bot)
            else:
                await bot.send_message(
                    chat_id=message.chat.id,
                    text="No conversation history available",
                )
        else:
            await bot.send_message(
                chat_id=message.chat.id,
                text="No chat history available",
            )

@dp.message(Command("mode"))
async def command_mode_handler(message: Message) -> None:
    if EXTENSIONS_AVAILABLE:
        user_id = message.from_user.id
        current_mode = conversation_manager.get_user_mode(user_id)
        
        text = f"💬 <b>Conversation Mode Settings</b>\n\n"
        text += f"Current mode: {conversation_manager.get_mode_description(current_mode)}\n\n"
        text += "Select a mode:"
        
        await message.answer(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=get_conversation_mode_keyboard().as_markup()
        )
    else:
        await message.answer("⚠️ Extensions module not available. Please ensure extensions.py is in the same directory.")

# Callback handlers
@dp.callback_query(lambda query: query.data == "register")
async def register_callback_handler(query: types.CallbackQuery):
    user_id = query.from_user.id
    user_name = query.from_user.full_name
    register_user(user_id, user_name)
    await query.answer("✅ You have been registered successfully!")

@dp.callback_query(lambda query: query.data == "settings")
async def settings_callback_handler(query: types.CallbackQuery):
    settings_kb = InlineKeyboardBuilder()
    settings_kb.row(
        types.InlineKeyboardButton(text="🔄 Switch LLM", callback_data="switchllm"),
        types.InlineKeyboardButton(text="🗑️ Delete LLM", callback_data="delete_model"),
    )
    settings_kb.row(
        types.InlineKeyboardButton(text="📋 Select System Prompt", callback_data="select_prompt"),
        types.InlineKeyboardButton(text="🗑️ Delete System Prompt", callback_data="delete_prompt"),
    )
    
    # Add conversation mode button if extensions available
    if EXTENSIONS_AVAILABLE:
        settings_kb.row(
            types.InlineKeyboardButton(text="💬 Conversation Mode", callback_data="conversation_mode"),
        )
    
    settings_kb.row(
        types.InlineKeyboardButton(text="📋 List Users", callback_data="list_users"),
    )
    
    await bot.send_message(
        chat_id=query.message.chat.id,
        text="⚙️ <b>Settings</b>\n\nChoose an option:",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=settings_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data == "about")
@perms_admins
async def about_callback_handler(query: types.CallbackQuery):
    dotenv_model = os.getenv("INITMODEL")
    global modelname
    text = (
        f"<b>🤖 Ollama Telegram Bot</b>\n\n"
        f"Currently using: <code>{modelname}</code>\n"
        f"Default in .env: <code>{dotenv_model}</code>\n\n"
        f"This project is under <a href='https://github.com/ruecat/ollama-telegram/blob/main/LICENSE'>MIT License</a>\n"
        f"<a href='https://github.com/ruecat/ollama-telegram'>Source Code</a>"
    )
    await bot.send_message(
        chat_id=query.message.chat.id,
        text=text,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

# Keep all your other callback handlers...
@dp.callback_query(lambda query: query.data == "switchllm")
async def switchllm_callback_handler(query: types.CallbackQuery):
    try:
        models = await model_list()
        
        if not models:
            await query.answer("No models available", show_alert=True)
            return
            
        switchllm_builder = InlineKeyboardBuilder()
        
        # Store models in a temporary dict with indices
        global model_choices
        model_choices = {}
        
        for idx, model in enumerate(models):
            mname = model["name"]
            model_choices[str(idx)] = mname  # Store mapping
            
            modelfamilies = ""
            if model.get("details", {}).get("families"):
                modelicon = {"llama": "🦙", "clip": "📷"}
                try:
                    modelfamilies = "".join(
                        [modelicon.get(family, "✨") for family in model["details"]["families"]]
                    )
                except KeyError:
                    modelfamilies = "✨"
            
            # Truncate model name if too long for display
            display_name = mname if len(mname) <= 30 else mname[:27] + "..."
            
            switchllm_builder.row(
                types.InlineKeyboardButton(
                    text=f"{display_name} {modelfamilies}", 
                    callback_data=f"m_{idx}"  # Use index instead of name
                )
            )
        
        await query.message.edit_text(
            f"{len(models)} models available.\n🦙 = Regular\n🦙📷 = Multimodal", 
            reply_markup=switchllm_builder.as_markup(),
        )
    except Exception as e:
        logging.error(f"Error in switchllm_callback_handler: {e}")
        await query.answer(f"Error loading models: {str(e)}", show_alert=True)

@dp.callback_query(lambda query: query.data.startswith("m_"))
async def model_callback_handler(query: types.CallbackQuery):
    try:
        global modelname, model_choices
        
        # Extract index from callback data
        idx = query.data.split("m_")[1]
        
        # Get actual model name from stored mapping
        if 'model_choices' in globals() and idx in model_choices:
            modelname = model_choices[idx]
            await query.answer(f"✅ Chosen model: {modelname}")
            logging.info(f"Model switched to: {modelname}")
        else:
            await query.answer("❌ Model selection expired. Please try again.", show_alert=True)
    except Exception as e:
        logging.error(f"Error in model_callback_handler: {e}")
        await query.answer(f"Error selecting model: {str(e)}", show_alert=True)

# Add new callback handlers for conversation modes
if EXTENSIONS_AVAILABLE:
    @dp.callback_query(lambda query: query.data == "conversation_mode")
    async def conversation_mode_callback_handler(query: types.CallbackQuery):
        user_id = query.from_user.id
        current_mode = conversation_manager.get_user_mode(user_id)
        
        text = f"💬 <b>Conversation Mode Settings</b>\n\n"
        text += f"Current mode: {conversation_manager.get_mode_description(current_mode)}\n\n"
        text += "Select a mode:"
        
        await query.message.edit_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=get_conversation_mode_keyboard().as_markup()
        )
    
    @dp.callback_query(lambda query: query.data.startswith("set_mode_"))
    async def set_mode_callback_handler(query: types.CallbackQuery):
        mode_value = query.data.split("set_mode_")[1]
        await handle_mode_selection(query, query.from_user.id, mode_value)
    
    @dp.callback_query(lambda query: query.data == "view_compressed_history")
    async def view_compressed_history_handler(query: types.CallbackQuery):
        await show_compressed_history(query, query.from_user.id)

# Message handling
@dp.message()
@perms_allowed
async def handle_message(message: types.Message):
    await get_bot_info()
    if message.chat.type == "private":
        await ollama_request(message)
        return

    if await is_mentioned_in_group_or_supergroup(message):
        thread = await collect_message_thread(message)
        prompt = format_thread_for_prompt(thread)
        await ollama_request(message, prompt)

# Helper functions
async def is_mentioned_in_group_or_supergroup(message: types.Message):
    if message.chat.type not in ["group", "supergroup"]:
        return False
    global mention
    is_mentioned = (
        (message.text and mention and message.text.startswith(mention)) or
        (message.caption and mention and message.caption.startswith(mention))
    )
    is_reply_to_bot = (
        message.reply_to_message and
        message.reply_to_message.from_user.id == bot.id
    )
    return is_mentioned or is_reply_to_bot

async def collect_message_thread(message: types.Message, thread=None):
    if thread is None:
        thread = []
    thread.insert(0, message)
    if message.reply_to_message:
        await collect_message_thread(message.reply_to_message, thread)
    return thread

def format_thread_for_prompt(thread):
    prompt = "Conversation thread:\n\n"
    for msg in thread:
        sender = "User" if msg.from_user.id != bot.id else "Assistant"
        content = msg.text or msg.caption or "[No text content]"
        prompt += f"{sender}: {content}\n\n"
    return prompt.strip()

async def process_image(message):
    image_base64 = ""
    if message.content_type == "photo":
        image_buffer = io.BytesIO()
        await bot.download(message.photo[-1], destination=image_buffer)
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return image_base64

def clean_llm_response(response_text):
    """Remove system prompts and instructions from LLM response"""
    # Common patterns that indicate system prompts
    patterns_to_remove = [
        r'<OVERSEER>.*?</OVERSEER>',
        r'@AGENTIQUE-OVERSEER:.*?\n',
        r'</?antml:thinking>',
        r'SYSTEM:.*?\n',
        r'Assistant:',
        r'Human:',
        r'<\|.*?\|>',  # Common template markers
        r'\[INST\].*?\[/INST\]',  # Instruction markers
    ]
    
    cleaned = response_text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

async def add_prompt_to_active_chats(message, prompt, image_base64, modelname, system_prompt=None):
    context_key = get_context_key(message)
    user_id = message.from_user.id
    
    async with ACTIVE_CHATS_LOCK:
        messages = []
        existing = ACTIVE_CHATS.get(context_key, {}).get('messages', [])
        
        # If extensions available, prepare context based on mode
        if EXTENSIONS_AVAILABLE:
            mode = conversation_manager.get_user_mode(user_id)
            
            if mode == ConversationMode.CLEAN_SLATE:
                # Only keep system prompt for clean slate mode
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt.strip()
                    })
            elif mode == ConversationMode.COMPRESSED:
                # Use compressed context
                messages = await conversation_manager.prepare_context(
                    user_id, existing, prompt
                )
                # Add system prompt if not already included
                if system_prompt and not any(msg.get('role') == 'system' and system_prompt in msg.get('content', '') for msg in messages):
                    messages.insert(0, {
                        "role": "system",
                        "content": system_prompt.strip()
                    })
                # Return early as prepare_context already added the user message
                ACTIVE_CHATS[context_key] = {
                    "model": modelname,
                    "messages": messages,
                    "stream": True,
                }
                return
            else:
                # Normal mode - use existing logic
                if system_prompt:
                    existing_system_messages = [msg for msg in existing if msg.get('role') == 'system']
                    if not existing_system_messages:
                        messages.append({
                            "role": "system",
                            "content": system_prompt.strip()
                        })
                # Add existing messages
                if context_key in ACTIVE_CHATS:
                    messages.extend([msg for msg in existing if msg.get('role') != 'system'])
        else:
            # Fallback to original logic if extensions not available
            if system_prompt:
                existing_system_messages = [msg for msg in existing if msg.get('role') == 'system']
                if not existing_system_messages:
                    messages.append({
                        "role": "system",
                        "content": system_prompt.strip()
                    })
            
            if context_key in ACTIVE_CHATS:
                messages.extend([msg for msg in existing if msg.get('role') != 'system'])
        
        # Add new user message
        messages.append({
            "role": "user",
            "content": prompt,
            "images": ([image_base64] if image_base64 else []),
        })
        
        # Update active chat
        ACTIVE_CHATS[context_key] = {
            "model": modelname,
            "messages": messages,
            "stream": True,
        }

async def handle_response(message, response_data, full_response):
    full_response_stripped = full_response.strip()
    if full_response_stripped == "":
        return False
        
    if response_data.get("done"):
        # Clean the response before sending
        cleaned_response = clean_llm_response(full_response_stripped)
        
        # Calculate timing
        eval_duration = response_data.get('eval_duration', 0) / 1e9
        total_duration = response_data.get('total_duration', 0) / 1e9
        load_duration = response_data.get('load_duration', 0) / 1e9
        
        # Create more informative footer
        if total_duration > 0:
            other_duration = total_duration - eval_duration - load_duration
            text = f"{cleaned_response}\n\n"
            text += f"⚙️ {modelname}\n"
            text += f"⏱ Generated in {eval_duration:.1f}s"
            if other_duration > 1:
                text += f" (total: {total_duration:.1f}s)"
        else:
            text = cleaned_response
        
        await send_response(message, text)

        # Update context with cleaned response
        context_key = get_context_key(message)
        user_id = message.from_user.id
        
        async with ACTIVE_CHATS_LOCK:
            if ACTIVE_CHATS.get(context_key) is not None:
                # Only add to context if not in clean slate mode
                if EXTENSIONS_AVAILABLE:
                    mode = conversation_manager.get_user_mode(user_id)
                    if mode != ConversationMode.CLEAN_SLATE:
                        ACTIVE_CHATS[context_key]["messages"].append(
                            {"role": "assistant", "content": cleaned_response}
                        )
                        
                        # Mark for compression if in compressed mode
                        if mode == ConversationMode.COMPRESSED:
                            # Get the last user message
                            user_messages = [msg for msg in ACTIVE_CHATS[context_key]["messages"] if msg["role"] == "user"]
                            if user_messages:
                                last_user_msg = user_messages[-1]["content"]
                                conversation_manager.mark_for_compression(
                                    user_id, last_user_msg, cleaned_response
                                )
                else:
                    # Normal behavior if no extensions
                    ACTIVE_CHATS[context_key]["messages"].append(
                        {"role": "assistant", "content": cleaned_response}
                    )
        
        # Track stats if available
        if EXTENSIONS_AVAILABLE:
            original_length = len(full_response_stripped)
            compressed_length = len(cleaned_response)
            conversation_stats.add_message(user_id, original_length, compressed_length)
        
        logging.info(f"[Response]: Generated response for {message.from_user.first_name}")
        return True
    return False

async def safe_send_message(chat_id, text, bot):
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        error_str = str(e)
        if "can't parse entities" in error_str.lower():
            escaped_text = escape_markdown(text)
            await bot.send_message(chat_id=chat_id, text=escaped_text, parse_mode=ParseMode.MARKDOWN_V2)
        else:
            # Last resort - send without formatting
            await bot.send_message(chat_id=chat_id, text=text)

async def safe_edit_message(message, text, bot):
    try:
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=message.message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        # If edit fails, send new message
        await safe_send_message(message.chat.id, text, bot)

async def send_response(message, text):
    chunks = split_into_chunks(text, 4000)
    if len(chunks) == 0:
        return

    if message.chat.type == "private" or message.chat.id < 0:
        await safe_send_message(message.chat.id, chunks[0], bot)
    else:
        await safe_edit_message(message, chunks[0], bot)

    for chunk in chunks[1:]:
        await safe_send_message(message.chat.id, chunk, bot)

# Main request handler
async def ollama_request(message: types.Message, prompt: str = None):
    try:
        full_response = ""
        await bot.send_chat_action(message.chat.id, "typing")
        image_base64 = await process_image(message)

        if prompt is None:
            prompt = message.text or message.caption

        # Get system prompt if selected
        system_prompt = None
        if selected_prompt_id is not None:
            system_prompts = get_system_prompts(user_id=message.from_user.id)
            if system_prompts:
                for sp in system_prompts:
                    if sp[0] == selected_prompt_id:
                        system_prompt = sp[2]
                        break

        # Save user message
        save_chat_message(message.from_user.id, "user", prompt)

        # Prepare active chat
        await add_prompt_to_active_chats(message, prompt, image_base64, modelname, system_prompt)

        logging.info(f"[OllamaAPI]: Processing '{prompt[:50]}...' for {message.from_user.first_name}")

        # Get payload
        context_key = get_context_key(message)
        payload = ACTIVE_CHATS.get(context_key)

        # Generate response
        async for response_data in generate(payload, modelname, prompt):
            msg = response_data.get("message")
            if msg is None:
                continue
            chunk = msg.get("content", "")
            full_response += chunk

            if any([c in chunk for c in ".\n!?"]) or response_data.get("done"):
                if await handle_response(message, response_data, full_response):
                    save_chat_message(message.from_user.id, "assistant", full_response)
                    break

    except Exception as e:
        logging.error(f"[OllamaAPI-ERR] Error: {traceback.format_exc()}")
        error_msg = f"❌ Something went wrong: {str(e)}"
        await safe_send_message(message.chat.id, error_msg, bot)

# Keep all your existing command handlers...
@dp.message(Command("addglobalprompt"))
async def add_global_prompt_handler(message: Message):
    prompt_text = message.text.split(maxsplit=1)[1] if len(message.text.split()) > 1 else None
    if prompt_text:
        add_system_prompt(message.from_user.id, prompt_text, True)
        await safe_send_message(message.chat.id, f"✅ Global prompt added successfully.", bot)
    else:
        await bot.send_message(message.chat.id, "Please provide a prompt text to add.")

@dp.message(Command("addprivateprompt"))
async def add_private_prompt_handler(message: Message):
    prompt_text = message.text.split(maxsplit=1)[1] if len(message.text.split()) > 1 else None
    if prompt_text:
        add_system_prompt(message.from_user.id, prompt_text, False)
        await safe_send_message(message.chat.id, f"✅ Private prompt added successfully.", bot)
    else:
        await bot.send_message(message.chat.id, "Please provide a prompt text to add.")

@dp.message(Command("pullmodel"))
async def pull_model_handler(message: Message) -> None:
    model_name = message.text.split(maxsplit=1)[1] if len(message.text.split()) > 1 else None
    logging.info(f"Downloading {model_name}")
    if model_name:
        response = await manage_model("pull", model_name)
        if response.status == 200:
            await bot.send_message(message.chat.id, f"✅ Model '{model_name}' is being pulled.")
        else:
            await bot.send_message(message.chat.id, f"❌ Failed to pull model '{model_name}': {response.reason}")
    else:
        await bot.send_message(message.chat.id, "Please provide a model name to pull.")

# Keep all other callback handlers...
@dp.callback_query(lambda query: query.data == "select_prompt")
async def select_prompt_callback_handler(query: types.CallbackQuery):
    prompts = get_system_prompts(user_id=query.from_user.id)
    prompt_kb = InlineKeyboardBuilder()
    for prompt in prompts:
        prompt_id, _, prompt_text, _, _ = prompt
        prompt_kb.row(
            types.InlineKeyboardButton(
                text=prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text, 
                callback_data=f"prompt_{prompt_id}"
            )
        )
    await query.message.edit_text(
        f"{len(prompts)} system prompts available.", 
        reply_markup=prompt_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data.startswith("prompt_"))
async def prompt_callback_handler(query: types.CallbackQuery):
    global selected_prompt_id
    selected_prompt_id = int(query.data.split("prompt_")[1])
    await query.answer(f"✅ Selected prompt ID: {selected_prompt_id}")

@dp.callback_query(lambda query: query.data == "delete_prompt")
async def delete_prompt_callback_handler(query: types.CallbackQuery):
    prompts = get_system_prompts(user_id=query.from_user.id)
    delete_prompt_kb = InlineKeyboardBuilder()
    for prompt in prompts:
        prompt_id, _, prompt_text, _, _ = prompt
        delete_prompt_kb.row(
            types.InlineKeyboardButton(
                text=prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text, 
                callback_data=f"delete_prompt_{prompt_id}"
            )
        )
    await query.message.edit_text(
        f"{len(prompts)} system prompts available for deletion.", 
        reply_markup=delete_prompt_kb.as_markup()
    )

@dp.callback_query(lambda query: query.data.startswith("delete_prompt_"))
async def delete_prompt_confirm_handler(query: types.CallbackQuery):
    prompt_id = int(query.data.split("delete_prompt_")[1])
    delete_system_prompt(prompt_id)
    await query.answer(f"✅ Deleted prompt ID: {prompt_id}")

@dp.callback_query(lambda query: query.data == "delete_model")
async def delete_model_callback_handler(query: types.CallbackQuery):
    try:
        models = await model_list()
        
        if not models:
            await query.answer("No models available", show_alert=True)
            return
            
        delete_model_kb = InlineKeyboardBuilder()
        
        # Store models for deletion
        global delete_model_choices
        delete_model_choices = {}
        
        for idx, model in enumerate(models):
            mname = model["name"]
            delete_model_choices[str(idx)] = mname
            
            # Truncate model name if too long
            display_name = mname if len(mname) <= 30 else mname[:27] + "..."
            
            delete_model_kb.row(
                types.InlineKeyboardButton(
                    text=display_name, 
                    callback_data=f"del_m_{idx}"
                )
            )
        
        delete_model_kb.row(
            types.InlineKeyboardButton(text="❌ Cancel", callback_data="cancel_delete")
        )
        
        await query.message.edit_text(
            f"{len(models)} models available for deletion.\n⚠️ This action cannot be undone!", 
            reply_markup=delete_model_kb.as_markup()
        )
    except Exception as e:
        logging.error(f"Error in delete_model_callback_handler: {e}")
        await query.answer(f"Error loading models: {str(e)}", show_alert=True)

@dp.callback_query(lambda query: query.data.startswith("del_m_"))
async def delete_model_confirm_handler(query: types.CallbackQuery):
    try:
        global delete_model_choices
        
        idx = query.data.split("del_m_")[1]
        
        if 'delete_model_choices' in globals() and idx in delete_model_choices:
            mname = delete_model_choices[idx]
            response = await manage_model("delete", mname)
            
            if response and response.status == 200:
                await query.answer(f"✅ Deleted model: {mname}")
                await query.message.edit_text(f"✅ Successfully deleted model: {mname}")
            else:
                reason = response.reason if response else "Unknown error"
                await query.answer(f"❌ Failed to delete: {reason}", show_alert=True)
        else:
            await query.answer("❌ Model selection expired. Please try again.", show_alert=True)
    except Exception as e:
        logging.error(f"Error in delete_model_confirm_handler: {e}")
        await query.answer(f"Error deleting model: {str(e)}", show_alert=True)

@dp.callback_query(lambda query: query.data == "cancel_delete")
async def cancel_delete_handler(query: types.CallbackQuery):
    await query.message.edit_text("❌ Model deletion cancelled.")
    await query.answer("Cancelled")

@dp.callback_query(lambda query: query.data == "list_users")
@perms_admins
async def list_users_callback_handler(query: types.CallbackQuery):
    users = get_all_users_from_db()
    user_kb = InlineKeyboardBuilder()
    for user_id, user_name in users:
        user_kb.row(types.InlineKeyboardButton(text=f"{user_name} ({user_id})", callback_data=f"remove_{user_id}"))
    user_kb.row(types.InlineKeyboardButton(text="Cancel", callback_data="cancel_remove"))
    await query.message.answer("Select a user to remove:", reply_markup=user_kb.as_markup())

@dp.callback_query(lambda query: query.data.startswith("remove_"))
@perms_admins
async def remove_user_from_list_handler(query: types.CallbackQuery):
    if query.data == "remove_":  # Skip if it's just "remove_"
        return
    try:
        user_id = int(query.data.split("_")[1])
        if remove_user_from_db(user_id):
            await query.answer(f"✅ User {user_id} has been removed.")
            await query.message.edit_text(f"User {user_id} has been removed.")
        else:
            await query.answer(f"❌ User {user_id} not found.")
    except (IndexError, ValueError):
        await query.answer("❌ Invalid user ID")

@dp.callback_query(lambda query: query.data == "cancel_remove")
@perms_admins
async def cancel_remove_handler(query: types.CallbackQuery):
    await query.message.edit_text("User removal cancelled.")

# Main entry point
async def main():
    init_db()
    global allowed_ids
    allowed_ids = load_allowed_ids_from_db()
    logging.info(f"Loaded allowed_ids: {allowed_ids}")
    
    # Enhanced commands
    commands = [
        types.BotCommand(command="start", description="Start the bot"),
        types.BotCommand(command="help", description="Show help"),
        types.BotCommand(command="reset", description="Reset Chat"),
        types.BotCommand(command="history", description="View messages"),
        types.BotCommand(command="mode", description="Change conversation mode"),
        types.BotCommand(command="pullmodel", description="Pull Ollama model"),
        types.BotCommand(command="addglobalprompt", description="Add global prompt"),
        types.BotCommand(command="addprivateprompt", description="Add private prompt"),
    ]
    
    await bot.set_my_commands(commands)
    await dp.start_polling(bot, skip_update=True)

if __name__ == "__main__":
    asyncio.run(main())
