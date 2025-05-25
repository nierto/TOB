#!/usr/bin/env python3
"""
Extensions module for Ollama Telegram Bot
Provides advanced features like conversation compression and clean-slate messaging
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum

# Import from your interactions module
from func.interactions import generate

class ConversationMode(Enum):
    """Conversation handling modes"""
    NORMAL = "normal"  # Standard context accumulation
    CLEAN_SLATE = "clean_slate"  # Each message is independent
    COMPRESSED = "compressed"  # Dynamic summarization with SPR

@dataclass
class SPRSummary:
    """Structured summary of a conversation turn"""
    timestamp: str
    title: str
    user_intent: str
    model_summary: str
    
    def to_context_string(self) -> str:
        """Convert to context window format"""
        return (
            f"<SPR TIMESTAMP: {self.timestamp} TITLE: {self.title}>\n"
            f"User: {self.user_intent}\n"
            f"Model: {self.model_summary}\n"
            f"</SPR>"
        )

class ConversationCompressor:
    """Handles dynamic conversation compression using SPROPTIMIZER"""
    
    def __init__(self, optimizer_model: str = "SPROPTIMIZER:latest"):
        self.optimizer_model = optimizer_model
        self.compression_prompt_template = """Compress the exchange below into sparse priming form.

		Context:
		User: {user_message}
		Assistant: {assistant_response}

		Extract:
		1. TITLE — 1–5 words capturing topic or action.
		2. USER_INTENT — one-sentence intent, goal, or question.
		3. RESPONSE_SUMMARY — abstract, high-salience synthesis of assistant's reply (2–3 lines max).

		Constraints:
		• No filler or surface phrasing.
		• Discard pleasantries, hedging, and repetition.
		• Prioritize abstraction, causality, conceptual anchors.

		Output format:
		TITLE: ...
		USER_INTENT: ...
		RESPONSE_SUMMARY: ...
		"""


    async def compress_exchange(
        self, 
        user_message: str, 
        assistant_response: str,
        timestamp: Optional[datetime] = None
    ) -> SPRSummary:
        """Compress a single conversation exchange into an SPR summary"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Prepare prompt for SPROPTIMIZER
        prompt = self.compression_prompt_template.format(
            user_message=user_message,
            assistant_response=assistant_response
        )
        
        # Generate summary using SPROPTIMIZER
        payload = {
            "model": self.optimizer_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a conversation summarizer. Extract only the most essential information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
        
        try:
            full_response = ""
            async for response_data in generate(payload, self.optimizer_model, prompt):
                msg = response_data.get("message", {})
                chunk = msg.get("content", "")
                full_response += chunk
                
                if response_data.get("done"):
                    break
            
            # Parse the response
            summary = self._parse_summary_response(full_response, timestamp)
            return summary
            
        except Exception as e:
            logging.error(f"Error compressing exchange: {e}")
            # Fallback summary
            return SPRSummary(
                timestamp=timestamp.isoformat(),
                title="Exchange",
                user_intent=user_message[:100] + "..." if len(user_message) > 100 else user_message,
                model_summary=assistant_response[:200] + "..." if len(assistant_response) > 200 else assistant_response
            )
    
    def _parse_summary_response(self, response: str, timestamp: datetime) -> SPRSummary:
        """Parse the SPROPTIMIZER response into structured summary"""
        
        # Extract components using regex
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        intent_match = re.search(r'USER_INTENT:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        summary_match = re.search(r'RESPONSE_SUMMARY:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        
        title = title_match.group(1).strip() if title_match else "Conversation"
        user_intent = intent_match.group(1).strip() if intent_match else "User query"
        model_summary = summary_match.group(1).strip() if summary_match else "Model response"
        
        # Clean up the summary
        model_summary = re.sub(r'\s+', ' ', model_summary)  # Normalize whitespace
        
        return SPRSummary(
            timestamp=timestamp.isoformat(),
            title=title,
            user_intent=user_intent,
            model_summary=model_summary
        )

class EnhancedConversationManager:
    """Manages conversation context with different modes"""
    
    def __init__(self):
        self.compressor = ConversationCompressor()
        self.user_modes: Dict[int, ConversationMode] = {}  # user_id -> mode
        self.compressed_history: Dict[int, List[SPRSummary]] = {}  # user_id -> summaries
        self.pending_compressions: Dict[int, Tuple[str, str]] = {}  # user_id -> (user_msg, assistant_msg)
        
    def set_user_mode(self, user_id: int, mode: ConversationMode):
        """Set conversation mode for a user"""
        self.user_modes[user_id] = mode
        logging.info(f"Set conversation mode for user {user_id}: {mode.value}")
        
    def get_user_mode(self, user_id: int) -> ConversationMode:
        """Get conversation mode for a user"""
        return self.user_modes.get(user_id, ConversationMode.NORMAL)
    
    async def prepare_context(
        self, 
        user_id: int, 
        current_messages: List[Dict[str, Any]],
        new_user_message: str
    ) -> List[Dict[str, Any]]:
        """Prepare conversation context based on user mode"""
        
        mode = self.get_user_mode(user_id)
        
        if mode == ConversationMode.CLEAN_SLATE:
            # Return only system message (if any) and new user message
            system_messages = [msg for msg in current_messages if msg.get("role") == "system"]
            return system_messages + [{
                "role": "user",
                "content": new_user_message
            }]
            
        elif mode == ConversationMode.COMPRESSED:
            # Compress previous exchange if exists
            if user_id in self.pending_compressions:
                prev_user, prev_assistant = self.pending_compressions[user_id]
                summary = await self.compressor.compress_exchange(prev_user, prev_assistant)
                
                if user_id not in self.compressed_history:
                    self.compressed_history[user_id] = []
                self.compressed_history[user_id].append(summary)
                
                # Keep only last N summaries to prevent context overflow
                max_summaries = 10
                if len(self.compressed_history[user_id]) > max_summaries:
                    self.compressed_history[user_id] = self.compressed_history[user_id][-max_summaries:]
                
                del self.pending_compressions[user_id]
            
            # Build context with compressed history
            messages = []
            
            # Add system message if exists
            system_messages = [msg for msg in current_messages if msg.get("role") == "system"]
            messages.extend(system_messages)
            
            # Add compressed history as context
            if user_id in self.compressed_history and self.compressed_history[user_id]:
                compressed_context = "\n\n".join(
                    summary.to_context_string() 
                    for summary in self.compressed_history[user_id]
                )
                messages.append({
                    "role": "system",
                    "content": f"Previous conversation context (compressed):\n{compressed_context}"
                })
            
            # Add the new user message
            messages.append({
                "role": "user",
                "content": new_user_message
            })
            
            return messages
            
        else:  # NORMAL mode
            return current_messages
    
    def mark_for_compression(self, user_id: int, user_message: str, assistant_response: str):
        """Mark an exchange for compression in the next turn"""
        if self.get_user_mode(user_id) == ConversationMode.COMPRESSED:
            self.pending_compressions[user_id] = (user_message, assistant_response)
    
    def get_mode_description(self, mode: ConversationMode) -> str:
        """Get human-readable description of a mode"""
        descriptions = {
            ConversationMode.NORMAL: "🔄 Normal - Full conversation history",
            ConversationMode.CLEAN_SLATE: "🆕 Clean Slate - Each message independent",
            ConversationMode.COMPRESSED: "🗜️ Compressed - Smart summarization with SPROPTIMIZER"
        }
        return descriptions.get(mode, "Unknown mode")
    
    def get_compressed_history_summary(self, user_id: int) -> str:
        """Get a summary of compressed history for display"""
        if user_id not in self.compressed_history or not self.compressed_history[user_id]:
            return "No compressed history yet."
        
        summaries = self.compressed_history[user_id]
        summary_text = f"📚 Compressed History ({len(summaries)} exchanges):\n\n"
        
        for i, summary in enumerate(summaries[-5:], 1):  # Show last 5
            summary_text += f"{i}. **{summary.title}** ({summary.timestamp[:10]})\n"
            summary_text += f"   Q: {summary.user_intent[:50]}...\n"
            summary_text += f"   A: {summary.model_summary[:50]}...\n\n"
        
        if len(summaries) > 5:
            summary_text += f"... and {len(summaries) - 5} more exchanges"
        
        return summary_text

# Global instance
conversation_manager = EnhancedConversationManager()

# Keyboard builders for mode selection
def get_conversation_mode_keyboard():
    """Create keyboard for conversation mode selection"""
    from aiogram.utils.keyboard import InlineKeyboardBuilder
    from aiogram import types
    
    kb = InlineKeyboardBuilder()
    
    for mode in ConversationMode:
        kb.row(
            types.InlineKeyboardButton(
                text=conversation_manager.get_mode_description(mode),
                callback_data=f"set_mode_{mode.value}"
            )
        )
    
    kb.row(
        types.InlineKeyboardButton(
            text="📊 View Compressed History",
            callback_data="view_compressed_history"
        )
    )
    
    return kb

# Helper functions for integration
async def handle_mode_selection(query, user_id: int, mode_value: str):
    """Handle conversation mode selection callback"""
    try:
        mode = ConversationMode(mode_value)
        conversation_manager.set_user_mode(user_id, mode)
        
        # Clear compressed history when switching modes
        if mode != ConversationMode.COMPRESSED and user_id in conversation_manager.compressed_history:
            del conversation_manager.compressed_history[user_id]
        
        await query.answer(f"✅ Switched to {conversation_manager.get_mode_description(mode)}")
        
        # Update message with confirmation
        await query.message.edit_text(
            f"Conversation mode updated!\n\nCurrent mode: {conversation_manager.get_mode_description(mode)}",
        )
        
    except ValueError:
        await query.answer("❌ Invalid mode selected", show_alert=True)

async def show_compressed_history(query, user_id: int):
    """Show compressed conversation history"""
    summary = conversation_manager.get_compressed_history_summary(user_id)
    await query.message.answer(summary, parse_mode="Markdown")
    await query.answer("📚 Showing compressed history")

# Stats tracking
class ConversationStats:
    """Track conversation statistics"""
    
    def __init__(self):
        self.message_counts: Dict[int, int] = {}
        self.compression_savings: Dict[int, int] = {}  # Characters saved
        
    def add_message(self, user_id: int, original_length: int, compressed_length: int = 0):
        """Track a message and compression savings"""
        self.message_counts[user_id] = self.message_counts.get(user_id, 0) + 1
        
        if compressed_length > 0:
            savings = original_length - compressed_length
            self.compression_savings[user_id] = self.compression_savings.get(user_id, 0) + savings
    
    def get_stats(self, user_id: int) -> Dict[str, Any]:
        """Get statistics for a user"""
        return {
            "total_messages": self.message_counts.get(user_id, 0),
            "compression_savings": self.compression_savings.get(user_id, 0),
            "avg_compression_ratio": (
                self.compression_savings.get(user_id, 0) / max(self.message_counts.get(user_id, 1), 1)
            )
        }

# Global stats instance
conversation_stats = ConversationStats()

# Export all components
__all__ = [
    'ConversationMode',
    'SPRSummary',
    'ConversationCompressor',
    'EnhancedConversationManager',
    'conversation_manager',
    'get_conversation_mode_keyboard',
    'handle_mode_selection',
    'show_compressed_history',
    'ConversationStats',
    'conversation_stats'
]
