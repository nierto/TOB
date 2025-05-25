# 🤖 Telegram Ollama Bot (TOB)

An advanced Telegram bot for Ollama with intelligent conversation management, dynamic compression, and production-ready features.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Based on](https://img.shields.io/badge/based%20on-ruecat%2Follama--telegram-lightgrey.svg)](https://github.com/ruecat/ollama-telegram)

## 🌟 Features

### Core Capabilities
- 🧠 **Multiple Conversation Modes**
  - **Normal Mode**: Standard conversation with full context history
  - **Clean Slate Mode**: Each message is independent, no context carried over
  - **Compressed Mode**: Intelligent summarization using SPROPTIMIZER model
- 🗜️ **Dynamic Conversation Compression**: Maintains context without token overflow
- 🛡️ **Robust Error Handling**: Automatic markdown escaping and fallback mechanisms
- 📊 **Performance Metrics**: Detailed timing breakdowns for each response
- 🔧 **System Prompt Management**: Clean responses without AI instruction leakage
- 🖼️ **Multimodal Support**: Send images for analysis with vision-capable models
- 👥 **Group Chat Support**: Mention the bot or reply to its messages

### Enhanced Features
- 📈 Real-time performance metrics (generation time vs total time)
- 🔄 Hot-swappable models without restart
- 📝 Persistent conversation history with search capabilities
- 🎯 Per-user system prompts (global and private)
- 🚦 Rate limiting and spam protection
- 💾 Automatic message chunking for long responses
- 🔍 Conversation search and export functionality

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ollama-telegram-enhanced.git
   cd ollama-telegram-enhanced
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   TOKEN=your_telegram_bot_token
   OLLAMA_BASE_URL=localhost
   OLLAMA_PORT=11434
   INITMODEL=llama2
   USER_IDS=your_telegram_user_id
   ADMIN_IDS=your_telegram_user_id
   ```

4. **Pull required Ollama models**
   ```bash
   # Required for basic operation
   ollama pull llama2
   
   # Optional: For compressed conversation mode
   ollama pull SPROPTIMIZER:latest
   ```

5. **Run the bot**
   ```bash
   python run.py
   ```

## 💬 Conversation Modes

### Normal Mode (Default)
Standard conversation mode where all context is maintained:
```
User: What is Python?
Bot: Python is a high-level programming language...
User: What are its main uses?
Bot: [Remembers previous context about Python] Its main uses include...
```

### Clean Slate Mode
Each message is treated independently:
```
User: What is Python?
Bot: Python is a high-level programming language...
User: What are its main uses?
Bot: [No context] Could you specify what you'd like to know the uses of?
```

### Compressed Mode
Previous exchanges are intelligently summarized:
```
<SPR TIMESTAMP: 2024-05-24T20:45:00 TITLE: Python Basics>
User: User asked about Python programming language
Model: Explained Python as high-level interpreted language with key features
</SPR>
```

## 📋 Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize the bot and show menu |
| `/help` | Display available commands and current mode |
| `/mode` | Change conversation mode (Normal/Clean Slate/Compressed) |
| `/reset` | Clear current conversation context |
| `/history` | View recent conversation history |
| `/stats` | Display usage statistics |
| `/search <query>` | Search through your conversations |
| `/export` | Export chat history (JSON/TXT) |
| `/pullmodel <name>` | Download a new Ollama model |
| `/addglobalprompt <text>` | Add a global system prompt |
| `/addprivateprompt <text>` | Add a private system prompt |

## ⚙️ Configuration

### Advanced Settings
Additional environment variables:
```env
# Optional configurations
ALLOW_ALL_USERS_IN_GROUPS=0  # Allow all users in group chats
LOG_LEVEL=INFO                # Logging level
TIMEOUT=3000                  # Request timeout in seconds
RATE_LIMIT_MESSAGES=30        # Messages per minute limit
MAX_CONTEXT_MESSAGES=50       # Maximum context window
```

### System Requirements
- **RAM**: 8GB minimum (16GB recommended for larger models)
- **Storage**: Varies by model (3-7GB per model typically)
- **CPU**: Modern multi-core processor
- **GPU**: Optional but recommended for faster inference

## 🔧 Troubleshooting

### Common Issues

**Bot doesn't respond**
- Ensure Ollama is running: `ollama serve`
- Check model is installed: `ollama list`
- Verify bot token and user IDs in `.env`

**"BUTTON_DATA_INVALID" error**
- Fixed in this version! Long model names are automatically handled

**System prompts appearing in responses**
- Fixed! Automatic cleaning of AI instruction leakage

**Slow responses**
- Keep models loaded: `ollama run modelname`
- Consider using smaller models for faster responses
- Check the timing breakdown in bot responses

## 🏗️ Architecture

```
ollama-telegram-enhanced/
├── bot/
│   ├── run.py              # Main bot application
│   ├── extensions.py       # Conversation management module
│   └── func/
│       └── interactions.py # Ollama API interactions
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md             # This file
```

## 👥 Credits

### Original Project
This project is based on [ollama-telegram](https://github.com/ruecat/ollama-telegram) by [ruecat](https://github.com/ruecat), which provided the foundational architecture for Ollama-Telegram integration.

### Enhanced Version
**Major enhancements and revisions by**:
- **Niels Erik Toren** - Architecture improvements, conversation modes, compression system
- **Claude (Anthropic)** - Code optimization, error handling, documentation

### Key Improvements Made
1. **Conversation Management System**: Complete rewrite with multiple modes and compression
2. **Error Handling**: Robust markdown escaping and graceful fallbacks
3. **Performance Optimization**: Better response streaming and chunking
4. **UI/UX Enhancements**: Cleaner interfaces and informative feedback
5. **Bug Fixes**: Telegram callback data limits, system prompt leakage

## 📄 License

This project is licensed under the MIT License - see the original [LICENSE](LICENSE) file for details.

Based on [ruecat/ollama-telegram](https://github.com/ruecat/ollama-telegram) (MIT License)  
Enhanced version © 2024 Niels Erik Toren & Claude (Anthropic)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🔮 Future Enhancements

- [ ] Voice message transcription support
- [ ] Scheduled message summaries
- [ ] Multi-language support
- [ ] Web dashboard for statistics
- [ ] Plugin system for custom extensions
- [ ] Retrieval Augmented Generation (RAG) support

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/yourusername/ollama-telegram-enhanced/issues)
3. Create a new issue with detailed information

---

<p align="center">
  Made with ❤️ for the open-source community<br>
  Powered by <a href="https://ollama.ai/">Ollama</a> and <a href="https://python-telegram-bot.org/">python-telegram-bot</a>
</p>
