# Proactive Memory Demos

This repo contains testable versions of the three blog patterns:

1. Session-start scan
2. Context-trigger scan
3. Scheduled reflection scan

The core code is in `src/proactive_memory/demos.py`. Tests use fake Mem0/OpenAI-style clients, so they run without API keys:

```bash
python -m pytest
```

For a live demo, install the optional dependencies and provide your keys:

```bash
python -m pip install -e ".[live,test]"
export MEM0_API_KEY=your-mem0-key
export OPENROUTER_API_KEY=your-openrouter-key
export OPENROUTER_MODEL=openai/gpt-4o-mini
```

The implementation keeps Mem0 and the chat provider behind small injectable interfaces. That makes the behavior easy to test while preserving the same control flow as the blog examples.
