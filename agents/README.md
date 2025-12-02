# Multi-Agent Threat Processing System

Autonomous agents for processing threat reports from WhatsApp with validation and severity classification.

## Components

### 1. Validator Agent
Ensures report completeness by extracting and validating:
- **Who**: People involved
- **What**: Threat/event description
- **Where**: Location
- **When**: Time/date

Uses Named Entity Recognition (NER) and semantic similarity scoring.

### 2. Escalation Agent
Classifies threat severity using OpenRouter LLM:
- **Critical**: Immediate SMS alert (<2 min)
- **High**: 30-minute escalation window
- **Medium**: Daily intelligence digest
- **Low**: Weekly intelligence digest

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenRouter API Key

The Escalation Agent uses OpenRouter LLM for severity classification. Set your API key:

**Option 1: Environment Variable (Recommended)**
```bash
# Windows
set OPENROUTER_API_KEY=your_api_key_here

# Linux/Mac
export OPENROUTER_API_KEY=your_api_key_here
```

**Option 2: Update config.py**
Edit `agents/config.py` and set:
```python
MODEL_CONFIG = {
    "openrouter_api_key": "your_api_key_here",
    ...
}
```

### 3. Get OpenRouter API Key

1. Sign up at [OpenRouter.ai](https://openrouter.ai)
2. Get your API key from the dashboard
3. Set it as described above

## Usage

### Process Simulated Messages

```bash
cd agents
py main.py --mode simulate --count 5
```

### Process Single Message

```bash
cd agents
py main.py --mode message --message "Your threat report text here"
```

### Interactive Mode

```bash
cd agents
py main.py --mode interactive
```

## Output

Processed reports are saved as JSON files in `agents/storage/` directory. Each report includes:

- Report ID and metadata
- Validation results (completeness scores, missing fields)
- Escalation results (severity, priority, escalation window)
- Timestamps

## Configuration

Edit `agents/config.py` to customize:

- OpenRouter model selection
- Validation thresholds
- Escalation windows
- Urgency keywords

## Notes

- If OpenRouter API key is not set, the system falls back to keyword-based classification
- The system works with both Swahili and English text
- All processed reports are stored for dashboard consumption

