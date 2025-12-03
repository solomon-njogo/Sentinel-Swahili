# Multi-Agent Threat Processing System

Autonomous agents for processing threat reports from WhatsApp with validation and severity classification.

## WhatsApp Bot Integration

The system includes a Twilio WhatsApp bot that receives threat reports from users and processes them through the multi-agent pipeline automatically.

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

### 4. Setup Twilio WhatsApp Bot

#### 4.1. Create Twilio Account

1. Sign up for a Twilio account at [twilio.com](https://www.twilio.com)
2. Get a WhatsApp-enabled phone number:
   - Go to [Twilio Console > Messaging > Try it out > Send a WhatsApp message](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn)
   - Follow the instructions to join the sandbox or request production access
   - Note your WhatsApp number (format: `whatsapp:+14155238886`)

#### 4.2. Get Twilio Credentials

1. From your [Twilio Console Dashboard](https://console.twilio.com):
   - Copy your **Account SID**
   - Copy your **Auth Token**

#### 4.3. Configure Environment Variables

Create a `.env` file in the project root (copy from `.env.example` if available):

```bash
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Webhook Configuration
WEBHOOK_URL=https://your-ngrok-url.ngrok.io/webhook
PORT=5000

# OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**For Local Development:**
- Use [ngrok](https://ngrok.com/) to expose your local server
- Install ngrok: `choco install ngrok` (Windows) or download from [ngrok.com](https://ngrok.com/download)
- Run: `ngrok http 5000`
- Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`) and set `WEBHOOK_URL=https://abc123.ngrok.io/webhook`

#### 4.4. Configure Twilio Webhook

1. Go to [Twilio Console > Messaging > Settings > WhatsApp Sandbox Settings](https://console.twilio.com/us1/develop/sms/sandbox/whatsapp-learn)
2. Set the **When a message comes in** webhook URL to your `WEBHOOK_URL`
3. Set HTTP method to **POST**
4. Save the configuration

#### 4.5. Run the Webhook Server

```bash
cd agents
py whatsapp_webhook.py
```

The server will start on port 5000 (or the port specified in your `.env` file).

**Health Check:**
- Visit `http://localhost:5000/health` to verify the server is running

#### 4.6. Test the Bot

1. Send a WhatsApp message to your Twilio WhatsApp number
2. The bot will:
   - Receive and process your message
   - Validate completeness (who, what, where, when)
   - Classify severity (Critical, High, Medium, Low)
   - Send an automated response in Swahili
   - Store the processed report

**Example Message (Swahili):**
```
Dharura! Kuna shambulio la risasi katika Soko la Mwenge, Dar es Salaam. 
Watu wengi wamejeruhiwa. Tafadhali fika haraka. 
Tarehe: leo saa 3:00 jioni. 
Washambuliaji: Watu wawili wamevaa nguo nyeusi.
```

## Usage

### WhatsApp Bot (Production)

Run the webhook server to receive real WhatsApp messages:

```bash
cd agents
py whatsapp_webhook.py
```

Users can send threat reports directly via WhatsApp, and the system will:
- Process the report automatically
- Send confirmation and status messages
- Request missing information if needed
- Store processed reports for dashboard viewing

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

## Response Messages

The WhatsApp bot sends automated responses in Swahili:

- **Receipt Confirmation**: Acknowledges message receipt with report ID
- **Severity Acknowledgment**: Confirms threat level (Critical, High, Medium, Low)
- **Missing Information Request**: Asks for additional details if report is incomplete
- **Processing Summary**: Provides validation and severity results
- **Error Messages**: User-friendly error messages for processing failures

## Troubleshooting

### Webhook Not Receiving Messages

1. Verify ngrok is running and URL is correct
2. Check Twilio webhook URL is set correctly in Console
3. Ensure server is running and accessible
4. Check logs in `logs/whatsapp_webhook_*.log`

### Messages Not Being Processed

1. Verify environment variables are set correctly
2. Check OpenRouter API key is configured
3. Review server logs for errors
4. Ensure Twilio credentials are valid

### Responses Not Being Sent

1. Verify `TWILIO_WHATSAPP_NUMBER` is set correctly
2. Check Twilio account has sufficient credits
3. Verify WhatsApp number is approved/sandbox configured
4. Check Twilio Console for message delivery status

## Notes

- If OpenRouter API key is not set, the system falls back to keyword-based classification
- The system works with both Swahili and English text, but responses are in Swahili
- All processed reports are stored for dashboard consumption
- For production, use a proper webhook URL (not ngrok) and ensure SSL/TLS is enabled

