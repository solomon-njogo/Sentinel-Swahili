"""
WhatsApp webhook server for receiving threat reports via Twilio.
Processes messages through the multi-agent pipeline and sends automated responses.
"""

import sys
import os
from pathlib import Path
from typing import Optional
from flask import Flask, request, Response, jsonify
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.main import ThreatProcessingPipeline
from agents.config import TWILIO_CONFIG, FIELD_PROMPTS, LANGUAGE_CONFIG
from agents.data_models import ValidationStatus, SeverityLevel
from src.utils.logger import setup_logging_with_increment, get_logger
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Twilio client
twilio_client = None
if TWILIO_CONFIG["account_sid"] and TWILIO_CONFIG["auth_token"]:
    twilio_client = Client(
        TWILIO_CONFIG["account_sid"],
        TWILIO_CONFIG["auth_token"]
    )

# Initialize pipeline
pipeline = None

# Response message templates in Swahili
RESPONSE_MESSAGES = {
    "receipt_confirmation": (
        "Asante! Taarifa yako imepokelewa. "
        "Tunaichambua sasa hivi. Nambari ya taarifa: {report_id}"
    ),
    "processing_complete": (
        "Taarifa yako imechambuliwa. "
        "Kiwango cha hatari: {severity}. "
        "Kamili: {completeness:.0%}"
    ),
    "missing_info": (
        "Asante kwa taarifa yako. "
        "Tafadhali toa maelezo zaidi kuhusu: {missing_fields}. "
        "Hii itasaidia kuchambua taarifa yako kwa usahihi zaidi."
    ),
    "critical_acknowledgment": (
        "Dharura imetambuliwa! "
        "Taarifa yako imepokelewa na inachambuliwa haraka. "
        "Tutachukua hatua za haraka."
    ),
    "high_acknowledgment": (
        "Hatari kubwa imetambuliwa. "
        "Taarifa yako inachambuliwa na tutachukua hatua ndani ya dakika 30."
    ),
    "medium_acknowledgment": (
        "Taarifa yako imepokelewa. "
        "Itachambuliwa na kujumuishwa katika ripoti ya kila siku."
    ),
    "low_acknowledgment": (
        "Asante kwa taarifa yako. "
        "Itachambuliwa na kujumuishwa katika ripoti ya kila wiki."
    ),
    "error_generic": (
        "Samahani, kumekuwa na tatizo katika kuchakata taarifa yako. "
        "Tafadhali jaribu tena baadaye au wasiliana nasi moja kwa moja."
    ),
    "error_empty": (
        "Samahani, hujatoa taarifa yoyote. "
        "Tafadhali toa maelezo ya tishio au tukio."
    )
}


def get_response_message(key: str, **kwargs) -> str:
    """
    Get a response message template with substitutions.
    
    Args:
        key: Message template key
        **kwargs: Values to substitute in template
        
    Returns:
        Formatted message string
    """
    template = RESPONSE_MESSAGES.get(key, RESPONSE_MESSAGES["error_generic"])
    try:
        return template.format(**kwargs)
    except KeyError:
        return RESPONSE_MESSAGES["error_generic"]


def format_missing_fields(missing_fields: list) -> str:
    """
    Format missing fields list into a readable Swahili string.
    
    Args:
        missing_fields: List of missing field names
        
    Returns:
        Formatted string
    """
    if not missing_fields:
        return ""
    
    field_names_sw = {
        "who": "Nani (watu waliohusika)",
        "what": "Nini (kile kilichotokea)",
        "where": "Wapi (mahali)",
        "when": "Lini (wakati/tarehe)"
    }
    
    formatted = []
    for field in missing_fields:
        formatted.append(field_names_sw.get(field, field))
    
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} na {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + f", na {formatted[-1]}"


def send_whatsapp_message(to: str, message: str) -> bool:
    """
    Send a WhatsApp message via Twilio.
    
    Args:
        to: Recipient phone number (format: whatsapp:+1234567890)
        message: Message text to send
        
    Returns:
        True if successful, False otherwise
    """
    if not twilio_client:
        logger = get_logger(__name__)
        logger.error("Twilio client not initialized. Check TWILIO_CONFIG.")
        return False
    
    if not TWILIO_CONFIG["whatsapp_number"]:
        logger = get_logger(__name__)
        logger.error("Twilio WhatsApp number not configured.")
        return False
    
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_CONFIG["whatsapp_number"],
            to=to
        )
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error sending WhatsApp message: {e}", exc_info=True)
        return False


def generate_response(processed_report) -> str:
    """
    Generate appropriate response message based on processed report.
    
    Args:
        processed_report: ProcessedReport object
        
    Returns:
        Response message string
    """
    validation = processed_report.validation
    escalation = processed_report.escalation
    
    # Build response message
    response_parts = []
    
    # Add receipt confirmation
    response_parts.append(
        get_response_message(
            "receipt_confirmation",
            report_id=processed_report.report_id
        )
    )
    
    # Add severity-specific acknowledgment
    severity = escalation.severity
    if severity == SeverityLevel.CRITICAL:
        response_parts.append(get_response_message("critical_acknowledgment"))
    elif severity == SeverityLevel.HIGH:
        response_parts.append(get_response_message("high_acknowledgment"))
    elif severity == SeverityLevel.MEDIUM:
        response_parts.append(get_response_message("medium_acknowledgment"))
    else:
        response_parts.append(get_response_message("low_acknowledgment"))
    
    # Add missing information request if incomplete
    if validation.status == ValidationStatus.INCOMPLETE and validation.missing_fields:
        missing_fields_str = format_missing_fields(validation.missing_fields)
        response_parts.append(
            get_response_message("missing_info", missing_fields=missing_fields_str)
        )
    else:
        # Add processing summary if complete
        response_parts.append(
            get_response_message(
                "processing_complete",
                severity=severity.value,
                completeness=validation.overall_completeness
            )
        )
    
    return "\n\n".join(response_parts)


@app.before_request
def log_request_info():
    """Log all incoming requests for debugging."""
    logger = get_logger(__name__)
    logger.info(
        f"Incoming {request.method} request to {request.path} "
        f"from {request.remote_addr}"
    )
    if request.method == "POST":
        logger.debug(f"POST data: {dict(request.form)}")
        logger.debug(f"POST values: {dict(request.values)}")


@app.route("/", methods=["GET"])
def root():
    """Root endpoint showing server status."""
    logger = get_logger(__name__)
    logger.info("Root endpoint accessed")
    
    status = {
        "status": "running",
        "service": "whatsapp_webhook",
        "pipeline_initialized": pipeline is not None,
        "twilio_configured": twilio_client is not None,
        "webhook_url": TWILIO_CONFIG["webhook_url"],
        "endpoints": {
            "/": "Server status",
            "/health": "Health check",
            "/webhook": "Twilio webhook (GET/POST)"
        }
    }
    
    return jsonify(status), 200


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    """
    Handle incoming WhatsApp messages from Twilio.
    Supports both GET (for verification) and POST (for messages).
    
    Returns:
        TwiML response
    """
    logger = get_logger(__name__)
    
    # Handle GET requests (Twilio webhook verification)
    if request.method == "GET":
        logger.info("Webhook verification request received")
        return Response(
            '{"status": "webhook_ready"}',
            mimetype="application/json",
            status=200
        )
    
    # Handle POST requests (actual messages)
    logger.info(f"POST request received. Form data: {dict(request.form)}")
    logger.info(f"Request values: {dict(request.values)}")
    
    # Check if pipeline is initialized
    if pipeline is None:
        logger.error("Pipeline not initialized")
        resp = MessagingResponse()
        resp.message(get_response_message("error_generic"))
        return Response(str(resp), mimetype="text/xml")
    
    try:
        # Get message data from Twilio webhook
        incoming_message = request.values.get("Body", "").strip()
        sender_number = request.values.get("From", "")
        
        logger.info(f"Received message from {sender_number}: {incoming_message[:100]}...")
        
        # Validate message
        if not incoming_message:
            logger.warning("Empty message received")
            response_message = get_response_message("error_empty")
            
            # Return TwiML response (Twilio will send this message)
            resp = MessagingResponse()
            resp.message(response_message)
            logger.info(f"Returning TwiML response: {response_message[:100]}...")
            return Response(str(resp), mimetype="text/xml")
        
        # Process message through pipeline
        try:
            processed_report = pipeline.process_message(
                message=incoming_message,
                source="whatsapp_twilio"
            )
            
            logger.info(
                f"Processed report {processed_report.report_id}: "
                f"Severity={processed_report.escalation.severity.value}, "
                f"Completeness={processed_report.validation.overall_completeness:.2f}"
            )
            
            # Generate response message
            response_message = generate_response(processed_report)
            logger.info(f"Generated response message: {response_message[:200]}...")
            
            # Return TwiML response (Twilio will send this message automatically)
            resp = MessagingResponse()
            resp.message(response_message)
            logger.info("Returning TwiML response to Twilio")
            return Response(str(resp), mimetype="text/xml")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            response_message = get_response_message("error_generic")
            
            resp = MessagingResponse()
            resp.message(response_message)
            return Response(str(resp), mimetype="text/xml")
    
    except Exception as e:
        logger.error(f"Error in webhook handler: {e}", exc_info=True)
        resp = MessagingResponse()
        resp.message(get_response_message("error_generic"))
        return Response(str(resp), mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return Response(
        '{"status": "healthy", "service": "whatsapp_webhook"}',
        mimetype="application/json",
        status=200
    )


@app.route("/test-webhook", methods=["POST"])
def test_webhook():
    """
    Test endpoint to simulate a Twilio webhook request.
    Useful for debugging without needing Twilio.
    """
    logger = get_logger(__name__)
    logger.info("Test webhook endpoint called")
    
    # Check if pipeline is initialized
    if pipeline is None:
        logger.error("Pipeline not initialized")
        return jsonify({"error": "Pipeline not initialized"}), 500
    
    try:
        # Get test message from request
        test_message = request.json.get("message", "") if request.is_json else request.form.get("Body", "")
        
        if not test_message:
            return jsonify({"error": "No message provided. Send JSON: {\"message\": \"your message\"}"}), 400
        
        logger.info(f"Test message received: {test_message[:100]}...")
        
        # Process message through pipeline
        processed_report = pipeline.process_message(
            message=test_message,
            source="test"
        )
        
        logger.info(
            f"Processed report {processed_report.report_id}: "
            f"Severity={processed_report.escalation.severity.value}, "
            f"Completeness={processed_report.validation.overall_completeness:.2f}"
        )
        
        # Generate response message
        response_message = generate_response(processed_report)
        
        return jsonify({
            "status": "success",
            "report_id": processed_report.report_id,
            "severity": processed_report.escalation.severity.value,
            "completeness": processed_report.validation.overall_completeness,
            "response_message": response_message
        }), 200
        
    except Exception as e:
        logger.error(f"Error in test webhook: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def main():
    """Initialize and run the webhook server."""
    global pipeline
    
    # Setup logging
    log_file_path = setup_logging_with_increment(
        log_level=logging.INFO,
        log_dir="logs",
        prefix="whatsapp_webhook"
    )
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_file_path}")
    logger.info("Starting WhatsApp Webhook Server")
    
    # Validate Twilio configuration
    if not TWILIO_CONFIG["account_sid"]:
        logger.error("TWILIO_ACCOUNT_SID not set. Please configure in .env file.")
        sys.exit(1)
    
    if not TWILIO_CONFIG["auth_token"]:
        logger.error("TWILIO_AUTH_TOKEN not set. Please configure in .env file.")
        sys.exit(1)
    
    if not TWILIO_CONFIG["whatsapp_number"]:
        logger.error("TWILIO_WHATSAPP_NUMBER not set. Please configure in .env file.")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = ThreatProcessingPipeline()
        logger.info("Threat processing pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)
    
    # Get port from config
    port = TWILIO_CONFIG["port"]
    webhook_url = TWILIO_CONFIG["webhook_url"]
    
    logger.info(f"Webhook server starting on port {port}")
    logger.info(f"Webhook URL from config: {webhook_url}")
    
    # Check if webhook URL is still placeholder
    if "your-ngrok-url" in webhook_url or "localhost" in webhook_url:
        logger.warning("=" * 60)
        logger.warning("WEBHOOK URL NOT CONFIGURED!")
        logger.warning("=" * 60)
        logger.warning("Current webhook URL: " + webhook_url)
        logger.warning("")
        logger.warning("To fix this:")
        logger.warning("1. Install and run ngrok: ngrok http 5000")
        logger.warning("2. Copy the HTTPS URL (e.g., https://abc123.ngrok.io)")
        logger.warning("3. Set WEBHOOK_URL=https://abc123.ngrok.io/webhook in your .env file")
        logger.warning("4. Configure this URL in Twilio Console:")
        logger.warning("   https://console.twilio.com/us1/develop/sms/sandbox/whatsapp-learn")
        logger.warning("5. Set HTTP method to POST")
        logger.warning("=" * 60)
    else:
        logger.info("Webhook URL configured: " + webhook_url)
        logger.info("Make sure this URL is set in Twilio Console")
        logger.info("Test endpoint available at: http://localhost:{}/test-webhook".format(port))
    
    # Add request logging
    logging.getLogger('werkzeug').setLevel(logging.INFO)
    
    # Run Flask app
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()

