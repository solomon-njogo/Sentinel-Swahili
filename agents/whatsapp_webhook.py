"""
WhatsApp webhook server for receiving threat reports via Twilio.
Processes messages through the multi-agent pipeline and sends automated responses.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict
from flask import Flask, request, Response, jsonify
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.main import ThreatProcessingPipeline
from agents.conversation_manager import ConversationManager
from agents.conversation_flow_manager import ConversationFlowManager, ConversationState
from agents.config import TWILIO_CONFIG, FIELD_PROMPTS, LANGUAGE_CONFIG, FLOW_MESSAGES, BUTTON_CONFIG
from agents.data_models import ValidationStatus, SeverityLevel
from src.utils.logger import setup_logging_with_increment, get_logger
from datetime import datetime
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

# Initialize conversation manager
conversation_manager = None

# Initialize conversation flow manager
flow_manager = None

# Response message templates in Swahili with structured formatting
RESPONSE_MESSAGES = {
    "header": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📋 TAARIFA YA TISHIO\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ),
    "receipt_confirmation": (
        "✅ *Taarifa Imepokelewa*\n\n"
        "Asante! Taarifa yako imepokelewa na tunaichambua sasa hivi.\n\n"
        "🆔 *Nambari ya Taarifa:* {report_id}"
    ),
    "update_confirmation": (
        "🔄 *Taarifa Imesasishwa*\n\n"
        "Asante! Taarifa yako imesasishwa na tunaichambua tena.\n\n"
        "🆔 *Nambari ya Taarifa:* {report_id}"
    ),
    "processing_complete": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 *Matokeo ya Uchambuzi*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🎯 *Kiwango cha Hatari:* {severity}\n"
        "📈 *Kamili:* {completeness:.0%}"
    ),
    "missing_info": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ *Maelezo Yanayohitajika*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Asante kwa taarifa yako. Ili tuweze kuchambua taarifa yako kwa usahihi, tafadhali ongeza maelezo yafuatayo:\n\n"
        "{missing_fields}\n\n"
        "Tuma ujumbe mwingine na maelezo haya ili tuweze kukusaidia haraka zaidi."
    ),
    "critical_acknowledgment": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🚨 *DHARURA IMETAMBULIWA*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Taarifa yako imepokelewa na inachambuliwa haraka.\n"
        "Tutachukua hatua za haraka zaidi."
    ),
    "high_acknowledgment": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ *HATARI KUBWA*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Hatari kubwa imetambuliwa.\n"
        "Taarifa yako inachambuliwa na tutachukua hatua ndani ya dakika 30."
    ),
    "medium_acknowledgment": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📌 *HATARI YA KATI*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Taarifa yako imepokelewa.\n"
        "Itachambuliwa na kujumuishwa katika ripoti ya kila siku."
    ),
    "low_acknowledgment": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ℹ️ *HATARI YA CHINI*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Asante kwa taarifa yako.\n"
        "Itachambuliwa na kujumuishwa katika ripoti ya kila wiki."
    ),
    "error_generic": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "❌ *TATIZO LIMETOKEA*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Samahani, kumekuwa na tatizo katika kuchakata taarifa yako.\n"
        "Tafadhali jaribu tena baadaye au wasiliana nasi moja kwa moja."
    ),
    "error_empty": (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ *TAARIFA HAIJAPOKELEWA*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Samahani, hujatoa taarifa yoyote.\n"
        "Tafadhali toa maelezo ya tishio au tukio."
    ),
    "footer": (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Asante kwa kutumia huduma yetu"
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


def format_severity_swahili(severity: SeverityLevel) -> str:
    """
    Convert severity level to Swahili for display.
    
    Args:
        severity: SeverityLevel enum value
        
    Returns:
        Swahili translation of severity level
    """
    severity_map = {
        SeverityLevel.CRITICAL: "Dharura",
        SeverityLevel.HIGH: "Kubwa",
        SeverityLevel.MEDIUM: "Kati",
        SeverityLevel.LOW: "Chini"
    }
    return severity_map.get(severity, severity.value)


def format_missing_fields(missing_fields: list) -> str:
    """
    Format missing fields list into a readable Swahili string with bullet points.
    Uses prompts from FIELD_PROMPTS config to provide clear instructions.
    
    Args:
        missing_fields: List of missing field names
        
    Returns:
        Formatted string with bullet points and clear instructions
    """
    if not missing_fields:
        return ""
    
    formatted = []
    for field in missing_fields:
        # Use the prompt from FIELD_PROMPTS config for clearer instructions
        prompt = FIELD_PROMPTS.get(field, {}).get("sw", f"Tafadhali toa maelezo kuhusu {field}")
        formatted.append(f"• {prompt}")
    
    return "\n".join(formatted)


def send_whatsapp_message(to: str, message: str, with_skip_button: bool = False) -> bool:
    """
    Send a WhatsApp message via Twilio.
    
    Args:
        to: Recipient phone number (format: whatsapp:+1234567890)
        message: Message text to send
        with_skip_button: Whether to add a SKIP button
        
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


def create_message_with_button(message: str, button_label: str = "SKIP") -> str:
    """
    Create a message with WhatsApp interactive button.
    Note: Twilio WhatsApp buttons require specific formatting.
    For now, we'll add the button option as text in the message.
    
    Args:
        message: Main message text
        button_label: Button label (default: "SKIP")
        
    Returns:
        Message with button instruction
    """
    return f"{message}\n\n(Jibu '{button_label}' ikiwa hujui au unataka kuruka)"


def _generate_completion_response(collected_data: Dict[str, Optional[str]], report_id: str) -> str:
    """
    Generate completion response when all fields are collected.
    
    Args:
        collected_data: Dictionary with collected where, what, who, when
        report_id: Report ID
        
    Returns:
        Completion message
    """
    completion_msg = FLOW_MESSAGES.get("completion", "Asante sana!")
    return f"{completion_msg}\n\n🆔 Nambari ya Taarifa: {report_id}"


def _create_final_report(
    collected_data: Dict[str, Optional[str]],
    report_id: str,
    sender_number: str
) -> None:
    """
    Create final comprehensive report from collected data.
    
    Args:
        collected_data: Dictionary with collected where, what, who, when
        report_id: Report ID
        sender_number: Sender phone number
    """
    logger = get_logger(__name__)
    try:
        # Build comprehensive message from collected data
        parts = []
        if collected_data.get("what"):
            parts.append(f"Nini: {collected_data['what']}")
        if collected_data.get("where"):
            parts.append(f"Wapi: {collected_data['where']}")
        if collected_data.get("who"):
            parts.append(f"Nani: {collected_data['who']}")
        if collected_data.get("when"):
            parts.append(f"Lini: {collected_data['when']}")
        
        comprehensive_message = "\n".join(parts)
        
        # Load existing report if it exists
        existing_report = pipeline.storage.load_report(report_id)
        if existing_report:
            # Update with comprehensive data
            processed_report = pipeline.update_report(
                existing_report_id=report_id,
                new_message=comprehensive_message
            )
        else:
            # Create new report
            processed_report = pipeline.process_message(
                message=comprehensive_message,
                source="whatsapp_twilio",
                report_id=report_id
            )
        
        # Update report with structured collected data
        processed_report.collected_where = collected_data.get("where")
        processed_report.collected_what = collected_data.get("what")
        processed_report.collected_who = collected_data.get("who")
        processed_report.collected_when = collected_data.get("when")
        processed_report.conversation_flow = {
            "completed_via_flow": True,
            "collected_at": datetime.now().isoformat()
        }
        
        # Save updated report
        pipeline.storage.save_report(processed_report)
        logger.info(f"Created final comprehensive report {report_id}")
        
    except Exception as e:
        logger.error(f"Error creating final report: {e}", exc_info=True)


def generate_response(processed_report, is_update: bool = False) -> str:
    """
    Generate appropriate response message based on processed report.
    Creates a structured, visually appealing message for WhatsApp.
    
    Args:
        processed_report: ProcessedReport object
        is_update: Whether this is an update to an existing report
        
    Returns:
        Formatted response message string
    """
    validation = processed_report.validation
    escalation = processed_report.escalation
    severity = escalation.severity
    
    # Build response message with structure
    response_parts = []
    
    # Add header
    response_parts.append(get_response_message("header"))
    response_parts.append("")  # Empty line for spacing
    
    # Add receipt confirmation or update confirmation
    if is_update:
        response_parts.append(
            get_response_message(
                "update_confirmation",
                report_id=processed_report.report_id
            )
        )
    else:
        response_parts.append(
            get_response_message(
                "receipt_confirmation",
                report_id=processed_report.report_id
            )
        )
    response_parts.append("")  # Empty line for spacing
    
    # Add severity-specific acknowledgment
    if severity == SeverityLevel.CRITICAL:
        response_parts.append(get_response_message("critical_acknowledgment"))
    elif severity == SeverityLevel.HIGH:
        response_parts.append(get_response_message("high_acknowledgment"))
    elif severity == SeverityLevel.MEDIUM:
        response_parts.append(get_response_message("medium_acknowledgment"))
    else:
        response_parts.append(get_response_message("low_acknowledgment"))
    
    response_parts.append("")  # Empty line for spacing
    
    # Add missing information request if there are missing fields
    # Show missing info for both INCOMPLETE and FAILED statuses if fields are missing
    if validation.missing_fields:
        missing_fields_str = format_missing_fields(validation.missing_fields)
        response_parts.append(
            get_response_message("missing_info", missing_fields=missing_fields_str)
        )
        response_parts.append("")  # Empty line for spacing
        
        # Also show current completeness status even when incomplete
        severity_sw = format_severity_swahili(severity)
        response_parts.append(
            get_response_message(
                "processing_complete",
                severity=severity_sw,
                completeness=validation.overall_completeness
            )
        )
    else:
        # Add processing summary if complete
        severity_sw = format_severity_swahili(severity)
        response_parts.append(
            get_response_message(
                "processing_complete",
                severity=severity_sw,
                completeness=validation.overall_completeness
            )
        )
    
    # Add footer
    response_parts.append(get_response_message("footer"))
    
    return "\n".join(response_parts)


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
    global flow_manager
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
    
    # Check if conversation manager is initialized
    if conversation_manager is None:
        logger.error("Conversation manager not initialized")
        resp = MessagingResponse()
        resp.message(get_response_message("error_generic"))
        return Response(str(resp), mimetype="text/xml")
    
    # Check if flow manager is initialized
    if flow_manager is None:
        logger.error("Flow manager not initialized")
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
        
        # Check for active session
        active_session = conversation_manager.get_active_session(sender_number)
        
        # Handle conversation flow
        try:
            if active_session:
                # Existing session - continue conversation flow
                flow_state = active_session.get("flow_state", {})
                collected_data = active_session.get("collected_data", {
                    "where": None,
                    "what": None,
                    "who": None,
                    "when": None
                })
                current_state_str = flow_state.get("state", ConversationState.INITIAL.value)
                current_state = ConversationState(current_state_str)
                
                logger.info(f"Continuing flow for {sender_number}, state: {current_state.value}")
                
                # Process response
                flow_result = flow_manager.process_response(
                    message=incoming_message,
                    current_state=current_state,
                    collected_data=collected_data
                )
                
                # Update session with new flow state
                flow_state.update({
                    "state": flow_result["state"],
                    "missing_fields": flow_result["missing_fields"],
                    "next_question_field": flow_result["next_question_field"]
                })
                
                conversation_manager.update_session(
                    phone_number=sender_number,
                    flow_state=flow_state,
                    collected_data=flow_result["collected_data"]
                )
                
                # Generate response based on flow state
                if flow_result["is_complete"]:
                    # All fields collected - create final report
                    response_message = _generate_completion_response(
                        flow_result["collected_data"],
                        active_session.get("report_id")
                    )
                    # Close session
                    conversation_manager.close_session(sender_number)
                elif flow_result.get("needs_followup"):
                    # Vague answer - ask follow-up
                    next_field = flow_result["next_question_field"]
                    response_message = flow_manager.get_question_message(next_field, is_followup=True)
                    response_message = create_message_with_button(response_message)
                else:
                    # Move to next question
                    next_field = flow_result["next_question_field"]
                    if next_field:
                        # Add reassurance if not first question
                        reassurance = flow_manager.get_reassurance_message()
                        question = flow_manager.get_question_message(next_field)
                        response_message = f"{reassurance}\n\n{question}"
                        response_message = create_message_with_button(response_message)
                    else:
                        response_message = FLOW_MESSAGES.get("completion", "Asante!")
                
                        # Create final report if complete
                if flow_result["is_complete"]:
                    _create_final_report(
                        flow_result["collected_data"],
                        active_session.get("report_id"),
                        sender_number
                    )
                
            else:
                # New session - initialize flow
                logger.info(f"New session for {sender_number}, initializing flow")
                
                # Initialize conversation flow
                flow_state = flow_manager.initialize_flow(incoming_message)
                
                # Create initial report
                report_id = pipeline.storage.generate_report_id()
                processed_report = pipeline.process_message(
                    message=incoming_message,
                    source="whatsapp_twilio",
                    report_id=report_id
                )
                
                # Create session with flow state
                conversation_manager.create_session(
                    phone_number=sender_number,
                    report_id=report_id,
                    flow_state=flow_state,
                    collected_data=flow_state["collected_data"]
                )
                
                # Generate response
                if flow_state["is_urgent"]:
                    # Urgent - acknowledge and still ask questions but faster
                    urgency_msg = get_response_message("critical_acknowledgment") if flow_state["urgency_severity"] == "Critical" else get_response_message("high_acknowledgment")
                    greeting = flow_manager.get_initial_greeting()
                    response_message = f"{urgency_msg}\n\n{greeting}"
                else:
                    response_message = flow_manager.get_initial_greeting()
                
                # Ask first question if needed
                next_field = flow_state.get("next_question_field")
                if next_field:
                    question = flow_manager.get_question_message(next_field)
                    response_message = f"{response_message}\n\n{question}"
                    response_message = create_message_with_button(response_message)
                else:
                    # All info provided in first message
                    response_message = f"{response_message}\n\n{FLOW_MESSAGES.get('completion', 'Asante!')}"
                    # Create final report immediately
                    _create_final_report(
                        flow_state["collected_data"],
                        report_id,
                        sender_number
                    )
                    conversation_manager.close_session(sender_number)
            
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
    global pipeline, conversation_manager, flow_manager
    
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
    
    # Initialize conversation manager
    try:
        conversation_manager = ConversationManager()
        logger.info("Conversation manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize conversation manager: {e}", exc_info=True)
        sys.exit(1)
    
    # Initialize conversation flow manager
    try:
        flow_manager = ConversationFlowManager()
        logger.info("Conversation flow manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize flow manager: {e}", exc_info=True)
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

