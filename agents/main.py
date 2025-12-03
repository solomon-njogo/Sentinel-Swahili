"""
Main orchestration script for the multi-agent threat processing system.
Pipeline: WhatsApp input → Validator → Escalation → Storage
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data_models import (
    ThreatReport, ProcessedReport, ValidationResult, EscalationResult,
    ExtractedEntities, FieldCompleteness, ValidationStatus, SeverityLevel
)
from agents.validator_agent import ValidatorAgent
from agents.escalation_agent import EscalationAgent
from agents.storage import ReportStorage
from agents.whatsapp_simulator import WhatsAppSimulator
from agents.config import LANGUAGE_CONFIG
from src.utils.logger import setup_logging_with_increment, get_logger
from datetime import datetime
from typing import Optional, Dict
import logging


class ThreatProcessingPipeline:
    """Main pipeline orchestrating all agents"""
    
    def __init__(self, run_number: Optional[int] = None):
        """
        Initialize the pipeline.
        
        Args:
            run_number: Run number to tie reports to log file
        """
        self.logger = get_logger(__name__)
        self.validator = ValidatorAgent()
        self.escalator = EscalationAgent()
        self.storage = ReportStorage(run_number=run_number)
    
    def process_report(self, report: ThreatReport) -> ProcessedReport:
        """
        Process a threat report through the complete pipeline.
        
        Args:
            report: Threat report to process
            
        Returns:
            Processed report ready for storage
        """
        self.logger.info(f"Processing report {report.report_id}")
        
        try:
            # Step 1: Validate report
            self.logger.info("Step 1: Validating report completeness...")
            # All messages are in Swahili
            validation_result = self.validator.validate(
                text=report.raw_message,
                language="sw"  # Always Swahili
            )
            
            # Update report with validation result
            report.validation_result = validation_result
            
            # Step 2: Escalate based on severity
            self.logger.info("Step 2: Classifying severity and escalating...")
            escalation_result = self.escalator.escalate(
                text=report.raw_message,
                validation_result=validation_result
            )
            
            # Update report with escalation result
            report.escalation_result = escalation_result
            
            # Step 3: Create processed report
            processed_report = ProcessedReport(
                report_id=report.report_id,
                raw_message=report.raw_message,
                source=report.source,
                received_at=report.received_at,
                validation=validation_result,
                escalation=escalation_result,
                metadata=report.metadata
            )
            
            # Step 4: Store processed report
            self.logger.info("Step 3: Storing processed report...")
            filepath = self.storage.save_report(processed_report)
            
            self.logger.success(
                f"Report {report.report_id} processed successfully. "
                f"Severity: {escalation_result.severity.value}, "
                f"Completeness: {validation_result.overall_completeness:.2f}, "
                f"Saved to: {filepath}"
            )
            
            return processed_report
        
        except Exception as e:
            self.logger.error(f"Error processing report {report.report_id}: {e}", exc_info=True)
            raise
    
    def process_message(self, message: str, source: str = "whatsapp", report_id: Optional[str] = None) -> ProcessedReport:
        """
        Process a raw message through the pipeline.
        
        Args:
            message: Raw message text
            source: Source of the message (default: "whatsapp")
            report_id: Optional report ID. If None, generates one.
            
        Returns:
            Processed report
        """
        # Generate report ID if not provided
        if not report_id:
            report_id = self.storage.generate_report_id()
        
        # Create threat report
        report = ThreatReport(
            report_id=report_id,
            raw_message=message,
            source=source,
            metadata={"language": LANGUAGE_CONFIG["primary_language"]}
        )
        
        # Process through pipeline
        return self.process_report(report)
    
    def _dict_to_processed_report(self, report_dict: dict) -> ProcessedReport:
        """
        Reconstruct a ProcessedReport from a dictionary (loaded from storage).
        
        Args:
            report_dict: Dictionary representation of ProcessedReport
            
        Returns:
            ProcessedReport object
        """
        # Parse validation result
        validation_dict = report_dict.get("validation", {})
        entities_dict = validation_dict.get("entities", {})
        entities = ExtractedEntities(
            who=entities_dict.get("who", []),
            what=entities_dict.get("what", []),
            where=entities_dict.get("where", []),
            when=entities_dict.get("when", [])
        )
        
        field_scores = [
            FieldCompleteness(**field_dict)
            for field_dict in validation_dict.get("field_scores", [])
        ]
        
        validation = ValidationResult(
            status=ValidationStatus(validation_dict.get("status", "incomplete")),
            overall_completeness=validation_dict.get("overall_completeness", 0.0),
            entities=entities,
            field_scores=field_scores,
            missing_fields=validation_dict.get("missing_fields", []),
            prompts=validation_dict.get("prompts", []),
            timestamp=datetime.fromisoformat(validation_dict.get("timestamp", datetime.now().isoformat()))
        )
        
        # Parse escalation result
        escalation_dict = report_dict.get("escalation", {})
        escalation = EscalationResult(
            severity=SeverityLevel(escalation_dict.get("severity", "Low")),
            priority_score=escalation_dict.get("priority_score", 0.0),
            escalation_window_minutes=escalation_dict.get("escalation_window_minutes", 1440),
            urgency_keywords_found=escalation_dict.get("urgency_keywords_found", []),
            classification_confidence=escalation_dict.get("classification_confidence", 0.0),
            requires_immediate_alert=escalation_dict.get("requires_immediate_alert", False),
            timestamp=datetime.fromisoformat(escalation_dict.get("timestamp", datetime.now().isoformat()))
        )
        
        # Create ProcessedReport
        return ProcessedReport(
            report_id=report_dict.get("report_id"),
            raw_message=report_dict.get("raw_message", ""),
            source=report_dict.get("source", "whatsapp"),
            received_at=datetime.fromisoformat(report_dict.get("received_at", datetime.now().isoformat())),
            validation=validation,
            escalation=escalation,
            status=report_dict.get("status", "processed"),
            processed_at=datetime.fromisoformat(report_dict.get("processed_at", datetime.now().isoformat())),
            metadata=report_dict.get("metadata", {})
        )
    
    def update_report(
        self, 
        existing_report_id: str, 
        new_message: str,
        collected_data: Optional[Dict[str, Optional[str]]] = None
    ) -> ProcessedReport:
        """
        Update an existing report with additional information.
        Merges new message with existing message and re-processes.
        
        Args:
            existing_report_id: ID of the existing report to update
            new_message: New message text to merge with existing report
            collected_data: Optional structured collected data (where, what, who, when)
            
        Returns:
            Updated ProcessedReport
        """
        self.logger.info(f"Updating report {existing_report_id} with new message")
        
        # Load existing report from storage
        report_dict = self.storage.load_report(existing_report_id)
        if not report_dict:
            raise ValueError(f"Report {existing_report_id} not found in storage")
        
        # Reconstruct ProcessedReport from dict
        existing_report = self._dict_to_processed_report(report_dict)
        
        # Merge messages: combine existing message with new message
        merged_message = f"{existing_report.raw_message}\n\n{new_message}"
        self.logger.debug(f"Merged message length: {len(merged_message)} characters")
        
        # Create a new ThreatReport with merged message
        # Keep original report_id, received_at, and source
        updated_report = ThreatReport(
            report_id=existing_report.report_id,
            raw_message=merged_message,
            source=existing_report.source,
            received_at=existing_report.received_at,
            metadata=existing_report.metadata
        )
        
        # Re-validate with merged message
        self.logger.info("Re-validating merged report...")
        validation_result = self.validator.validate(
            text=merged_message,
            language="sw"  # Always Swahili
        )
        updated_report.validation_result = validation_result
        
        # Re-escalate with merged message
        self.logger.info("Re-escalating merged report...")
        escalation_result = self.escalator.escalate(
            text=merged_message,
            validation_result=validation_result
        )
        updated_report.escalation_result = escalation_result
        
        # Create updated ProcessedReport with structured data if provided
        updated_processed_report = ProcessedReport(
            report_id=existing_report.report_id,
            raw_message=merged_message,
            source=existing_report.source,
            received_at=existing_report.received_at,
            validation=validation_result,
            escalation=escalation_result,
            status="updated",
            processed_at=datetime.now(),
            metadata=existing_report.metadata,
            collected_where=collected_data.get("where") if collected_data and collected_data.get("where") else report_dict.get('collected_where') if report_dict else None,
            collected_what=collected_data.get("what") if collected_data and collected_data.get("what") else report_dict.get('collected_what') if report_dict else None,
            collected_who=collected_data.get("who") if collected_data and collected_data.get("who") else report_dict.get('collected_who') if report_dict else None,
            collected_when=collected_data.get("when") if collected_data and collected_data.get("when") else report_dict.get('collected_when') if report_dict else None
        )
        
        # Save updated report (overwrites existing file)
        filepath = self.storage.save_report(updated_processed_report)
        
        self.logger.success(
            f"Report {existing_report_id} updated successfully. "
            f"Severity: {escalation_result.severity.value}, "
            f"Completeness: {validation_result.overall_completeness:.2f}, "
            f"Saved to: {filepath}"
        )
        
        return updated_processed_report
    
    def process_simulated_messages(self, count: int = 5) -> list:
        """
        Process simulated WhatsApp messages.
        Ensures at least one message from each severity category (critical, high, medium, low).
        
        Args:
            count: Number of messages to process (minimum 4 to cover all categories)
            
        Returns:
            List of processed reports
        """
        simulator = WhatsAppSimulator()
        processed_reports = []
        
        # Ensure count is at least 4 to cover all categories
        if count < 4:
            count = 4
            self.logger.info(f"Adjusted count to {count} to ensure all severity categories are tested")
        
        self.logger.info(f"Processing {count} simulated messages (ensuring all severity categories)...")
        
        # Generate messages ensuring all categories are represented
        message_dicts = simulator.generate_messages_with_all_categories(count=count)
        
        for i, message_dict in enumerate(message_dicts, 1):
            # Create threat report using pipeline's storage (with run_number)
            report = simulator.create_threat_report(message_dict, storage=self.storage)
            
            # Process through pipeline
            try:
                processed = self.process_report(report)
                processed_reports.append(processed)
                self.logger.debug(
                    f"Processed message {i}/{count}: "
                    f"Severity={message_dict.get('severity')}, "
                    f"Completeness={message_dict.get('completeness')}"
                )
            except Exception as e:
                self.logger.error(f"Error processing simulated message {i}: {e}")
                continue
        
        # Verify all categories were processed
        severities_processed = set(r.escalation.severity.value.lower() for r in processed_reports)
        expected_severities = {"critical", "high", "medium", "low"}
        missing_severities = expected_severities - severities_processed
        
        if missing_severities:
            self.logger.warning(f"Missing severity categories in processed reports: {missing_severities}")
        else:
            self.logger.info(f"All severity categories processed: {severities_processed}")
        
        self.logger.info(f"Successfully processed {len(processed_reports)}/{count} messages")
        return processed_reports


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Threat Processing System"
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "message", "interactive"],
        default="simulate",
        help="Processing mode (default: simulate)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of simulated messages to process (default: 5)"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Raw message text to process (for message mode)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    log_file_path = setup_logging_with_increment(
        log_level=log_level,
        log_dir="logs",
        prefix="agents"
    )
    
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_file_path}")
    logger.info("Starting Multi-Agent Threat Processing System")
    
    # Extract run number from log filename (e.g., agents_001.log -> 1)
    run_number = None
    try:
        log_filename = Path(log_file_path).name
        # Extract number from filename like "agents_001.log"
        if "_" in log_filename:
            number_str = log_filename.split("_")[1].split(".")[0]
            run_number = int(number_str)
            logger.info(f"Run number: {run_number}")
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not extract run number from log filename: {e}")
    
    # Initialize pipeline with run number
    pipeline = ThreatProcessingPipeline(run_number=run_number)
    
    try:
        if args.mode == "simulate":
            # Process simulated messages
            logger.info(f"Processing {args.count} simulated messages...")
            processed_reports = pipeline.process_simulated_messages(count=args.count)
            
            # Print summary
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSING SUMMARY")
            logger.info("=" * 80)
            for report in processed_reports:
                logger.info(
                    f"Report {report.report_id}: "
                    f"Severity={report.escalation.severity.value}, "
                    f"Completeness={report.validation.overall_completeness:.2f}, "
                    f"Status={report.validation.status.value}"
                )
            logger.info("=" * 80)
        
        elif args.mode == "message":
            # Process single message
            if not args.message:
                logger.error("--message argument required for message mode")
                return
            
            logger.info("Processing single message...")
            processed = pipeline.process_message(args.message)
            
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSING RESULT")
            logger.info("=" * 80)
            logger.info(f"Report ID: {processed.report_id}")
            logger.info(f"Severity: {processed.escalation.severity.value}")
            logger.info(f"Priority: {processed.escalation.priority_score:.2f}")
            logger.info(f"Completeness: {processed.validation.overall_completeness:.2f}")
            logger.info(f"Status: {processed.validation.status.value}")
            logger.info(f"Escalation Window: {processed.escalation.escalation_window_minutes} minutes")
            if processed.validation.missing_fields:
                logger.info(f"Missing Fields: {', '.join(processed.validation.missing_fields)}")
            logger.info("=" * 80)
        
        elif args.mode == "interactive":
            # Interactive mode
            logger.info("Interactive mode - Enter messages (type 'quit' to exit)")
            while True:
                try:
                    message = input("\nEnter threat report: ").strip()
                    if message.lower() in ['quit', 'exit', 'q']:
                        break
                    if not message:
                        continue
                    
                    processed = pipeline.process_message(message)
                    print(f"\n✓ Processed: {processed.report_id}")
                    print(f"  Severity: {processed.escalation.severity.value}")
                    print(f"  Completeness: {processed.validation.overall_completeness:.2f}")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")
        
        logger.success("Pipeline execution completed successfully")
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

