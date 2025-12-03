"""
WhatsApp message simulator for testing the agent system.
Generates sample Swahili threat reports with various scenarios.
All messages are in Swahili only.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random

from agents.data_models import ThreatReport


class WhatsAppSimulator:
    """Simulate WhatsApp messages for testing - Swahili only"""
    
    def __init__(self):
        """Initialize simulator"""
        self.sample_messages = self._generate_sample_messages()
    
    def _generate_sample_messages(self) -> List[Dict[str, str]]:
        """Generate sample Swahili threat report messages"""
        return [
            # Critical - Complete report
            {
                "message": "Dharura! Kuna shambulio la risasi katika Soko la Mwenge, Dar es Salaam. "
                          "Watu wengi wamejeruhiwa. Tafadhali fika haraka. "
                          "Tarehe: leo saa 3:00 jioni. "
                          "Washambuliaji: Watu wawili wamevaa nguo nyeusi.",
                "severity": "critical",
                "completeness": "complete",
                "language": "sw"
            },
            # Critical - Incomplete (missing when)
            {
                "message": "Bomu limepatikana katika eneo la Kariakoo, Dar es Salaam. "
                          "Polisi wamefika na kuzuia eneo. "
                          "Washambuliaji: Haijulikani. "
                          "Tafadhali fika haraka!",
                "severity": "critical",
                "completeness": "incomplete",
                "language": "sw"
            },
            # High - Complete report
            {
                "message": "Kuna tishio la silaha katika Shule ya Sekondari ya Mlimani, Nairobi. "
                          "Mwalimu John Kamau ameona mwanafunzi akiwa na kisu. "
                          "Tarehe: kesho saa 8:00 asubuhi. "
                          "Tafadhali chukua hatua.",
                "severity": "high",
                "completeness": "complete",
                "language": "sw"
            },
            # High - Incomplete (missing who)
            {
                "message": "Kuna silaha zilizopatikana karibu na Stesheni ya Reli, Mombasa. "
                          "Polisi wamefika. Tarehe: leo saa 2:00 jioni. "
                          "Hatari kubwa!",
                "severity": "high",
                "completeness": "incomplete",
                "language": "sw"
            },
            # Medium - Complete report
            {
                "message": "Kuna shughuli za kushuku katika eneo la Kimara, Dar es Salaam. "
                          "Watu wameonekana wakikusanya taarifa kuhusu eneo. "
                          "Jina: Ahmed Hassan. Tarehe: jana saa 4:00 jioni.",
                "severity": "medium",
                "completeness": "complete",
                "language": "sw"
            },
            # Medium - Incomplete (missing where)
            {
                "message": "Kuna shughuli isiyo ya kawaida. "
                          "Watu wameonekana wakifanya uchunguzi wa eneo. "
                          "Jina: Juma Mwangi. Tarehe: leo saa 10:00 asubuhi.",
                "severity": "medium",
                "completeness": "incomplete",
                "language": "sw"
            },
            # Low - Complete report
            {
                "message": "Kuna taarifa za shughuli za kushuku katika eneo la Kisumu. "
                          "Hakuna hatari ya haraka. "
                          "Watu: Haijulikani. "
                          "Tarehe: wiki ijayo Jumatatu saa 9:00 asubuhi.",
                "severity": "low",
                "completeness": "complete",
                "language": "sw"
            },
            # Low - Incomplete (missing what)
            {
                "message": "Kuna jambo la kusumbua katika eneo la Arusha. "
                          "Watu: Mwanafunzi aliyeitwa Peter. "
                          "Tarehe: leo saa 1:00 jioni.",
                "severity": "low",
                "completeness": "incomplete",
                "language": "sw"
            },
            # Critical - Complete (different scenario)
            {
                "message": "Hatari kubwa! Kuna watu wamevaa silaha katika Hospitali ya Muhimbili, Dar es Salaam. "
                          "Watu: Watu watatu wamevaa nguo nyeusi. "
                          "Tarehe: sasa hivi. "
                          "Tafadhali fika haraka sana!",
                "severity": "critical",
                "completeness": "complete",
                "language": "sw"
            },
            # High - Complete (weapon threat)
            {
                "message": "Kuna tishio la bunduki katika Chuo Kikuu cha Nairobi, Nairobi. "
                          "Mtu: Mwanafunzi aliyeitwa James Ochieng. "
                          "Tarehe: kesho saa 2:00 mchana. "
                          "Tafadhali chukua hatua haraka.",
                "severity": "high",
                "completeness": "complete",
                "language": "sw"
            },
        ]
    
    def generate_message(self, index: Optional[int] = None) -> Dict[str, str]:
        """
        Generate a sample Swahili message.
        
        Args:
            index: Specific message index. If None, returns random message.
            
        Returns:
            Message dictionary with text and metadata
        """
        if index is not None and 0 <= index < len(self.sample_messages):
            return self.sample_messages[index]
        else:
            return random.choice(self.sample_messages)
    
    def generate_messages_with_all_categories(self, count: int = 4) -> List[Dict[str, str]]:
        """
        Generate messages ensuring at least one from each severity category.
        
        Args:
            count: Total number of messages to generate (minimum 4 to cover all categories)
            
        Returns:
            List of message dictionaries with at least one from each category
        """
        # Ensure minimum count covers all categories
        if count < 4:
            count = 4
        
        # Group messages by severity
        messages_by_severity = {
            "critical": self.generate_by_severity("critical"),
            "high": self.generate_by_severity("high"),
            "medium": self.generate_by_severity("medium"),
            "low": self.generate_by_severity("low")
        }
        
        selected_messages = []
        
        # Step 1: Select at least one from each category
        for severity in ["critical", "high", "medium", "low"]:
            if messages_by_severity[severity]:
                selected_messages.append(random.choice(messages_by_severity[severity]))
        
        # Step 2: Fill remaining slots with random messages from any category
        remaining = count - len(selected_messages)
        if remaining > 0:
            all_available = [msg for msgs in messages_by_severity.values() for msg in msgs]
            additional = random.sample(all_available, min(remaining, len(all_available)))
            selected_messages.extend(additional)
        
        # Shuffle to randomize order
        random.shuffle(selected_messages)
        
        return selected_messages[:count]
    
    def generate_all_messages(self) -> List[Dict[str, str]]:
        """
        Get all sample Swahili messages.
        
        Returns:
            List of all message dictionaries
        """
        return self.sample_messages
    
    def generate_by_severity(self, severity: str) -> List[Dict[str, str]]:
        """
        Get messages filtered by severity.
        
        Args:
            severity: Severity level ("critical", "high", "medium", "low")
            
        Returns:
            List of messages with specified severity
        """
        return [
            msg for msg in self.sample_messages
            if msg.get("severity") == severity.lower()
        ]
    
    def generate_by_completeness(self, completeness: str) -> List[Dict[str, str]]:
        """
        Get messages filtered by completeness.
        
        Args:
            completeness: Completeness level ("complete", "incomplete")
            
        Returns:
            List of messages with specified completeness
        """
        return [
            msg for msg in self.sample_messages
            if msg.get("completeness") == completeness.lower()
        ]
    
    def create_threat_report(
        self,
        message: Dict[str, str],
        report_id: Optional[str] = None,
        storage: Optional[object] = None
    ) -> ThreatReport:
        """
        Create a ThreatReport from a simulated message.
        
        Args:
            message: Message dictionary
            report_id: Optional report ID. If None, generates one.
            storage: Optional ReportStorage instance to use for ID generation
            
        Returns:
            ThreatReport object
        """
        if storage is None:
            from agents.storage import ReportStorage
            storage = ReportStorage()
        
        report_id = report_id or storage.generate_report_id()
        
        # Simulate received time (random within last 24 hours)
        received_at = datetime.now() - timedelta(
            hours=random.randint(0, 24),
            minutes=random.randint(0, 60)
        )
        
        report = ThreatReport(
            report_id=report_id,
            raw_message=message["message"],
            source="whatsapp_simulator",
            received_at=received_at,
            metadata={
                "simulated_severity": message.get("severity"),
                "simulated_completeness": message.get("completeness"),
                "language": "sw"  # Always Swahili
            }
        )
        
        return report
