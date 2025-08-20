import os
import pickle
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gmail_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

@dataclass
class EmailData:
    """Data class to store email information"""
    id: str
    subject: str
    sender: str
    date: datetime
    body: str
    thread_id: str
    labels: List[str]

@dataclass
class NegotiationAnalysis:
    """Data class to store negotiation analysis results"""
    is_negotiation: bool
    is_followup: bool
    negotiation_type: str
    confidence_score: float
    keywords_found: List[str]
    reason: str

class GmailNegotiationAgent:
    """
    Gmail agent for identifying supplier negotiation emails
    """
    
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.pickle'):
        """
        Initialize the Gmail agent
        
        Args:
            credentials_file: Path to Gmail API credentials JSON file
            token_file: Path to store OAuth2 token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.last_run_file = 'last_run.txt'
        
        # Negotiation keywords and patterns
        self.negotiation_keywords = {
            'price': ['price', 'cost', 'quote', 'quotation', 'pricing', 'rate', 'discount', 'budget'],
            'terms': ['terms', 'conditions', 'contract', 'agreement', 'payment terms', 'delivery'],
            'negotiation': ['negotiate', 'negotiation', 'counter offer', 'proposal', 'offer'],
            'supplier': ['supplier', 'vendor', 'procurement', 'purchase', 'sourcing'],
            'followup': ['follow up', 'followup', 'following up', 'checking in', 'status update', 'reminder']
        }
        
        # Patterns that indicate negotiation emails
        self.negotiation_patterns = [
            r'\b(?:price|cost|quote)\s+(?:negotiation|discussion|proposal)\b',
            r'\b(?:counter|new)\s+(?:offer|proposal)\b',
            r'\b(?:payment|delivery)\s+terms\b',
            r'\b(?:discount|reduction)\s+(?:request|proposal)\b',
            r'\bRFQ\b|RFP\b',  # Request for Quote/Proposal
            r'\b(?:procurement|sourcing)\s+(?:opportunity|request)\b'
        ]
        
        logger.info("Gmail Negotiation Agent initialized")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Gmail API using OAuth2
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            creds = None
            
            # Load existing token if available
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
                    logger.info("Loaded existing authentication token")
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    logger.info("Refreshed expired authentication token")
                else:
                    if not os.path.exists(self.credentials_file):
                        logger.error(f"Credentials file {self.credentials_file} not found")
                        return False
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    logger.info("Obtained new authentication credentials")
                
                # Save credentials for next run
                with open(self.token_file, 'wb') as token:
                    pickle.dump(creds, token)
                    logger.info("Saved authentication token")
            
            # Build Gmail service
            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail service authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def get_last_run_time(self) -> datetime:
        """
        Get the timestamp of the last successful run
        
        Returns:
            datetime: Last run timestamp, defaults to 24 hours ago if not found
        """
        try:
            if os.path.exists(self.last_run_file):
                with open(self.last_run_file, 'r') as f:
                    timestamp = f.read().strip()
                    last_run = datetime.fromisoformat(timestamp)
                    logger.info(f"Last run time: {last_run}")
                    return last_run
            else:
                # Default to 24 hours ago if no last run file
                default_time = datetime.now() - timedelta(hours=24)
                logger.info(f"No last run file found, defaulting to: {default_time}")
                return default_time
        except Exception as e:
            logger.error(f"Error reading last run time: {str(e)}")
            return datetime.now() - timedelta(hours=24)
    
    def update_last_run_time(self) -> None:
        """
        Update the last run timestamp to current time
        """
        try:
            current_time = datetime.now()
            with open(self.last_run_file, 'w') as f:
                f.write(current_time.isoformat())
            logger.info(f"Updated last run time to: {current_time}")
        except Exception as e:
            logger.error(f"Error updating last run time: {str(e)}")
    
    def fetch_emails_since_last_run(self) -> List[EmailData]:
        """
        Fetch emails from Gmail inbox since last run
        
        Returns:
            List[EmailData]: List of email data objects
        """
        if not self.service:
            logger.error("Gmail service not authenticated")
            return []
        
        try:
            last_run = self.get_last_run_time()
            
            # Convert datetime to Gmail query format
            query_date = last_run.strftime('%Y/%m/%d')
            query = f'in:inbox after:{query_date}'
            
            logger.info(f"Searching for emails with query: {query}")
            
            # Search for emails
            result = self.service.users().messages().list(
                userId='me', 
                q=query,
                maxResults=100  # Adjust as needed
            ).execute()
            
            messages = result.get('messages', [])
            logger.info(f"Found {len(messages)} emails since last run")
            
            emails = []
            for message in messages:
                email_data = self._get_email_details(message['id'])
                if email_data:
                    emails.append(email_data)
            
            logger.info(f"Successfully processed {len(emails)} emails")
            return emails
            
        except HttpError as error:
            logger.error(f"Gmail API error: {error}")
            return []
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            return []
    
    def _get_email_details(self, message_id: str) -> Optional[EmailData]:
        """
        Get detailed information for a specific email
        
        Args:
            message_id: Gmail message ID
            
        Returns:
            EmailData: Email data object or None if error
        """
        try:
            message = self.service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = {h['name']: h['value'] for h in message['payload']['headers']}
            
            # Extract email body
            body = self._extract_email_body(message['payload'])
            
            # Create EmailData object
            email_data = EmailData(
                id=message_id,
                subject=headers.get('Subject', ''),
                sender=headers.get('From', ''),
                date=self._parse_email_date(headers.get('Date', '')),
                body=body,
                thread_id=message.get('threadId', ''),
                labels=message.get('labelIds', [])
            )
            
            return email_data
            
        except Exception as e:
            logger.error(f"Error getting email details for {message_id}: {str(e)}")
            return None
    
    def _extract_email_body(self, payload: Dict) -> str:
        """
        Extract email body from Gmail API payload
        
        Args:
            payload: Gmail message payload
            
        Returns:
            str: Email body text
        """
        body = ""
        
        try:
            # Handle different payload structures
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        data = part['body']['data']
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
                        break
                    elif part['mimeType'] == 'text/html':
                        data = part['body']['data']
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
            else:
                if payload['body'].get('data'):
                    body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting email body: {str(e)}")
            body = ""
        
        return body
    
    def _parse_email_date(self, date_string: str) -> datetime:
        """
        Parse email date string to datetime object
        
        Args:
            date_string: Email date string
            
        Returns:
            datetime: Parsed datetime object
        """
        try:
            # Handle different date formats
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_string)
        except Exception as e:
            logger.error(f"Error parsing date {date_string}: {str(e)}")
            return datetime.now()
    
    def analyze_negotiation_content(self, email: EmailData) -> NegotiationAnalysis:
        """
        Analyze email content to determine if it's a negotiation or followup
        
        Args:
            email: EmailData object to analyze
            
        Returns:
            NegotiationAnalysis: Analysis results
        """
        logger.info(f"Analyzing email: {email.subject}")
        
        # Combine subject and body for analysis
        full_text = f"{email.subject} {email.body}".lower()
        
        # Check for negotiation patterns
        pattern_matches = []
        for pattern in self.negotiation_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                pattern_matches.append(pattern)
        
        # Count keyword matches by category
        keyword_matches = {category: [] for category in self.negotiation_keywords}
        total_keywords = 0
        
        for category, keywords in self.negotiation_keywords.items():
            for keyword in keywords:
                if keyword in full_text:
                    keyword_matches[category].append(keyword)
                    total_keywords += 1
        
        # Determine if it's a negotiation email
        is_negotiation = (
            len(pattern_matches) > 0 or
            total_keywords >= 3 or
            (keyword_matches['negotiation'] and keyword_matches['supplier'])
        )
        
        # Determine if it's a followup
        is_followup = (
            len(keyword_matches['followup']) > 0 and
            (keyword_matches['negotiation'] or keyword_matches['supplier'])
        )
        
        # Determine negotiation type
        negotiation_type = 'unknown'
        if keyword_matches['price']:
            negotiation_type = 'price_negotiation'
        elif keyword_matches['terms']:
            negotiation_type = 'terms_negotiation'
        elif is_followup:
            negotiation_type = 'followup'
        
        # Calculate confidence score
        confidence_score = min(1.0, (len(pattern_matches) * 0.3 + total_keywords * 0.1))
        
        # All found keywords
        all_keywords = []
        for keywords_list in keyword_matches.values():
            all_keywords.extend(keywords_list)
        
        # Reason for classification
        reason = f"Patterns: {len(pattern_matches)}, Keywords: {total_keywords}"
        if pattern_matches:
            reason += f", Matched patterns: {pattern_matches[:2]}"
        
        analysis = NegotiationAnalysis(
            is_negotiation=is_negotiation,
            is_followup=is_followup,
            negotiation_type=negotiation_type,
            confidence_score=confidence_score,
            keywords_found=all_keywords,
            reason=reason
        )
        
        logger.info(f"Analysis complete - Negotiation: {is_negotiation}, Followup: {is_followup}, Score: {confidence_score:.2f}")
        return analysis
    
    def process_emails(self) -> Dict[str, List[EmailData]]:
        """
        Main processing function to fetch and analyze emails
        
        Returns:
            Dict containing categorized emails
        """
        logger.info("Starting email processing...")
        
        # Authenticate with Gmail
        if not self.authenticate():
            logger.error("Failed to authenticate with Gmail")
            return {}
        
        # Fetch emails since last run
        emails = self.fetch_emails_since_last_run()
        
        if not emails:
            logger.info("No emails to process")
            return {'negotiation': [], 'followup': [], 'ignored': []}
        
        # Categorize emails
        results = {
            'negotiation': [],
            'followup': [],
            'ignored': []
        }
        
        for email in emails:
            try:
                analysis = self.analyze_negotiation_content(email)
                
                if analysis.is_negotiation and not analysis.is_followup:
                    results['negotiation'].append(email)
                    logger.info(f"Classified as NEGOTIATION: {email.subject}")
                elif analysis.is_followup:
                    results['followup'].append(email)
                    logger.info(f"Classified as FOLLOWUP: {email.subject}")
                else:
                    results['ignored'].append(email)
                    logger.info(f"Classified as IGNORED: {email.subject}")
                    
            except Exception as e:
                logger.error(f"Error processing email {email.id}: {str(e)}")
                results['ignored'].append(email)
        
        # Update last run time
        self.update_last_run_time()
        
        logger.info(f"Processing complete - Negotiation: {len(results['negotiation'])}, "
                   f"Followup: {len(results['followup'])}, Ignored: {len(results['ignored'])}")
        
        return results
    
    def generate_report(self, results: Dict[str, List[EmailData]]) -> str:
        """
        Generate a summary report of processed emails
        
        Args:
            results: Categorized email results
            
        Returns:
            str: Report string
        """
        report = f"""
=== Gmail Negotiation Analysis Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- New Negotiations: {len(results['negotiation'])}
- Followup Threads: {len(results['followup'])}
- Ignored Emails: {len(results['ignored'])}

Negotiation Emails:
"""
        for email in results['negotiation']:
            report += f"- {email.subject} (from: {email.sender})\n"
        
        report += "\nFollowup Emails:\n"
        for email in results['followup']:
            report += f"- {email.subject} (from: {email.sender})\n"
        
        return report

def main():
    """
    Main function to run the Gmail negotiation agent
    """
    try:
        # Initialize agent
        agent = GmailNegotiationAgent()
        
        # Process emails
        results = agent.process_emails()
        
        # Generate and print report
        report = agent.generate_report(results)
        print(report)
        
        # Log summary
        logger.info("Gmail negotiation agent completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
