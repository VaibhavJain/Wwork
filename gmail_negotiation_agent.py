import os
import pickle
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
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

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gmail_langchain_agent.log'),
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

class NegotiationAnalysis(BaseModel):
    """Pydantic model for LangChain output parsing"""
    is_negotiation: bool = Field(description="Whether the email is a negotiation request")
    is_followup: bool = Field(description="Whether the email is a followup to existing negotiation")
    negotiation_type: str = Field(description="Type of negotiation: price_negotiation, terms_negotiation, followup, or none")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    keywords_found: List[str] = Field(description="List of relevant keywords found")
    reasoning: str = Field(description="Explanation of the classification decision")
    supplier_mentioned: bool = Field(description="Whether supplier/vendor is mentioned")
    urgency_level: str = Field(description="Urgency level: low, medium, high")

class GmailLangChainNegotiationAgent:
    """
    Gmail agent for identifying supplier negotiation emails using LangChain 0.3
    """
    
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.pickle', 
                 openai_api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the Gmail LangChain agent
        
        Args:
            credentials_file: Path to Gmail API credentials JSON file
            token_file: Path to store OAuth2 token
            openai_api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
            model_name: OpenAI model to use for analysis
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.last_run_file = 'last_run.txt'
        
        # Initialize OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint="https://.openai.azure.com/",
            api_key="",
            azure_deployment="gpt-4.1-mini",
            api_version="2024-05-01-preview",
            model_name="gpt-4.1-mini",
            temperature=0.1
        )
        
        # Initialize LangChain components
        self._setup_langchain_components()
        
        logger.info(f"Gmail LangChain Agent initialized with model: {model_name}")
    
    def _setup_langchain_components(self):
        """
        Setup LangChain components for email analysis
        """
        # Text splitter for long emails
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Output parser for structured responses
        self.output_parser = PydanticOutputParser(pydantic_object=NegotiationAnalysis)
        
        # Create analysis prompt template
        self.analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert email classifier specializing in identifying supplier negotiation emails.

Analyze the following email and classify it according to these criteria:

EMAIL CONTENT:
Subject: {subject}
From: {sender}
Body: {body}

CLASSIFICATION RULES:
1. NEGOTIATION EMAIL: Contains explicit requests for price quotes, contract negotiations, vendor proposals, or procurement discussions
2. FOLLOWUP EMAIL: References ongoing negotiations, contains status updates, reminders, or responses to previous negotiation threads
3. IGNORE: General business emails, marketing, notifications, or unrelated content

ANALYSIS FACTORS:
- Keywords: price, quote, negotiation, vendor, supplier, contract, proposal, terms, procurement, RFQ, RFP
- Context: Business relationship, purchasing intent, commercial discussions
- Urgency: Deadline mentions, urgent language, time-sensitive requests
- Supplier identification: Mentions of vendors, suppliers, or procurement teams

{format_instructions}

Provide detailed reasoning for your classification decision.
""")
        
        # Create the analysis chain
        self.analysis_chain = (
            self.analysis_prompt 
            | self.llm 
            | self.output_parser
        )
        
        # Batch processing chain for multiple emails
        self.batch_chain = RunnableLambda(self._process_email_batch)
        
        logger.info("LangChain components initialized successfully")
    
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
                        # Remove HTML tags for cleaner analysis
                        import re
                        html_body = base64.urlsafe_b64decode(data).decode('utf-8')
                        body = re.sub('<[^<]+?>', '', html_body)
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
        Analyze email content using LangChain to determine if it's a negotiation or followup
        
        Args:
            email: EmailData object to analyze
            
        Returns:
            NegotiationAnalysis: Analysis results from LangChain
        """
        logger.info(f"Analyzing email with LangChain: {email.subject}")
        
        try:
            # Prepare input for LangChain
            analysis_input = {
                "subject": email.subject,
                "sender": email.sender,
                "body": email.body[:2000],  # Limit body length for API efficiency
                "format_instructions": self.output_parser.get_format_instructions()
            }
            
            # Track token usage
            with get_openai_callback() as cb:
                # Run analysis chain
                analysis = self.analysis_chain.invoke(analysis_input)
                
                logger.info(f"LLM Analysis - Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
            
            logger.info(f"LangChain analysis complete - Negotiation: {analysis.is_negotiation}, "
                       f"Followup: {analysis.is_followup}, Score: {analysis.confidence_score:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LangChain analysis for email {email.id}: {str(e)}")
            
            # Fallback to rule-based analysis
            return self._fallback_analysis(email)
    
    def _fallback_analysis(self, email: EmailData) -> NegotiationAnalysis:
        """
        Fallback rule-based analysis when LangChain fails
        
        Args:
            email: EmailData object to analyze
            
        Returns:
            NegotiationAnalysis: Basic rule-based analysis
        """
        logger.warning("Using fallback rule-based analysis")
        
        full_text = f"{email.subject} {email.body}".lower()
        
        # Basic keyword detection
        negotiation_keywords = ['price', 'quote', 'negotiation', 'vendor', 'supplier', 'contract', 'proposal']
        followup_keywords = ['follow up', 'followup', 'status', 'reminder', 'checking in']
        
        found_keywords = [kw for kw in negotiation_keywords + followup_keywords if kw in full_text]
        
        is_negotiation = any(kw in full_text for kw in negotiation_keywords)
        is_followup = any(kw in full_text for kw in followup_keywords)
        
        return NegotiationAnalysis(
            is_negotiation=is_negotiation,
            is_followup=is_followup,
            negotiation_type='followup' if is_followup else 'price_negotiation' if is_negotiation else 'none',
            confidence_score=0.5 if is_negotiation or is_followup else 0.1,
            keywords_found=found_keywords,
            reasoning="Fallback rule-based analysis due to LLM error",
            supplier_mentioned='supplier' in full_text or 'vendor' in full_text,
            urgency_level='medium' if 'urgent' in full_text else 'low'
        )
    
    def _process_email_batch(self, emails: List[EmailData]) -> List[Tuple[EmailData, NegotiationAnalysis]]:
        """
        Process a batch of emails for analysis
        
        Args:
            emails: List of EmailData objects
            
        Returns:
            List of tuples containing email and analysis
        """
        results = []
        total_cost = 0.0
        
        logger.info(f"Processing batch of {len(emails)} emails")
        
        for email in emails:
            try:
                analysis = self.analyze_negotiation_content(email)
                results.append((email, analysis))
            except Exception as e:
                logger.error(f"Error processing email {email.id}: {str(e)}")
                # Add with fallback analysis
                fallback = self._fallback_analysis(email)
                results.append((email, fallback))
        
        logger.info(f"Batch processing complete. Processed {len(results)} emails")
        return results
    
    def create_langchain_documents(self, emails: List[EmailData]) -> List[Document]:
        """
        Convert emails to LangChain Document objects for advanced processing
        
        Args:
            emails: List of EmailData objects
            
        Returns:
            List[Document]: LangChain documents
        """
        documents = []
        
        for email in emails:
            # Create document with email content and metadata
            doc = Document(
                page_content=f"Subject: {email.subject}\nBody: {email.body}",
                metadata={
                    'id': email.id,
                    'sender': email.sender,
                    'date': email.date.isoformat(),
                    'thread_id': email.thread_id,
                    'labels': email.labels
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} LangChain documents")
        return documents
    
    def process_emails(self) -> Dict[str, List[Tuple[EmailData, NegotiationAnalysis]]]:
        """
        Main processing function to fetch and analyze emails using LangChain
        
        Returns:
            Dict containing categorized emails with their analyses
        """
        logger.info("Starting LangChain email processing...")
        
        # Authenticate with Gmail
        if not self.authenticate():
            logger.error("Failed to authenticate with Gmail")
            return {}
        
        # Fetch emails since last run
        emails = self.fetch_emails_since_last_run()
        
        if not emails:
            logger.info("No emails to process")
            return {'negotiation': [], 'followup': [], 'ignored': []}
        
        # Process emails using LangChain
        email_analyses = self._process_email_batch(emails)
        
        # Categorize emails based on LangChain analysis
        results = {
            'negotiation': [],
            'followup': [],
            'ignored': []
        }
        
        for email, analysis in email_analyses:
            try:
                if analysis.is_negotiation and not analysis.is_followup:
                    results['negotiation'].append((email, analysis))
                    logger.info(f"Classified as NEGOTIATION (confidence: {analysis.confidence_score:.2f}): {email.subject}")
                elif analysis.is_followup:
                    results['followup'].append((email, analysis))
                    logger.info(f"Classified as FOLLOWUP (confidence: {analysis.confidence_score:.2f}): {email.subject}")
                else:
                    results['ignored'].append((email, analysis))
                    logger.info(f"Classified as IGNORED (confidence: {analysis.confidence_score:.2f}): {email.subject}")
                    
            except Exception as e:
                logger.error(f"Error categorizing email {email.id}: {str(e)}")
                results['ignored'].append((email, analysis))
        
        # Update last run time
        self.update_last_run_time()
        
        logger.info(f"LangChain processing complete - Negotiation: {len(results['negotiation'])}, "
                   f"Followup: {len(results['followup'])}, Ignored: {len(results['ignored'])}")
        
        return results
    
    def generate_advanced_report(self, results: Dict[str, List[Tuple[EmailData, NegotiationAnalysis]]]) -> str:
        """
        Generate an advanced summary report with LangChain analysis details
        
        Args:
            results: Categorized email results with analyses
            
        Returns:
            str: Detailed report string
        """
        report = f"""
=== Gmail LangChain Negotiation Analysis Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- New Negotiations: {len(results['negotiation'])}
- Followup Threads: {len(results['followup'])}
- Ignored Emails: {len(results['ignored'])}

=== NEGOTIATION EMAILS ===
"""
        
        for email, analysis in results['negotiation']:
            report += f"""
Subject: {email.subject}
From: {email.sender}
Confidence: {analysis.confidence_score:.2f}
Type: {analysis.negotiation_type}
Keywords: {', '.join(analysis.keywords_found[:5])}
Reasoning: {analysis.reasoning[:100]}...
Urgency: {analysis.urgency_level}
---
"""
        
        report += "\n=== FOLLOWUP EMAILS ===\n"
        for email, analysis in results['followup']:
            report += f"""
Subject: {email.subject}
From: {email.sender}
Confidence: {analysis.confidence_score:.2f}
Keywords: {', '.join(analysis.keywords_found[:5])}
Reasoning: {analysis.reasoning[:100]}...
---
"""
        
        # Generate insights using LangChain
        insights = self._generate_insights(results)
        report += f"\n=== AI INSIGHTS ===\n{insights}\n"
        
        return report
    
    def _generate_insights(self, results: Dict[str, List[Tuple[EmailData, NegotiationAnalysis]]]) -> str:
        """
        Generate insights about the email analysis using LangChain
        
        Args:
            results: Analysis results
            
        Returns:
            str: Generated insights
        """
        try:
            # Prepare summary data for insight generation
            neg_count = len(results['negotiation'])
            followup_count = len(results['followup'])
            ignored_count = len(results['ignored'])
            
            # Extract common patterns
            all_keywords = []
            urgency_levels = []
            
            for category in results.values():
                for _, analysis in category:
                    all_keywords.extend(analysis.keywords_found)
                    urgency_levels.append(analysis.urgency_level)
            
            # Count frequencies
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            urgency_counts = Counter(urgency_levels)
            
            insights = f"""
• Total emails processed: {neg_count + followup_count + ignored_count}
• Negotiation rate: {(neg_count / (neg_count + followup_count + ignored_count)) * 100:.1f}%
• Most common keywords: {', '.join([k for k, v in keyword_counts.most_common(5)])}
• Urgency distribution: {dict(urgency_counts)}
• Active negotiation threads: {followup_count}
"""
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return "Insights generation failed"

def main():
    """
    Main function to run the Gmail LangChain negotiation agent
    """
    try:
        # Initialize agent (requires OPENAI_API_KEY environment variable)
        agent = GmailLangChainNegotiationAgent(
            model_name="scm-gpt-4.1-mini"  # or "gpt-4" for better accuracy
        )
        
        # Process emails using LangChain
        results = agent.process_emails()
        
        # Generate and print advanced report
        report = agent.generate_advanced_report(results)
        print(report)
        
        # Save detailed results to JSON
        results_data = {}
        for category, email_analyses in results.items():
            results_data[category] = []
            for email, analysis in email_analyses:
                results_data[category].append({
                    'email': {
                        'id': email.id,
                        'subject': email.subject,
                        'sender': email.sender,
                        'date': email.date.isoformat()
                    },
                    'analysis': analysis.dict()
                })
        
        with open('negotiation_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info("Gmail LangChain negotiation agent completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
