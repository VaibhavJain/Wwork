"""
Supplier Price Negotiation Reply Agent using LangChain 0.3 and Azure OpenAI

This agent analyzes supplier price hike requests and generates professional
counter-offer email responses based on fact verification and negotiation strategy.

Author: Python Developer
Version: 1.0.0
Dependencies: langchain 0.3.x, azure-openai, pydantic
"""

import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# LangChain 0.3 imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.types import EmailStr

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('supplier_reply_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JustificationStatus(str, Enum):
    """Enumeration for justification validation status"""
    VALID = "valid"
    INVALID = "invalid"
    PARTIALLY_VALID = "partially_valid"
    REQUIRES_VERIFICATION = "requires_verification"

class NegotiationStrategy(str, Enum):
    """Enumeration for negotiation strategies"""
    ACCEPT = "accept"
    COUNTER_OFFER = "counter_offer"
    REJECT = "reject"
    REQUEST_DATA = "request_data"
    CONDITIONAL_ACCEPT = "conditional_accept"

class UrgencyLevel(str, Enum):
    """Enumeration for response urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ProductNegotiation:
    """Data class for individual product negotiation details"""
    product_name: str
    current_price: float
    requested_price: float
    counter_offer_price: Optional[float] = None
    justification: str = ""
    our_analysis: str = ""

class SupplierEmail(BaseModel):
    """Pydantic model for supplier email data"""
    sender: EmailStr = Field(description="Supplier email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    received_date: datetime = Field(description="Email received timestamp")
    supplier_name: str = Field(description="Supplier company name")
    products: List[ProductNegotiation] = Field(description="List of products under negotiation")
    
    @validator('body')
    def validate_body(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Email body too short")
        return v

class JustificationAnalysis(BaseModel):
    """Analysis of supplier's price hike justifications"""
    justification_type: str = Field(description="Type of justification (raw_material, operational_cost, etc.)")
    claimed_impact: float = Field(description="Claimed percentage impact on price")
    status: JustificationStatus = Field(description="Validity status of the justification")
    evidence_quality: str = Field(description="Quality of provided evidence: strong, moderate, weak, none")
    market_data_verification: bool = Field(description="Whether market data supports the claim")
    reasoning: str = Field(description="Detailed reasoning for the analysis")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis (0-1)")

class NegotiationResponse(BaseModel):
    """Generated negotiation response details"""
    strategy: NegotiationStrategy = Field(description="Recommended negotiation strategy")
    counter_offers: List[ProductNegotiation] = Field(description="Counter offers for products")
    key_arguments: List[str] = Field(description="Key negotiation arguments to use")
    tone: str = Field(description="Recommended tone: professional, firm, collaborative, etc.")
    urgency: UrgencyLevel = Field(description="Response urgency level")
    next_steps: List[str] = Field(description="Suggested next steps")
    email_subject: str = Field(description="Suggested email subject line")
    email_body: str = Field(description="Generated email response body")

class SupplierReplyAgent:
    """
    Advanced supplier price negotiation agent using LangChain 0.3 and Azure OpenAI
    
    This agent analyzes supplier price hike requests, validates justifications,
    and generates professional counter-offer email responses.
    """
    
    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        azure_deployment_name: str,
        azure_api_version: str = "2024-02-01",
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        """
        Initialize the Supplier Reply Agent
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            azure_deployment_name: Azure OpenAI deployment name
            azure_api_version: Azure OpenAI API version
            model_name: Model name (gpt-4, gpt-35-turbo)
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens per response
        """
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_deployment_name = azure_deployment_name
        self.azure_api_version = azure_api_version
        self.model_name = model_name
        
        # Initialize Azure OpenAI LLM
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                azure_deployment=azure_deployment_name,
                api_version=azure_api_version,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Azure OpenAI LLM initialized successfully with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI LLM: {str(e)}")
            raise
        
        # Initialize LangChain components
        self._setup_langchain_components()
        
        # Load market data and negotiation rules
        self.market_data = self._load_market_data()
        self.negotiation_rules = self._load_negotiation_rules()
        
        logger.info("Supplier Reply Agent initialized successfully")
    
    def _setup_langchain_components(self):
        """Setup LangChain components for email analysis and response generation"""
        try:
            # Text splitter for long emails
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            # Output parsers
            self.justification_parser = PydanticOutputParser(pydantic_object=JustificationAnalysis)
            self.response_parser = PydanticOutputParser(pydantic_object=NegotiationResponse)
            
            # Justification analysis prompt
            self.justification_analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert procurement analyst specializing in supplier price analysis and market intelligence.

Analyze the following supplier's price hike justification:

SUPPLIER EMAIL:
From: {supplier_name}
Subject: {subject}
Content: {email_body}

PRODUCT DETAILS:
{product_details}

MARKET CONTEXT:
{market_data}

ANALYSIS INSTRUCTIONS:
1. Identify all justifications provided (raw material costs, operational expenses, market conditions, etc.)
2. Evaluate the validity of each justification against current market data
3. Assess the quality of evidence provided
4. Determine if the claimed price impact is reasonable
5. Provide confidence score for your analysis

{format_instructions}

Provide thorough analysis with specific reasoning for each assessment.
""")
            
            # Response generation prompt
            self.response_generation_prompt = ChatPromptTemplate.from_template("""
You are an expert procurement negotiator tasked with crafting a professional response to a supplier's price hike request.

SUPPLIER REQUEST ANALYSIS:
{justification_analysis}

COMPANY POSITION:
- Budget constraints: {budget_constraints}
- Strategic importance: {supplier_importance}
- Alternative suppliers: {alternatives_available}
- Contract terms: {contract_terms}

NEGOTIATION GUIDELINES:
{negotiation_rules}

RESPONSE REQUIREMENTS:
1. Professional and respectful tone
2. Acknowledge valid concerns while challenging invalid ones
3. Present counter-offers with clear justification
4. Maintain supplier relationship
5. Protect company interests
6. Include specific next steps

{format_instructions}

Generate a complete email response that balances firmness with collaboration.
""")
            
            # Create analysis chain
            self.justification_chain = (
                self.justification_analysis_prompt
                | self.llm
                | self.justification_parser
            )
            
            # Create response generation chain
            self.response_chain = (
                self.response_generation_prompt
                | self.llm
                | self.response_parser
            )
            
            logger.info("LangChain components setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up LangChain components: {str(e)}")
            raise
    
    def _load_market_data(self) -> Dict[str, Any]:
        """Load market data for justification verification"""
        try:
            # In a real implementation, this would load from external data sources
            market_data = {
                "raw_materials": {
                    "steel": {"current_trend": "increasing", "percentage_change": 15.2},
                    "aluminum": {"current_trend": "stable", "percentage_change": 2.1},
                    "oil": {"current_trend": "decreasing", "percentage_change": -8.5},
                    "copper": {"current_trend": "increasing", "percentage_change": 12.8}
                },
                "operational_costs": {
                    "labor": {"current_trend": "increasing", "percentage_change": 8.5},
                    "energy": {"current_trend": "increasing", "percentage_change": 18.3},
                    "transportation": {"current_trend": "stable", "percentage_change": 3.2}
                },
                "last_updated": datetime.now().isoformat()
            }
            logger.info("Market data loaded successfully")
            return market_data
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return {}
    
    def _load_negotiation_rules(self) -> Dict[str, Any]:
        """Load company negotiation rules and policies"""
        try:
            rules = {
                "max_acceptable_increase": 15.0,  # Maximum acceptable price increase percentage
                "counter_offer_reduction": 50.0,  # Percentage to reduce from supplier's ask
                "evidence_requirements": {
                    "raw_material": "Market data or supplier invoices required",
                    "operational": "Detailed cost breakdown required",
                    "regulatory": "Official documentation required"
                },
                "escalation_thresholds": {
                    "critical_supplier": 20.0,
                    "regular_supplier": 10.0,
                    "alternative_available": 5.0
                },
                "response_timeframes": {
                    "standard": 5,  # Business days
                    "urgent": 2,
                    "critical": 1
                }
            }
            logger.info("Negotiation rules loaded successfully")
            return rules
        except Exception as e:
            logger.error(f"Error loading negotiation rules: {str(e)}")
            return {}
    
    def parse_supplier_email(self, email_content: str, supplier_info: Dict[str, Any]) -> SupplierEmail:
        """
        Parse supplier email and extract negotiation details
        
        Args:
            email_content: Raw email content
            supplier_info: Additional supplier information
            
        Returns:
            SupplierEmail: Parsed email data
            
        Raises:
            ValueError: If email parsing fails
        """
        try:
            logger.info(f"Parsing supplier email from: {supplier_info.get('sender', 'Unknown')}")
            
            # Extract products and pricing from email content
            products = self._extract_product_details(email_content)
            
            supplier_email = SupplierEmail(
                sender=supplier_info.get('sender', ''),
                subject=supplier_info.get('subject', ''),
                body=email_content,
                received_date=supplier_info.get('received_date', datetime.now()),
                supplier_name=supplier_info.get('supplier_name', ''),
                products=products
            )
            
            logger.info(f"Successfully parsed email with {len(products)} products")
            return supplier_email
            
        except Exception as e:
            logger.error(f"Error parsing supplier email: {str(e)}")
            raise ValueError(f"Failed to parse supplier email: {str(e)}")
    
    def _extract_product_details(self, email_content: str) -> List[ProductNegotiation]:
        """Extract product pricing details from email content"""
        try:
            products = []
            
            # Use regex patterns to extract pricing information
            # This is a simplified example - in production, use more sophisticated parsing
            price_patterns = [
                r'(\w+(?:\s+\w+)*)\s*:?\s*\$?(\d+(?:\.\d{2})?)\s*(?:to|→)\s*\$?(\d+(?:\.\d{2})?)',
                r'(\w+(?:\s+\w+)*)\s*price\s*increase.*?\$?(\d+(?:\.\d{2})?)\s*to\s*\$?(\d+(?:\.\d{2})?)',
            ]
            
            for pattern in price_patterns:
                matches = re.finditer(pattern, email_content, re.IGNORECASE)
                for match in matches:
                    product_name = match.group(1).strip()
                    current_price = float(match.group(2))
                    requested_price = float(match.group(3))
                    
                    product = ProductNegotiation(
                        product_name=product_name,
                        current_price=current_price,
                        requested_price=requested_price,
                        justification="Extracted from email content"
                    )
                    products.append(product)
            
            if not products:
                # Fallback: create a generic product entry
                logger.warning("No specific product pricing found, creating generic entry")
                products.append(ProductNegotiation(
                    product_name="General Products",
                    current_price=100.0,
                    requested_price=115.0,
                    justification="Price increase mentioned in email"
                ))
            
            logger.info(f"Extracted {len(products)} products from email")
            return products
            
        except Exception as e:
            logger.error(f"Error extracting product details: {str(e)}")
            return []
    
    def analyze_justifications(self, supplier_email: SupplierEmail) -> List[JustificationAnalysis]:
        """
        Analyze supplier's price hike justifications
        
        Args:
            supplier_email: Parsed supplier email
            
        Returns:
            List[JustificationAnalysis]: Analysis results for each justification
        """
        try:
            logger.info(f"Analyzing justifications from {supplier_email.supplier_name}")
            
            # Prepare product details for analysis
            product_details = "\n".join([
                f"- {p.product_name}: ${p.current_price} → ${p.requested_price} "
                f"(+{((p.requested_price - p.current_price) / p.current_price * 100):.1f}%)"
                for p in supplier_email.products
            ])
            
            # Prepare analysis input
            analysis_input = {
                "supplier_name": supplier_email.supplier_name,
                "subject": supplier_email.subject,
                "email_body": supplier_email.body[:3000],  # Limit for token efficiency
                "product_details": product_details,
                "market_data": json.dumps(self.market_data, indent=2),
                "format_instructions": self.justification_parser.get_format_instructions()
            }
            
            # Track API usage
            with get_openai_callback() as cb:
                try:
                    # Run justification analysis
                    analysis = self.justification_chain.invoke(analysis_input)
                    
                    logger.info(f"Justification analysis completed - Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
                    
                    # Return as list for consistency (can be extended for multiple justifications)
                    return [analysis] if isinstance(analysis, JustificationAnalysis) else [analysis]
                    
                except Exception as chain_error:
                    logger.error(f"LangChain analysis failed: {str(chain_error)}")
                    return self._fallback_justification_analysis(supplier_email)
            
        except Exception as e:
            logger.error(f"Error in justification analysis: {str(e)}")
            return self._fallback_justification_analysis(supplier_email)
    
    def _fallback_justification_analysis(self, supplier_email: SupplierEmail) -> List[JustificationAnalysis]:
        """Fallback analysis when LLM fails"""
        logger.warning("Using fallback justification analysis")
        
        try:
            # Simple rule-based analysis
            email_lower = supplier_email.body.lower()
            
            # Determine justification type
            if any(word in email_lower for word in ['raw material', 'steel', 'aluminum', 'copper']):
                justification_type = "raw_material"
            elif any(word in email_lower for word in ['labor', 'operational', 'overhead']):
                justification_type = "operational_cost"
            else:
                justification_type = "general"
            
            # Calculate claimed impact
            avg_increase = sum(
                (p.requested_price - p.current_price) / p.current_price * 100 
                for p in supplier_email.products
            ) / len(supplier_email.products)
            
            return [JustificationAnalysis(
                justification_type=justification_type,
                claimed_impact=avg_increase,
                status=JustificationStatus.REQUIRES_VERIFICATION,
                evidence_quality="unknown",
                market_data_verification=False,
                reasoning="Fallback analysis due to LLM failure - requires manual review",
                confidence_score=0.3
            )]
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {str(e)}")
            return []
    
    def generate_response(
        self,
        supplier_email: SupplierEmail,
        justification_analyses: List[JustificationAnalysis],
        company_context: Optional[Dict[str, Any]] = None
    ) -> NegotiationResponse:
        """
        Generate professional response email to supplier
        
        Args:
            supplier_email: Original supplier email
            justification_analyses: Analysis of supplier's justifications
            company_context: Additional company context and constraints
            
        Returns:
            NegotiationResponse: Generated response with strategy and email content
        """
        try:
            logger.info(f"Generating response for {supplier_email.supplier_name}")
            
            # Default company context
            if not company_context:
                company_context = {
                    "budget_constraints": "Moderate budget pressure, seeking cost optimization",
                    "supplier_importance": "Important strategic supplier",
                    "alternatives_available": "Limited alternatives available",
                    "contract_terms": "Annual contract with quarterly reviews"
                }
            
            # Prepare response generation input
            response_input = {
                "justification_analysis": json.dumps([analysis.dict() for analysis in justification_analyses], indent=2),
                "budget_constraints": company_context.get("budget_constraints", ""),
                "supplier_importance": company_context.get("supplier_importance", ""),
                "alternatives_available": company_context.get("alternatives_available", ""),
                "contract_terms": company_context.get("contract_terms", ""),
                "negotiation_rules": json.dumps(self.negotiation_rules, indent=2),
                "format_instructions": self.response_parser.get_format_instructions()
            }
            
            # Track API usage
            with get_openai_callback() as cb:
                try:
                    # Generate response
                    response = self.response_chain.invoke(response_input)
                    
                    # Generate counter offers for products
                    response.counter_offers = self._generate_counter_offers(
                        supplier_email.products, justification_analyses
                    )
                    
                    logger.info(f"Response generated successfully - Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
                    return response
                    
                except Exception as chain_error:
                    logger.error(f"Response generation chain failed: {str(chain_error)}")
                    return self._fallback_response_generation(supplier_email, justification_analyses)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._fallback_response_generation(supplier_email, justification_analyses)
    
    def _generate_counter_offers(
        self,
        products: List[ProductNegotiation],
        analyses: List[JustificationAnalysis]
    ) -> List[ProductNegotiation]:
        """Generate counter offers based on analysis"""
        try:
            counter_offers = []
            
            for product in products:
                # Calculate counter offer based on analysis validity
                if analyses and analyses[0].status == JustificationStatus.VALID:
                    # Accept partial increase for valid justifications
                    reduction_factor = 0.7  # Accept 70% of requested increase
                else:
                    # Offer minimal increase for invalid/unverified justifications
                    reduction_factor = 0.3  # Accept only 30% of requested increase
                
                increase_amount = product.requested_price - product.current_price
                counter_increase = increase_amount * reduction_factor
                counter_offer_price = product.current_price + counter_increase
                
                counter_product = ProductNegotiation(
                    product_name=product.product_name,
                    current_price=product.current_price,
                    requested_price=product.requested_price,
                    counter_offer_price=round(counter_offer_price, 2),
                    justification=f"Counter offer based on analysis (reduction factor: {reduction_factor})",
                    our_analysis=f"Applied {reduction_factor*100}% of requested increase"
                )
                counter_offers.append(counter_product)
            
            return counter_offers
            
        except Exception as e:
            logger.error(f"Error generating counter offers: {str(e)}")
            return products  # Return original products as fallback
    
    def _fallback_response_generation(
        self,
        supplier_email: SupplierEmail,
        analyses: List[JustificationAnalysis]
    ) -> NegotiationResponse:
        """Fallback response generation when LLM fails"""
        logger.warning("Using fallback response generation")
        
        try:
            # Generate basic counter offers
            counter_offers = self._generate_counter_offers(supplier_email.products, analyses)
            
            # Create basic response
            return NegotiationResponse(
                strategy=NegotiationStrategy.COUNTER_OFFER,
                counter_offers=counter_offers,
                key_arguments=[
                    "Request for detailed justification documentation",
                    "Market analysis suggests more modest increases",
                    "Propose gradual implementation of price adjustments"
                ],
                tone="professional",
                urgency=UrgencyLevel.MEDIUM,
                next_steps=[
                    "Review provided documentation",
                    "Schedule negotiation meeting",
                    "Provide counter proposal details"
                ],
                email_subject=f"Re: {supplier_email.subject} - Counter Proposal",
                email_body=self._generate_fallback_email_body(supplier_email, counter_offers)
            )
            
        except Exception as e:
            logger.error(f"Fallback response generation failed: {str(e)}")
            raise
    
    def _generate_fallback_email_body(
        self,
        supplier_email: SupplierEmail,
        counter_offers: List[ProductNegotiation]
    ) -> str:
        """Generate basic email body for fallback scenario"""
        
        counter_offer_details = "\n".join([
            f"- {co.product_name}: We propose ${co.counter_offer_price} "
            f"(vs. your requested ${co.requested_price})"
            for co in counter_offers
        ])
        
        return f"""Dear {supplier_email.supplier_name} Team,

Thank you for your email regarding price adjustments. We have reviewed your request and understand the challenges facing your industry.

After careful consideration and market analysis, we would like to propose the following counter-offer:

{counter_offer_details}

We believe this adjustment balances the realities of current market conditions with our budget requirements. We would appreciate the opportunity to discuss this proposal further and explore ways to maintain our productive partnership.

Please let us know your availability for a discussion in the coming week.

Best regards,
Procurement Team

---
This email was generated by automated negotiation system.
Please review before sending."""
    
    def process_supplier_request(
        self,
        email_content: str,
        supplier_info: Dict[str, Any],
        company_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[SupplierEmail, List[JustificationAnalysis], NegotiationResponse]:
        """
        Main processing function - complete workflow for supplier price request
        
        Args:
            email_content: Raw supplier email content
            supplier_info: Supplier information dictionary
            company_context: Company context and constraints
            
        Returns:
            Tuple containing parsed email, analyses, and generated response
        """
        try:
            logger.info("Starting complete supplier request processing workflow")
            
            # Step 1: Parse supplier email
            logger.info("Step 1: Parsing supplier email")
            supplier_email = self.parse_supplier_email(email_content, supplier_info)
            
            # Step 2: Analyze justifications
            logger.info("Step 2: Analyzing price hike justifications")
            justification_analyses = self.analyze_justifications(supplier_email)
            
            # Step 3: Generate response
            logger.info("Step 3: Generating negotiation response")
            response = self.generate_response(supplier_email, justification_analyses, company_context)
            
            logger.info("Supplier request processing completed successfully")
            return supplier_email, justification_analyses, response
            
        except Exception as e:
            logger.error(f"Error in complete processing workflow: {str(e)}")
            raise
    
    def save_negotiation_record(
        self,
        supplier_email: SupplierEmail,
        analyses: List[JustificationAnalysis],
        response: NegotiationResponse,
        output_file: Optional[str] = None
    ) -> str:
        """
        Save complete negotiation record to file
        
        Args:
            supplier_email: Parsed supplier email
            analyses: Justification analyses
            response: Generated response
            output_file: Output file path (optional)
            
        Returns:
            str: Path to saved file
        """
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"negotiation_record_{timestamp}.json"
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "supplier_email": supplier_email.dict(),
                "justification_analyses": [analysis.dict() for analysis in analyses],
                "generated_response": response.dict(),
                "processing_metadata": {
                    "agent_version": "1.0.0",
                    "model_used": self.model_name,
                    "azure_deployment": self.azure_deployment_name
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, default=str)
            
            logger.info(f"Negotiation record saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving negotiation record: {str(e)}")
            raise

def main():
    """
    Main function demonstrating the Supplier Reply Agent usage
    """
    try:
        # Initialize agent with Azure OpenAI credentials
        agent = SupplierReplyAgent(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            model_name="gpt-4"
        )
        
        # Example supplier email content
        email_content = """
Subject: Price Adjustment Request - Q4 2024

Dear Procurement Team,

I hope this email finds you well. I am writing to inform you of necessary price adjustments 
for our products effective January 2025.

Due to significant increases in raw material costs, particularly steel prices which have 
risen 18% in the last quarter, and increased operational costs including a 12% rise in 
labor expenses, we need to adjust our pricing structure:

- Product A: $100.00 → $118.00 (18% increase)
- Product B: $75.00 → $85.50 (14% increase)  
- Product C: $150.00 → $169.50 (13% increase)

We have absorbed these cost increases for as long as possible but can no longer maintain 
current pricing. We value our partnership and hope for your understanding.

Please let me know if you need any supporting documentation.

Best regards,
John Smith
ABC Suppliers Inc.
        """
        
        # Supplier information
        supplier_info = {
            "sender": "john.smith@abcsuppliers.com",
            "subject": "Price Adjustment Request - Q4 2024",
            "supplier_name": "ABC Suppliers Inc.",
            "received_date": datetime.now()
        }
        
        # Company context
        company_context = {
            "budget_constraints": "Tight budget with 5% cost reduction target",
            "supplier_importance": "Critical supplier - limited alternatives",
            "alternatives_available": "2-3 alternative suppliers identified",
            "contract_terms": "Annual contract with quarterly price review clause"
        }
        
        # Process the supplier request
        print("Processing supplier price hike request...")
        supplier_email, analyses, response = agent.process_supplier_