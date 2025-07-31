"
import os
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import re

# Data Models
class JustificationItem(BaseModel):
    """Individual justification item with categorization"""
    item: str = Field(description="The specific item mentioned (e.g., sugar, wheat, electricity)")
    category: str = Field(description="Category: raw_material, operational, tax, or logistics")
    change: str = Field(description="The percentage or amount of change mentioned")

class ProductNegotiation(BaseModel):
    """Product negotiation summary"""
    product_name: str = Field(description="Name of the product being negotiated")
    current_price: Optional[str] = Field(description="Current price if mentioned")
    requested_price: Optional[str] = Field(description="New requested price")
    hike_percentage: Optional[str] = Field(description="Percentage increase requested")
    justification_table: List[JustificationItem] = Field(description="Structured justifications")

class EmailSummary(BaseModel):
    """Complete email summary"""
    supplier_name: Optional[str] = Field(description="Name of the supplier if mentioned")
    products: List[ProductNegotiation] = Field(description="List of products being negotiated")
    overall_summary: str = Field(description="Brief overall summary of the negotiation")

class SupplierEmailAgent:
    """Agent to summarize supplier negotiation emails"""
   
    def __init__(self, azure_endpoint: str, api_key: str, deployment_name: str, api_version: str = "2024-02-15-preview"):
        """
        Initialize the agent with Azure OpenAI configuration
       
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            deployment_name: Name of your Azure OpenAI deployment
            api_version: API version to use
        """
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_deployment=deployment_name,
            api_version=api_version,
            temperature=0.1
        )
       
        self.parser = PydanticOutputParser(pydantic_object=EmailSummary)
        self.setup_agent()
   
    def setup_agent(self):
        """Setup the LangChain agent with tools"""
       
        # Create tools
        email_parser_tool = Tool(
            name="email_parser",
            description="Parse supplier negotiation email and extract structured information",
            func=self.parse_email_content
        )
       
        justification_categorizer_tool = Tool(
            name="justification_categorizer",
            description="Categorize justification items into raw_material, operational, tax, or logistics",
            func=self.categorize_justification
        )
       
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
       
        # Create agent
        tools = [email_parser_tool, justification_categorizer_tool]
        self.agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
   
    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return """
        You are a specialized agent for summarizing supplier negotiation emails. Your task is to:
       
        1. Extract product information including names, current prices, and requested prices
        2. Identify percentage hikes or price increases
        3. Parse justification reasons and categorize them into a structured table
        4. Categorize justification items as:
           - raw_material: ingredients, materials, commodities (sugar, wheat, oil, etc.)
           - operational: electricity, labor, fuel, maintenance, rent
           - tax: GST, customs duty, excise tax, VAT
           - logistics: transportation, shipping, warehousing, delivery
       
        Always extract specific percentage changes or amounts mentioned for each justification item.
        Provide clear, structured output following the specified schema.
       
        Format instructions:
        {format_instructions}
        """
   
    def categorize_justification(self, item_description: str) -> str:
        """Categorize a justification item"""
        item_lower = item_description.lower()
       
        # Raw materials
        raw_materials = ['sugar', 'wheat', 'oil', 'flour', 'milk', 'cocoa', 'vanilla', 'salt',
                        'ingredients', 'raw material', 'commodity', 'grain', 'spice']
       
        # Operational costs
        operational = ['electricity', 'labor', 'labour', 'fuel', 'gas', 'water', 'maintenance',
                      'rent', 'salary', 'wage', 'power', 'energy', 'utilities']
       
        # Tax related
        tax_items = ['tax', 'gst', 'vat', 'duty', 'customs', 'excise', 'tariff']
       
        # Logistics
        logistics = ['transport', 'shipping', 'delivery', 'freight', 'logistics', 'warehousing',
                    'storage', 'distribution']
       
        for keyword in raw_materials:
            if keyword in item_lower:
                return "raw_material"
       
        for keyword in operational:
            if keyword in item_lower:
                return "operational"
       
        for keyword in tax_items:
            if keyword in item_lower:
                return "tax"
       
        for keyword in logistics:
            if keyword in item_lower:
                return "logistics"
       
        return "operational"  # default category
   
    def parse_email_content(self, email_content: str) -> str:
        """Parse email content and extract structured information"""
       
        parsing_prompt = ChatPromptTemplate.from_template("""
        Parse the following supplier negotiation email and extract information according to this structure:

        Email Content:
        {email_content}

        Extract:
        1. Product name(s)
        2. Current price (if mentioned)
        3. Requested new price
        4. Percentage hike requested
        5. All justification reasons with specific percentage changes

        For justifications, identify each item mentioned (like sugar, wheat, electricity, labor) and the specific change percentage.

        {format_instructions}
        """)
       
        chain = parsing_prompt | self.llm | self.parser
       
        try:
            result = chain.invoke({
                "email_content": email_content,
                "format_instructions": self.parser.get_format_instructions()
            })
            return json.dumps(result.dict(), indent=2)
        except Exception as e:
            return f"Error parsing email: {str(e)}"
   
    def extract_justifications_from_text(self, text: str) -> List[JustificationItem]:
        """Extract justification items from text using regex and NLP"""
        justifications = []
       
        # Pattern to match items with percentage changes
        patterns = [
            r'(\w+(?:\s+\w+)*?)\s+(?:has\s+)?(?:increased|gone up|hiked|raised)\s+by\s+(\d+(?:\.\d+)?%)',
            r'(\w+(?:\s+\w+)*?)\s+(?:price|cost|rate|tariff)\s+(?:has\s+)?(?:increased|gone up)\s+by\s+(\d+(?:\.\d+)?%)',
            r'(\d+(?:\.\d+)?%)\s+(?:increase|hike|rise)\s+in\s+(\w+(?:\s+\w+)*?)'
        ]
       
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    if match.group(2).endswith('%'):
                        item = match.group(1).strip()
                        change = match.group(2)
                    else:
                        item = match.group(2).strip()
                        change = match.group(1)
                   
                    category = self.categorize_justification(item)
                    justifications.append(JustificationItem(
                        item=item,
                        category=category,
                        change=change
                    ))
       
        return justifications
   
    def summarize_email(self, email_content: str) -> Dict[str, Any]:
        """Main method to summarize supplier negotiation email"""
       
        # Enhanced prompt for better extraction
        enhanced_prompt = f"""
        Analyze this supplier negotiation email and provide a structured summary:

        Email Content:
        {email_content}

        Please extract:
        1. Product information (name, current price, requested price, hike percentage)
        2. Justification reasons formatted as a table with columns: item, category, change
        3. Overall summary

        Categorize justification items as:
        - raw_material: sugar, wheat, oil, ingredients, etc.
        - operational: electricity, labor, fuel, maintenance, rent, etc.
        - tax: GST, customs duty, VAT, etc.
        - logistics: transportation, shipping, warehousing, etc.

        {self.parser.get_format_instructions()}
        """
       
        try:
            # Use the LLM directly for better control
            response = self.llm.invoke(enhanced_prompt)
           
            # Try to parse the response
            try:
                parsed_result = self.parser.parse(response.content)
                return parsed_result.dict()
            except:
                # Fallback: manual extraction
                return self.manual_extraction(email_content)
               
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return self.manual_extraction(email_content)
   
    def manual_extraction(self, email_content: str) -> Dict[str, Any]:
        """Fallback manual extraction method"""
       
        # Extract product name
        product_match = re.search(r'(?:supplying|supply)\s+([^.]+?)(?:\s+at|\s+for)', email_content, re.IGNORECASE)
        product_name = product_match.group(1).strip() if product_match else "Unknown Product"
       
        # Extract prices
        price_matches = re.findall(r'\$(\d+(?:\.\d+)?)', email_content)
        current_price = f"${price_matches[0]}" if len(price_matches) > 0 else None
        requested_price = f"${price_matches[1]}" if len(price_matches) > 1 else None
       
        # Extract hike percentage
        hike_match = re.search(r'increase.*?by\s+(\d+(?:\.\d+)?%)', email_content, re.IGNORECASE)
        hike_percentage = hike_match.group(1) if hike_match else None
       
        # Extract justifications
        justifications = self.extract_justifications_from_text(email_content)
       
        return {
            "supplier_name": None,
            "products": [{
                "product_name": product_name,
                "current_price": current_price,
                "requested_price": requested_price,
                "hike_percentage": hike_percentage,
                "justification_table": [j.dict() for j in justifications]
            }],
            "overall_summary": f"Supplier requesting {hike_percentage or 'price'} increase for {product_name} with {len(justifications)} justification reasons."
        }
   
    def format_output(self, summary: Dict[str, Any]) -> str:
        """Format the output in a readable way"""
        output = []
        output.append("=== SUPPLIER NEGOTIATION EMAIL SUMMARY ===\n")
       
        for product in summary.get("products", []):
            output.append(f"Product: {product.get('product_name', 'N/A')}")
            output.append(f"Current Price: {product.get('current_price', 'N/A')}")
            output.append(f"Requested Price: {product.get('requested_price', 'N/A')}")
            output.append(f"Hike Percentage: {product.get('hike_percentage', 'N/A')}")
            output.append("\nJustification Table:")
            output.append("| Item | Category | Change |")
            output.append("|------|----------|--------|")
           
            for justification in product.get('justification_table', []):
                item = justification.get('item', 'N/A')
                category = justification.get('category', 'N/A')
                change = justification.get('change', 'N/A')
                output.append(f"| {item} | {category} | {change} |")
           
            output.append("")
       
        output.append(f"Overall Summary: {summary.get('overall_summary', 'N/A')}")
       
        return "\n".join(output)

# Usage Example
def main():
    """Example usage of the Supplier Email Agent"""
   
    # Initialize agent with Azure OpenAI credentials
    agent = SupplierEmailAgent(
        azure_endpoint="https://your-resource-name.openai.azure.com/",
        api_key="your-api-key",
        deployment_name="your-deployment-name"
    )
   
    # Example email content
    sample_email = """
    I am supplying xx shampoo at $100. This price needs a revision to meet the expenses.
    I would request to increase the price by 10% to $110. This hike is due to following reasons:
    1. Price of sugar and wheat has increased by 10% and 3% respectively.
    2. Electricity tariff has gone up by 3.5% and labour price has increased by 30%.
    """
   
    # Summarize the email
    summary = agent.summarize_email(sample_email)
   
    # Format and print the output
    formatted_output = agent.format_output(summary)
    print(formatted_output)
   
    return summary

if __name__ == "__main__":
    main()
Made with
"
 
