import azure.functions as func
import datetime
import json
import logging
import os
import io
import re
import hashlib
import urllib.parse
from collections import defaultdict
from typing import List, Dict, Any, Optional
import tiktoken

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

app = func.FunctionApp()

def get_document_intelligence_client() -> DocumentIntelligenceClient:
    """
    Get Document Intelligence client using Managed Identity or Key-based authentication.
    Priority: Managed Identity > Key-based authentication
    """
    endpoint = os.environ.get("DOC_INTEL_ENDPOINT")
    if not endpoint:
        raise ValueError("DOC_INTEL_ENDPOINT environment variable is required")
    
    # Try Managed Identity first (recommended for Azure-hosted apps)
    try:
        credential = DefaultAzureCredential()
        return DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
    except Exception as e:
        logging.warning(f"Managed Identity authentication failed: {e}")
        
        # Fallback to Key-based authentication
        key = os.environ.get("DOCUMENT_INTELLIGENCE_KEY")
        if key:
            credential = AzureKeyCredential(key)
            return DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
        else:
            raise ValueError("Either Managed Identity must be configured or DOCUMENT_INTELLIGENCE_KEY environment variable is required")


def generate_base_document_id(source_url: Optional[str] = None, max_length: int = 1024) -> str:
    """
    Generate the base document ID from source URL (without page/chunk info).
    
    Azure AI Search document keys must:
    - Be 1024 characters or less
    - Contain only letters, numbers, dashes (-), underscores (_), and equal signs (=)
    - Be URL-safe for the Lookup API
    
    Args:
        source_url: Optional source URL of the document
        max_length: Maximum length for the document ID (default 1024)
    
    Returns:
        A valid base document ID that's deterministic and unique
    """
    if source_url:
        # Normalize the URL for consistent hashing
        parsed = urllib.parse.urlparse(source_url.lower().strip())
        normalized_url = urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),
            '',  # Remove params
            parsed.query,
            ''   # Remove fragment
        ))
        
        # Create hash of normalized URL (using full hash for zero collision risk)
        url_hash = hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()
        base_document_id = f"doc_{url_hash}"
    else:
        # Fallback for when no URL is provided
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_document_id = f"doc_{timestamp}"
    
    # Ensure it meets Azure AI Search requirements
    if len(base_document_id) > max_length:
        # Truncate while keeping the doc_ prefix
        base_document_id = base_document_id[:max_length]
    
    return base_document_id


def generate_chunk_id(source_url: Optional[str] = None, page_number: int = 1, max_length: int = 1024) -> str:
    """
    Generate a complete chunk ID with chunk/page identifier.
    
    Args:
        source_url: Optional source URL of the document
        page_number: Page/chunk number for multi-page documents
        max_length: Maximum length for the document ID (default 1024)
    
    Returns:
        A valid chunk ID that's deterministic and unique with chunk identifier
    """
    base_id = generate_base_document_id(source_url, max_length)
    chunk_id = f"_chunk_{page_number}"
    document_id = f"{base_id}{chunk_id}"
    
    # Ensure it meets Azure AI Search requirements
    if len(document_id) > max_length:
        # Truncate base while keeping the chunk identifier
        prefix_length = max_length - len(chunk_id)
        document_id = base_id[:prefix_length] + chunk_id
    
    return document_id


def count_tokens(text: str) -> int:
    """
    Count tokens for embedding models using tiktoken.
    Uses cl100k_base encoding which is compatible with all Azure OpenAI embedding models.
    """
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Token counting failed: {e}")
        # Fallback approximation: ~4 characters per token for English text
        return len(text) // 4


def group_pages_for_batch_embedding(
    pages: List[Dict[str, Any]], 
    max_tokens_per_batch: int = 7500,  # Buffer under 8,191 limit
    max_items_per_batch: int = 2000    # Buffer under 2048 limit
) -> List[Dict[str, Any]]:
    """
    Group pages into batches for Azure OpenAI embedding API.
    Adds 'batch_index' field to each page.
    
    Args:
        pages: List of page dictionaries with token_count
        max_tokens_per_batch: Maximum tokens per batch (default: 7500)
        max_items_per_batch: Maximum items per batch (default: 2000)
    
    Returns:
        Updated pages list with 'batch_index' field added to each page
    """
    current_batch = 0
    current_tokens = 0
    current_items = 0
    
    for page in pages:
        page_tokens = page['token_count']
        
        # Check if adding this page would exceed limits
        if (current_tokens + page_tokens > max_tokens_per_batch or 
            current_items >= max_items_per_batch):
            # Start new batch
            current_batch += 1
            current_tokens = 0
            current_items = 0
        
        # Add page to current batch
        page['batch_index'] = current_batch
        current_tokens += page_tokens
        current_items += 1
    
    return pages


def split_markdown_by_pages(full_markdown: str, source_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Split the full markdown content by page breaks using HTML comments.
    Document Intelligence includes <!-- PageBreak --> comments to mark page boundaries.
    
    Args:
        full_markdown: The full markdown content with page break comments
        source_url: Optional source URL for generating document IDs
        
    Returns:
        List of dictionaries with page_number, markdown_content, token_count, and document_id
    """
    # Split by PageBreak comments
    page_sections = full_markdown.split('<!-- PageBreak -->')
    
    pages = []
    
    for i, section in enumerate(page_sections):
        if not section.strip():
            continue
            
        # Extract page number from PageNumber comment if present
        page_number = i + 1  # Default fallback
        page_number_match = re.search(r'<!-- PageNumber="(\d+)" -->', section)
        if page_number_match:
            page_number = int(page_number_match.group(1))
        
        # Clean up the content by removing page metadata comments
        content = section.strip()
        
        # Remove page metadata comments from content while preserving other HTML comments
        content = re.sub(r'<!-- PageNumber="[^"]*" -->\s*', '', content)
        content = re.sub(r'<!-- PageHeader="[^"]*" -->\s*', '', content)  
        content = re.sub(r'<!-- PageFooter="[^"]*" -->\s*', '', content)
        content = content.strip()
        
        if content:  # Only add non-empty pages
            pages.append({
                "document_id": generate_chunk_id(source_url, page_number),
                "page_number": page_number,
                "markdown_content": content,
                "token_count": count_tokens(content)
            })
    
    # Group pages into batches for embedding API
    pages = group_pages_for_batch_embedding(pages)
    
    return pages


@app.function_name("generate_document_id_endpoint")
@app.route(route="generate-document-id", methods=["POST"])
def generate_document_id_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function endpoint to generate base document IDs for Azure AI Search.
    
    Expected input (POST with JSON body):
    - source_url: URL of the source document
    
    Returns: JSON object with base_document_id
    """
    try:
        logging.info("Processing document ID generation request")
        
        # Get JSON body
        try:
            req_body = req.get_json()
            if not req_body:
                return func.HttpResponse(
                    json.dumps({"error": "Request body is required"}),
                    status_code=400,
                    mimetype="application/json"
                )
            source_url = req_body.get('source_url')
            logging.info(f"Raw source_url from JSON body: '{source_url}'")
            if not source_url:
                return func.HttpResponse(
                    json.dumps({"error": "source_url is required"}),
                    status_code=400,
                    mimetype="application/json"
                )
        except Exception:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Generate the base document ID
        base_document_id = generate_base_document_id(source_url)
        
        response_data = {
            "base_document_id": base_document_id,
            "source_url": source_url
        }
        
        logging.info(f"Generated base document ID: {base_document_id}")
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error generating document ID: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )


@app.function_name("process_document_to_markdown")
@app.route(route="process-document", methods=["POST"])
def process_document_to_markdown(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function that processes document binary data and returns page-split markdown.
    
    Expected input: 
    - Raw binary file data in request body
    - Optional query parameter 'source_url' for document ID generation
    
    Returns: JSON array of objects with document_id, page_number, markdown_content, and token_count
    """
    try:
        logging.info("Processing document conversion request")
        
        # Get optional source URL parameter
        source_url = req.params.get('source_url')
        logging.info(f"Raw source_url parameter received: '{source_url}'")
        if source_url:
            logging.info(f"Processing document with source URL: {source_url}")
        else:
            logging.info("No source_url parameter provided")
        
        # Validate request
        if not req.get_body():
            return func.HttpResponse(
                json.dumps({"error": "Request body is empty. Please provide document binary data."}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Get document binary data
        document_bytes = req.get_body()
        logging.info(f"Received document of size: {len(document_bytes)} bytes")
        
        # Get Document Intelligence client
        client = get_document_intelligence_client()
        
        # Analyze document with Layout model and markdown output
        logging.info("Starting document analysis with Document Intelligence")
        
        # Create document stream from binary data
        document_stream = io.BytesIO(document_bytes)
        
        # Analyze document using Layout model with markdown output
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=document_stream,
            output_content_format="markdown"
        )
        
        # Wait for completion
        result = poller.result()
        logging.info("Document analysis completed successfully")
        
        # Get the full markdown content
        full_markdown = result.content if hasattr(result, 'content') else ""
        
        # Split markdown content by page breaks (using HTML comments)
        pages = split_markdown_by_pages(full_markdown, source_url)
        
        response_data = pages
        
        logging.info(f"Successfully processed document into {len(response_data)} pages")
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except ValueError as ve:
        logging.error(f"Configuration error: {ve}")
        return func.HttpResponse(
            json.dumps({"error": f"Configuration error: {str(ve)}"}),
            status_code=500,
            mimetype="application/json"
        )
        
    except AzureError as ae:
        logging.error(f"Azure Document Intelligence error: {ae}")
        return func.HttpResponse(
            json.dumps({"error": f"Document processing failed: {str(ae)}"}),
            status_code=502,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Unexpected error processing document: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )