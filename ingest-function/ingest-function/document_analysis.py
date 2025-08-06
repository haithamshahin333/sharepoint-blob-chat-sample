"""
Document analysis utilities using LangChain and Azure OpenAI.
Provides document-level analysis including summarization, topic extraction, and document type detection.

Enhanced with structured outputs using Pydantic schemas for reliable response parsing.
Features:
- Single-shot analysis for documents < 100K tokens using structured outputs
- Map-reduce approach for larger documents 
- Pydantic models ensure type safety and validation
- Token-aware routing for optimal performance
"""

import os
import logging
from typing import Dict, List, Any, Optional
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# Pydantic for structured outputs
from pydantic import BaseModel, Field

# Token counting (reuse from main module)
import tiktoken


# Pydantic models for structured outputs
class DocumentAnalysis(BaseModel):
    """Structured document analysis result."""
    summary: str = Field(
        description="Extractive summary using exact sentences and phrases from the document"
    )
    key_topics: List[str] = Field(
        description="List of 10 most important topics, themes, or subject areas covered",
        max_items=10
    )
    document_type: str = Field(
        description="Document type classification: Report, Manual, Policy, Presentation, Legal, Financial, Academic, Marketing, Technical, or Other"
    )
    published_date: str = Field(
        description="Published date in MM-DD-YYYY format. If no date can be found, return '00-00-0000'"
    )


class TopicsList(BaseModel):
    """Structured list of key topics."""
    topics: List[str] = Field(
        description="List of key topics, themes, or subject areas",
        max_items=10
    )


class DocumentType(BaseModel):
    """Document type classification."""
    document_type: str = Field(
        description="Document type: Report, Manual, Policy, Presentation, Legal, Financial, Academic, Marketing, Technical, or Other"
    )


class PublishedDate(BaseModel):
    """Document published date extraction."""
    published_date: str = Field(
        description="Published date in MM-DD-YYYY format. If no date can be found, return '00-00-0000'"
    )


def get_azure_openai_client() -> AzureChatOpenAI:
    """
    Create Azure OpenAI client using managed identity authentication.
    
    Returns:
        AzureChatOpenAI: Configured client for Azure OpenAI
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") 
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    if not deployment_name:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is required")
    
    # Create token provider using managed identity
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 
        "https://cognitiveservices.azure.com/.default"
    )
    
    # Create Azure OpenAI client
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment_name,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
        temperature=0.3,  # Low temperature for consistent analysis
        max_tokens=4000   # Allow for detailed summaries
    )


def count_tokens_for_text(text: str) -> int:
    """
    Count tokens using tiktoken (same encoding as main module).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Token count
    """
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Token counting failed: {e}")
        # Fallback approximation: ~4 characters per token for English text
        return len(text) // 4


def generate_single_shot_analysis(content: str, llm: AzureChatOpenAI) -> Dict[str, Any]:
    """
    Generate complete document analysis in one shot for content under 100K tokens.
    Uses structured outputs with Pydantic schemas for reliable parsing.
    
    Args:
        content: Full document content
        llm: Azure OpenAI client
        
    Returns:
        Dictionary with summary, key_topics, and document_type
    """
    try:
        # Create structured LLM with Pydantic schema
        structured_llm = llm.with_structured_output(DocumentAnalysis)
        
        # Create the prompt using ChatPromptTemplate for better control
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst. Your task is to analyze documents and provide structured analysis.

For the summary: Select and organize the most important sentences and phrases directly from the text. Do not paraphrase - use exact wording from the document. Focus on key findings, conclusions, and important statements.

For key topics: Extract the 10 most important topics, themes, or subject areas covered in the document.

For document type: Classify from these categories: Report, Manual, Policy, Presentation, Legal, Financial, Academic, Marketing, Technical, Other.

For published date: Look for any dates that indicate when the document was published, created, or released. Return in MM-DD-YYYY format. If no date can be found, return '00-00-0000'."""),
            ("human", "Please analyze the following document:\n\n{content}")
        ])
        
        # Create the chain and invoke
        chain = prompt | structured_llm
        result = chain.invoke({"content": content})
        
        # Convert Pydantic model to dictionary
        return {
            "summary": result.summary,
            "key_topics": result.key_topics,
            "document_type": result.document_type,
            "published_date": result.published_date
        }
        
    except Exception as e:
        logging.error(f"Structured output analysis failed: {e}")
        # Return minimal error response
        return {
            "summary": "Analysis failed",
            "key_topics": [],
            "document_type": "Unknown",
            "published_date": "00-00-0000"
        }


def generate_map_reduce_summary(content: str, llm: AzureChatOpenAI, total_tokens: int) -> str:
    """
    Generate summary using map-reduce approach for large documents (>= 100K tokens).
    Uses LangChain's RecursiveCharacterTextSplitter for intelligent token-based chunking.
    
    Args:
        content: Full document content  
        llm: Azure OpenAI client
        total_tokens: Total token count for the document
        
    Returns:
        Generated summary
    """
    # Decide whether to chunk the content or use it as-is
    if total_tokens > 100000:
        # Use LangChain's tiktoken-based text splitter for intelligent chunking
        logging.info(f"Splitting {total_tokens} tokens into chunks using LangChain RecursiveCharacterTextSplitter")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",  # Same encoding as our token counting
            chunk_size=100000,  # 100K tokens per chunk
            chunk_overlap=5000,  # 5K token overlap for context preservation
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]  # Intelligent separators
        )
        content_chunks = text_splitter.split_text(content)
    else:
        # Use content as single chunk for smaller documents
        content_chunks = [content]
    
    # Map phase: Summarize each chunk using modern pipe operator
    map_template = """Extract the most important sentences and key phrases directly from this document section. Use the exact wording from the text - do not paraphrase or rewrite.

Focus on:
- Key statements, conclusions, or findings
- Important facts, data, or decisions
- Critical information or main points

Use only the original text from the section.

Document section:
{text}

Extracted Key Content:"""
    
    map_prompt = PromptTemplate(template=map_template, input_variables=["text"])
    parser = StrOutputParser()
    map_chain = map_prompt | llm | parser
    
    # Generate chunk summaries
    chunk_summaries = []
    for i, chunk in enumerate(content_chunks, 1):
        try:
            summary = map_chain.invoke({"text": chunk})
            chunk_summaries.append(f"Section {i}: {summary.strip()}")
        except Exception as e:
            logging.warning(f"Failed to summarize chunk {i}: {e}")
            # Include chunk without summary as fallback
            chunk_summaries.append(f"Section {i}: [Summary generation failed]")
    
    # Reduce phase: Combine chunk summaries into final summary using modern pipe operator
    reduce_template = """Based on the following extracted content from each document section, create a comprehensive extractive summary by organizing the most important extracted sentences and phrases.

Instructions:
- Use the exact wording from the section extracts provided
- Organize the content to show overall document flow and key themes
- Maintain original terminology and phrasing
- Focus on the most critical extracted information across all sections
- Do not add new interpretations - only reorganize the extracted content

Section extracts:
{text}

Final Extractive Summary:"""
    
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"])
    reduce_chain = reduce_prompt | llm | parser
    
    # Combine all chunk summaries
    combined_summaries = "\n\n".join(chunk_summaries)
    final_summary = reduce_chain.invoke({"text": combined_summaries})
    
    return final_summary.strip()


def extract_key_topics(content: str, llm: AzureChatOpenAI, max_topics: int = 10, summary: Optional[str] = None) -> List[str]:
    """
    Extract key topics and themes from document content using structured outputs.
    
    Args:
        content: Document content (will be truncated if too long)
        llm: Azure OpenAI client  
        max_topics: Maximum number of topics to extract
        summary: Optional generated summary to use for large documents
        
    Returns:
        List of key topics
    """
    # Check token count and decide what content to use
    content_tokens = count_tokens_for_text(content)
    
    if content_tokens > 100000 and summary:
        # Use the generated summary for large documents
        logging.info(f"Using generated summary for topic extraction (original content: {content_tokens} tokens)")
        analysis_content = summary
        content_source = "summary"
    else:
        # Use full content for smaller documents (should only happen if called independently)
        analysis_content = content
        content_source = "full content"
    
    try:
        # Create structured LLM with Pydantic schema
        structured_llm = llm.with_structured_output(TopicsList)
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert at extracting key topics from documents. Analyze the following document {content_source} and extract the {max_topics} most important topics, themes, or subject areas covered."),
            ("human", "Document {content_source}:\n\n{content}")
        ])
        
        # Create the chain and invoke
        chain = prompt | structured_llm
        result = chain.invoke({"content": analysis_content, "content_source": content_source})
        
        # Return the topics list, ensuring we don't exceed max_topics
        return result.topics[:max_topics]
        
    except Exception as e:
        logging.error(f"Structured topic extraction failed: {e}")
        return []


def detect_document_type(content: str, llm: AzureChatOpenAI) -> str:
    """
    Detect the type/category of the document using structured outputs.
    
    Args:
        content: Document content (truncated if too long)
        llm: Azure OpenAI client
        
    Returns:
        Document type classification
    """
    # Use first portion of document for type detection
    content_tokens = count_tokens_for_text(content)
    if content_tokens > 20000:  # Limit for type detection  
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(content)
        content = tokenizer.decode(tokens[:20000])
    
    try:
        # Create structured LLM with Pydantic schema
        structured_llm = llm.with_structured_output(DocumentType)
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document classifier. Analyze the document excerpt and classify its type.

Choose the most appropriate category:
- Report (research, business, technical, etc.)
- Manual (user guide, instructions, procedures)
- Policy (guidelines, regulations, standards) 
- Presentation (slides, training materials)
- Legal (contract, agreement, legal document)
- Financial (statements, budgets, financial reports)
- Academic (research paper, thesis, educational content)
- Marketing (brochures, proposals, marketing materials)
- Technical (specifications, documentation, technical guides)
- Other (if none of the above fit well)"""),
            ("human", "Document excerpt:\n\n{content}")
        ])
        
        # Create the chain and invoke
        chain = prompt | structured_llm
        result = chain.invoke({"content": content})
        
        return result.document_type
        
    except Exception as e:
        logging.error(f"Structured document type detection failed: {e}")
        return "Unknown"


def detect_published_date(content: str, llm: AzureChatOpenAI) -> str:
    """
    Detect the published date of the document using structured outputs.
    
    Args:
        content: Document content (truncated if too long)
        llm: Azure OpenAI client
        
    Returns:
        Published date in MM-DD-YYYY format, or '00-00-0000' if not found
    """
    # Use first portion of document for date detection
    content_tokens = count_tokens_for_text(content)
    if content_tokens > 20000:  # Limit for date detection  
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(content)
        content = tokenizer.decode(tokens[:20000])
    
    try:
        # Create structured LLM with Pydantic schema
        structured_llm = llm.with_structured_output(PublishedDate)
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting publication dates from documents. Look for any dates that indicate when the document was published, created, or released.

Common locations for publication dates:
- Document headers or title pages
- Copyright notices
- Article dates
- Version dates
- Release dates
- Creation dates

Return the date in MM-DD-YYYY format. If no publication date can be found, return '00-00-0000'."""),
            ("human", "Document excerpt:\n\n{content}")
        ])
        
        # Create the chain and invoke
        chain = prompt | structured_llm
        result = chain.invoke({"content": content})
        
        return result.published_date
        
    except Exception as e:
        logging.error(f"Structured published date detection failed: {e}")
        return "00-00-0000"


def analyze_document_content(full_markdown: str, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze document content and generate document-level attributes.
    
    Args:
        full_markdown: Complete markdown content from Document Intelligence
        pages: List of page objects with token counts
        
    Returns:
        Document-level analysis including summary, topics, etc.
        Returns error info if analysis fails but doesn't raise exceptions.
    """
    try:
        # Calculate total metrics
        total_pages = len(pages)
        total_tokens = sum(page["token_count"] for page in pages)
        
        # Get Azure OpenAI client
        llm = get_azure_openai_client()
        
        # Determine analysis strategy based on token count
        if total_tokens < 100000:
            logging.info(f"Using single-shot analysis for {total_tokens} tokens")
            analysis_result = generate_single_shot_analysis(full_markdown, llm)
            summary = analysis_result["summary"]
            key_topics = analysis_result["key_topics"]
            document_type = analysis_result["document_type"]
            published_date = analysis_result["published_date"]
        else:
            logging.info(f"Using map-reduce approach for {total_tokens} tokens with intelligent chunking")
            summary = generate_map_reduce_summary(full_markdown, llm, total_tokens)
            # For large documents, extract topics from the generated summary
            key_topics = extract_key_topics(full_markdown, llm, summary=summary)
            document_type = detect_document_type(full_markdown, llm)
            published_date = detect_published_date(full_markdown, llm)
        
        return {
            "total_pages": total_pages,
            "total_tokens": total_tokens,
            "summary": summary,
            "key_topics": key_topics,
            "document_type": document_type,
            "published_date": published_date,
            "analysis_status": "success"
        }
        
    except Exception as e:
        logging.error(f"Document analysis failed: {e}")
        # Return basic metrics with error info
        return {
            "total_pages": len(pages),
            "total_tokens": sum(page.get("token_count", 0) for page in pages),
            "summary": None,
            "key_topics": [],
            "document_type": "Unknown",
            "published_date": "00-00-0000",
            "analysis_status": "failed",
            "error": str(e)
        }
