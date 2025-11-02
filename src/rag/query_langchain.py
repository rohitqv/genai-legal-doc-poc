"""
RAG pipeline using LangChain
"""
from typing import List, Dict, Optional
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from ..utils.config import LLM_MODEL_1, LLM_TEMPERATURE, LLM_MAX_TOKENS, TOP_K_CHUNKS
from ..utils.logger import logger


class HuggingFaceLLM(LLM):
    """Custom LangChain LLM wrapper for Hugging Face models"""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = 0 if torch.cuda.is_available() else -1
        
        logger.info(f"Loading LLM: {model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 0 else torch.float32,
            device_map="auto" if self.device == 0 else None
        )
        
        if self.device == -1:
            self.model = self.model.to("cpu")
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        return generated_text.strip()


class RAGPipeline:
    """RAG pipeline using LangChain and FAISS"""
    
    def __init__(
        self,
        embedding_manager,
        faiss_index,
        metadata: List[Dict],
        llm_model: str = None,
        top_k: int = None
    ):
        self.embedding_manager = embedding_manager
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.top_k = top_k or TOP_K_CHUNKS
        
        # Initialize LLM
        model_name = llm_model or LLM_MODEL_1
        try:
            self.llm = HuggingFaceLLM(
                model_name=model_name,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            logger.info(f"RAG pipeline initialized with LLM: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load LLM {model_name}: {str(e)}. Using fallback.")
            self.llm = None
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        top_k = top_k or self.top_k
        
        # Search FAISS index
        results = self.embedding_manager.search(
            query=query,
            index=self.faiss_index,
            metadata=self.metadata,
            top_k=top_k
        )
        
        # Get full text for each result
        retrieved_chunks = []
        for result in results:
            chunk_meta = result["metadata"]
            retrieved_chunks.append({
                "text": chunk_meta.get("text", ""),
                "score": result["score"],
                "doc_id": chunk_meta.get("doc_id", ""),
                "chunk_id": chunk_meta.get("chunk_id", ""),
                "chunk_index": chunk_meta.get("chunk_index", -1)
            })
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
        return retrieved_chunks
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        prompt_template: str = None
    ) -> str:
        """
        Generate answer using retrieved context
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            prompt_template: Custom prompt template
            
        Returns:
            Generated answer
        """
        if not self.llm:
            return "LLM not available. Please check model configuration."
        
        # Build context from chunks
        context = "\n\n".join([
            f"[Document {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Default prompt template
        if prompt_template is None:
            prompt_template = """You are a legal document analysis assistant. Use the following context to answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        formatted_prompt = prompt.format(context=context, question=query)
        
        try:
            answer = self.llm._call(formatted_prompt)
            logger.info("Generated answer using RAG pipeline")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def query(self, query: str, extract_attributes: bool = False) -> Dict:
        """
        Complete RAG query: retrieve + generate
        
        Args:
            query: User query
            extract_attributes: Whether to extract structured attributes
            
        Returns:
            Dictionary with answer, retrieved chunks, and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query)
        
        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "retrieved_chunks": [],
                "metadata": {}
            }
        
        # Generate answer
        answer = self.generate(query, retrieved_chunks)
        
        result = {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "num_chunks": len(retrieved_chunks),
            "metadata": {
                "top_score": retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
            }
        }
        
        return result
    
    def extract_attributes(
        self,
        doc_text: str,
        attribute_schema: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Extract structured attributes from document text
        
        Args:
            doc_text: Document text
            attribute_schema: Dictionary mapping attribute names to descriptions
            
        Returns:
            Dictionary of extracted attributes
        """
        if not self.llm:
            return {}
        
        # Build extraction prompt
        attributes_list = "\n".join([
            f"- {name}: {desc}"
            for name, desc in attribute_schema.items()
        ])
        
        prompt = f"""Extract the following attributes from the legal document text below. Return each attribute value on a new line in the format "AttributeName: Value". If an attribute is not found, use "N/A".

Attributes to extract:
{attributes_list}

Document text:
{doc_text}

Extracted attributes:"""
        
        try:
            response = self.llm._call(prompt)
            
            # Parse response into dictionary
            extracted = {}
            for line in response.split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        attr_name = parts[0].strip()
                        attr_value = parts[1].strip()
                        extracted[attr_name] = attr_value
            
            logger.info(f"Extracted {len(extracted)} attributes")
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting attributes: {str(e)}")
            return {}

