"""
Ensemble validation module for LLM outputs
"""
from typing import List, Dict, Any
from ..utils.config import (
    LLM_MODEL_1, LLM_MODEL_2, ENSEMBLE_AGREEMENT_THRESHOLD
)
from ..utils.logger import logger
from .query_langchain import HuggingFaceLLM


class EnsembleValidator:
    """Class for ensemble validation of LLM outputs"""
    
    def __init__(self, model1_name: str = None, model2_name: str = None):
        self.model1_name = model1_name or LLM_MODEL_1
        self.model2_name = model2_name or LLM_MODEL_2
        self.agreement_threshold = ENSEMBLE_AGREEMENT_THRESHOLD
        
        self.llm1 = None
        self.llm2 = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize both LLM models"""
        try:
            logger.info(f"Initializing ensemble models: {self.model1_name} and {self.model2_name}")
            self.llm1 = HuggingFaceLLM(self.model1_name)
            logger.info(f"Model 1 loaded: {self.model1_name}")
        except Exception as e:
            logger.warning(f"Could not load model 1 ({self.model1_name}): {str(e)}")
        
        try:
            self.llm2 = HuggingFaceLLM(self.model2_name)
            logger.info(f"Model 2 loaded: {self.model2_name}")
        except Exception as e:
            logger.warning(f"Could not load model 2 ({self.model2_name}): {str(e)}")
        
        if not self.llm1 and not self.llm2:
            raise RuntimeError("Could not load any ensemble models")
    
    def generate_parallel(self, prompt: str) -> Dict[str, Any]:
        """
        Generate outputs from both models in parallel
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dictionary with outputs from both models
        """
        results = {
            "prompt": prompt,
            "model1_output": None,
            "model2_output": None,
            "agreement": False
        }
        
        if self.llm1:
            try:
                results["model1_output"] = self.llm1._call(prompt)
            except Exception as e:
                logger.error(f"Error with model 1: {str(e)}")
        
        if self.llm2:
            try:
                results["model2_output"] = self.llm2._call(prompt)
            except Exception as e:
                logger.error(f"Error with model 2: {str(e)}")
        
        return results
    
    def compare_outputs(self, output1: str, output2: str, tolerance: float = 0.8) -> bool:
        """
        Compare two outputs for agreement
        
        Args:
            output1: First model output
            output2: Second model output
            tolerance: Similarity threshold (0-1)
            
        Returns:
            True if outputs agree (similar enough)
        """
        if not output1 or not output2:
            return False
        
        # Normalize outputs
        output1_norm = output1.lower().strip()
        output2_norm = output2.lower().strip()
        
        # Exact match
        if output1_norm == output2_norm:
            return True
        
        # Calculate simple similarity (Jaccard similarity on words)
        words1 = set(output1_norm.split())
        words2 = set(output2_norm.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        
        return similarity >= tolerance
    
    def validate_extraction(
        self,
        prompt: str,
        attribute_name: str
    ) -> Dict[str, Any]:
        """
        Validate attribute extraction using ensemble
        
        Args:
            prompt: Extraction prompt
            attribute_name: Name of attribute being extracted
            
        Returns:
            Dictionary with validated extraction result
        """
        logger.info(f"Validating extraction for attribute: {attribute_name}")
        
        # Generate from both models
        results = self.generate_parallel(prompt)
        
        output1 = results["model1_output"]
        output2 = results["model2_output"]
        
        # Check agreement
        if output1 and output2:
            agreement = self.compare_outputs(output1, output2)
            results["agreement"] = agreement
            
            if agreement:
                # Use the first output as the agreed value
                results["validated_value"] = output1
                results["confidence"] = "high"
                logger.info(f"Ensemble agreement for {attribute_name}: {output1[:50]}...")
            else:
                # Disagreement - use first output but flag it
                results["validated_value"] = output1
                results["confidence"] = "low"
                results["disagreement"] = {
                    "model1": output1,
                    "model2": output2
                }
                logger.warning(f"Ensemble disagreement for {attribute_name}")
        elif output1:
            # Only model 1 available
            results["validated_value"] = output1
            results["agreement"] = False
            results["confidence"] = "medium"
            logger.warning(f"Only model 1 available for {attribute_name}")
        elif output2:
            # Only model 2 available
            results["validated_value"] = output2
            results["agreement"] = False
            results["confidence"] = "medium"
            logger.warning(f"Only model 2 available for {attribute_name}")
        else:
            results["validated_value"] = None
            results["agreement"] = False
            results["confidence"] = "none"
            logger.error(f"No model outputs available for {attribute_name}")
        
        results["attribute_name"] = attribute_name
        return results
    
    def extract_with_validation(
        self,
        doc_text: str,
        attribute_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract attributes with ensemble validation
        
        Args:
            doc_text: Document text
            attribute_schema: Dictionary mapping attribute names to descriptions
            
        Returns:
            Dictionary with validated extractions
        """
        all_extractions = {}
        all_validations = {}
        
        # Extract each attribute with validation
        for attr_name, attr_desc in attribute_schema.items():
            prompt = f"""Extract the value for "{attr_name}" from the following legal document text. 
Description: {attr_desc}

Document text:
{doc_text}

Extract only the value for {attr_name}. If not found, respond with "N/A":"""
            
            validation_result = self.validate_extraction(prompt, attr_name)
            all_validations[attr_name] = validation_result
            
            if validation_result["validated_value"]:
                all_extractions[attr_name] = validation_result["validated_value"]
            else:
                all_extractions[attr_name] = "N/A"
        
        # Calculate overall agreement rate
        agreements = sum(1 for v in all_validations.values() if v.get("agreement", False))
        total = len(all_validations)
        agreement_rate = agreements / total if total > 0 else 0
        
        result = {
            "extractions": all_extractions,
            "validations": all_validations,
            "agreement_rate": agreement_rate,
            "num_attributes": total,
            "num_agreements": agreements
        }
        
        logger.info(f"Extracted {total} attributes with {agreements} agreements ({agreement_rate*100:.1f}%)")
        return result

