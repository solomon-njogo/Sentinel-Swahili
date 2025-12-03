"""
Model Evaluation Module
Computes evaluation metrics (perplexity, loss) and generates sample predictions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return float(torch.exp(torch.tensor(loss)).item())


def evaluate_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    test_dataset: Optional[Dataset] = None,
    max_length: int = 512,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Evaluate fine-tuned model on test dataset.
    
    Args:
        model_path: Path to fine-tuned model
        tokenizer_path: Path to tokenizer (if different from model_path)
        test_dataset: Test dataset for evaluation
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("=" * 80)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 80)
    
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Check if it's a PEFT model
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            logger.info("Loaded PEFT/LoRA model")
        except:
            model = base_model
            logger.info("Loaded base model (no PEFT adapters)")
        
        model.eval()
        
        if test_dataset is None:
            logger.warning("No test dataset provided. Skipping evaluation.")
            return {"success": False, "error": "No test dataset provided"}
        
        # Compute loss on test set
        logger.info(f"Evaluating on {len(test_dataset):,} test examples...")
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(test_dataset), batch_size):
                batch = test_dataset[i:i+batch_size]
                
                # Get input_ids
                input_ids_list = [ex["input_ids"] for ex in batch]
                
                # Pad sequences
                max_len = min(max([len(ids) for ids in input_ids_list]), max_length)
                padded_ids = []
                for ids in input_ids_list:
                    if len(ids) > max_len:
                        ids = ids[:max_len]
                    else:
                        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
                    padded_ids.append(ids)
                
                input_ids = torch.tensor(padded_ids).to(model.device)
                labels = input_ids.clone()
                
                # Forward pass
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item() * len(batch)
                total_tokens += input_ids.numel()
                num_batches += 1
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(test_dataset)} examples...")
        
        avg_loss = total_loss / len(test_dataset)
        perplexity = compute_perplexity(avg_loss)
        
        logger.info(f"Evaluation complete!")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Perplexity: {perplexity:.2f}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        
        return {
            "success": True,
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_examples": len(test_dataset)
        }
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def generate_sample_predictions(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> List[Dict[str, str]]:
    """
    Generate sample text predictions from the fine-tuned model.
    
    Args:
        model_path: Path to fine-tuned model
        tokenizer_path: Path to tokenizer
        prompts: List of prompt strings (default: Swahili examples)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        List of dictionaries with prompts and generated text
    """
    logger.info("Generating sample predictions...")
    
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    if prompts is None:
        prompts = [
            "Habari yako?",
            "Jina lako ni nani?",
            "Unaishi wapi?",
            "Unapenda kufanya nini?",
        ]
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Check if it's a PEFT model
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
        except:
            model = base_model
        
        model.eval()
        
        results = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                results.append({
                    "prompt": prompt,
                    "generated": generated_text
                })
                
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Generated: {generated_text}")
                logger.info("-" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        return []


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary with evaluation metrics
        output_path: Path to save JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any torch tensors to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to: {output_file}")

