import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
import json
import numpy as np
import random
import os
from utils import evaluate_nlg, evaluate_with_llm  # Assuming these are your evaluation functions

def main():
    parser = argparse.ArgumentParser(description="Run inference with gpt2-funetuned-eli5 model")
    parser.add_argument("--model_id", type=str, default="ashaduzzaman/gpt2-funetuned-eli5", 
                        help="HuggingFace model ID")
    parser.add_argument("--data_path", type=str, default="val_json_path", 
                        help="Path to evaluation data JSON")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_examples", type=int, default=None, 
                        help="Number of examples to evaluate (None for all)")
    parser.add_argument("--output_file", type=str, default="eli5_results.json", 
                        help="File to save results")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer examples")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    print("Model loaded successfully")

    # Load validation data
    print(f"Loading data from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        val_data = json.load(f)

    # Limit number of examples if specified
    if args.debug:
        num_examples = min(10, len(val_data))
        val_data = val_data[:num_examples]
        print(f"Debug mode: using only {num_examples} examples")
    elif args.num_examples is not None:
        val_data = val_data[:args.num_examples]
        print(f"Using {args.num_examples} examples")
    else:
        print(f"Using all {len(val_data)} examples")

    # Extract questions and reference answers
    question_val = [d["question"] for d in val_data]
    answers_val = [d["answer"].strip() for d in val_data]

    # Prepare for evaluation
    predictions = []
    references = answers_val

    # Create progress bar
    pbar = tqdm(total=len(question_val), desc="Generating Responses", unit="example")
    
    model.eval()
    
    with torch.no_grad():
        for idx, question in enumerate(question_val):
            # Prepare input
            input_text = question
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            
            # Generate response
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_output = generated_text.replace(question, "").strip()
            
            # Store for metrics calculation
            predictions.append(answer_output)
            
            # Print samples
            if idx < 5:
                print(f"\nExample {idx + 1}:")
                print(f"Question: {question}")
                print(f"Ground truth: {answers_val[idx]}")
                print(f"Generated: {answer_output}")
            
            pbar.update(1)
    
    pbar.close()
    
    # Calculate metrics
    print("\n===== Evaluation Results =====")
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    print(f"Exact Match Accuracy: {correct}/{len(predictions)} = {correct/len(predictions):.4f}")
    
    # Calculate NLG metrics
    print("\nCalculating NLG metrics...")
    nlg_metrics = evaluate_nlg(predictions, references)
    print(f"ROUGE-1: {nlg_metrics['rouge']['rouge1']:.4f}")
    print(f"ROUGE-2: {nlg_metrics['rouge']['rouge2']:.4f}")
    print(f"ROUGE-L: {nlg_metrics['rouge']['rougeL']:.4f}")
    print(f"BLEU: {nlg_metrics['bleu']:.4f}")
    
    # Optional: Evaluate with LLM if available
    try:
        print("\nEvaluating with LLM as judge (sample of 50 examples)...")
        eval_indices = random.sample(range(len(question_val)), min(50, len(question_val)))
        
        llm_metrics = {
            "accuracy_scores": [],
            "clarity_scores": [],
            "completeness_scores": [],
            "conciseness_scores": [],
            "engagement_scores": []
        }
        
        for idx in tqdm(eval_indices, desc="LLM Evaluation"):
            question = question_val[idx]
            explanation = predictions[idx]
            eval_result = evaluate_with_llm(question, explanation)
            
            if eval_result is None:
                continue
                
            for key in llm_metrics:
                score_key = key.replace('_scores', '_score')
                if score_key in eval_result:
                    llm_metrics[key].append(eval_result[score_key])
        
        # Calculate and display average scores
        for key, scores in llm_metrics.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"{key.replace('_scores', '')}: {avg_score:.4f}")
    except Exception as e:
        print(f"LLM evaluation failed: {e}")
    
    # Save results to file
    results = {
        "model": args.model_id,
        "exact_match": correct/len(predictions),
        "rouge1": nlg_metrics['rouge']['rouge1'],
        "rouge2": nlg_metrics['rouge']['rouge2'],
        "rougeL": nlg_metrics['rouge']['rougeL'],
        "bleu": nlg_metrics['bleu'],
        "samples": [
            {"question": q, "reference": r, "prediction": p}
            for q, r, p in zip(question_val[:10], references[:10], predictions[:10])
        ]
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()