import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import yaml
import json
import argparse
from tqdm import tqdm
import numpy as np

from utils import Config, set_seed, evaluate_nlg, evaluate_with_llm

def main():
    print(f"Available GPUs: {torch.cuda.device_count()}")

    parser = argparse.ArgumentParser(description="Vanilla GPT-2 inference")
    parser.add_argument("config_file", help="Path to the config YAML file")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    
    # Force evaluation-only mode
    config_dict["only_eval"] = True
    
    print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    
    # Set device - use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vanilla GPT-2 model and tokenizer
    print(f"Loading pretrained model: {configs.model_id}")
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    
    # Make sure padding is properly configured
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    model = model.to(device)
    
    # Apply bfloat16 if configured
    if configs.bf16 and device.type == 'cuda':
        model.to(torch.bfloat16)
    
    print("Model loaded successfully!")

    # Load validation data
    train_json_path = os.path.join('data', "eli5_train_119k.json")
    val_json_path = os.path.join('data', "eli5_val_1k.json")
    test_json_path = os.path.join('data', "eli5_test_5k.json")
    print(f"Loading validation data from: {val_json_path}")
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)

    # Prepare ground truth data for evaluation
    question_val = [d["question"] for d in val_data]
    answers_val = [d["answer"].strip() for d in val_data]
    
    # Limit to fewer examples if in debug mode
    if configs.debug:
        num_examples = min(10, len(question_val))
        question_val = question_val[:num_examples]
        answers_val = answers_val[:num_examples]
        print(f"Debug mode: using only {num_examples} examples")
    
    print(f"Loaded {len(question_val)} validation examples")

    # Define token generation length
    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 256
    
    print(f"Max new tokens for generation: {max_new_tokens}")

    # Evaluation loop
    predictions = []
    references = []
    
    # Create progress bar
    pbar = tqdm(total=len(question_val), desc="Evaluating GPT-2", unit="example")
    
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for idx, question in enumerate(question_val):
            # Prepare input
            input_text = question + "\n"
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            
            # Generate response
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Use greedy decoding for deterministic results
            )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_output = generated_text.replace(question, "").strip()
            
            # Store for metrics calculation
            predictions.append(answer_output)
            references.append(answers_val[idx])
            
            # Check if correct
            if answer_output == answers_val[idx]:
                correct += 1
            
            # Print samples
            if idx < 5:
                print(f"\nExample {idx + 1}:")
                print(f"Question: {question}")
                print(f"Ground truth: {answers_val[idx]}")
                print(f"Generated: {answer_output}")
            
            pbar.update(1)
            pbar.set_description(f"Accuracy: {correct/(idx+1):.4f}")
    
    pbar.close()
    
    # Calculate metrics
    print("\n===== Evaluation Results =====")
    print(f"Exact Match Accuracy: {correct}/{len(predictions)} = {correct/len(predictions):.4f}")
    
    # Calculate NLG metrics
    print("\nCalculating NLG metrics...")
    nlg_metrics = evaluate_nlg(predictions, references)
    print(f"ROUGE-1: {nlg_metrics['rouge']['rouge1']:.4f}")
    print(f"ROUGE-2: {nlg_metrics['rouge']['rouge2']:.4f}")
    print(f"ROUGE-L: {nlg_metrics['rouge']['rougeL']:.4f}")
    print(f"BLEU: {nlg_metrics['bleu']:.4f}")
    
    # Optional: Evaluate with LLM if needed
    if configs.debug:
        print("Skipping LLM evaluation in debug mode")
    else:
        print("\nEvaluating with LLM as judge (sample of 50 examples)...")
        eval_indices = random.sample(range(len(question_val)), min(50, len(question_val)))
        accuracy_scores = []
        clarity_scores = []
        completeness_scores = []
        conciseness_scores = []
        engagement_scores = []
        
        for idx in tqdm(eval_indices, desc="LLM Evaluation"):
            question = question_val[idx]
            explanation = predictions[idx]
            eval_result = evaluate_with_llm(question, explanation)
            if eval_result is None:
                continue
                
            accuracy_scores.append(eval_result["accuracy_score"])
            clarity_scores.append(eval_result["clarity_score"])
            completeness_scores.append(eval_result["completeness_score"])
            conciseness_scores.append(eval_result["conciseness_score"])
            engagement_scores.append(eval_result["engagement_score"])
        
        try:
            avg_accuracy = np.mean(accuracy_scores)
            avg_clarity = np.mean(clarity_scores)
            avg_completeness = np.mean(completeness_scores)
            avg_conciseness = np.mean(conciseness_scores)
            avg_engagement = np.mean(engagement_scores)
            
            print("\nLLM Evaluation Results:")
            print(f"Accuracy: {avg_accuracy:.4f}")
            print(f"Clarity: {avg_clarity:.4f}")
            print(f"Completeness: {avg_completeness:.4f}")
            print(f"Conciseness: {avg_conciseness:.4f}")
            print(f"Engagement: {avg_engagement:.4f}")
        except:
            print("Error in LLM evaluation - no valid scores returned")
    
    print("\n===== Evaluation Complete =====")

if __name__ == "__main__":
    main()