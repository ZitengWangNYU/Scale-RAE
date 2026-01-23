"""
GPT QA Generator - Utility script for generating question-answer pairs for GenEval prompts.

This script was used to generate the gpt_qa_dict.json file using OpenAI's GPT model.
It creates category-specific questions (object, counting, colors, position) for each prompt.

Usage:
    export OPENAI_API_KEY=your-api-key
    python gpt_qa_generator.py --input evaluation_metadata.jsonl --output gpt_qa_dict.json
"""

import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm


def query_gpt_qa(prompt, api_key=None, model="gpt-4o"):
    """
    Query GPT model to generate category-specific questions for a given prompt.
    
    Args:
        prompt: The image generation prompt to create questions for
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: GPT model to use (default: gpt-4o)
    
    Returns:
        Dictionary with 'prompt' and 'questions' keys
    """
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    question = f"""
Task: Given input prompts, describe each scene with category-specific questions. 
Generate questions for four categories: object, counting, colors, position.
Only generate questions for attributes explicitly described in the prompt.

Example A:
Prompt: a rubix cube with ten squares of purple
Questions:
- Is the object a rubix cube?
- Is there one rubix cube?
- Is the rubix cube purple?
- "" <-- position not specified, no question for this category

Example B:
Prompt: a photo of a bed right of a sports ball
Questions:
- Is there a bed?
- Is there a sports ball?
- Is there only one bed?
- Is there only one sports ball?
- Is the bed on the right of the sports ball?
- "" <-- color not specified, no question for this category

Tasked Prompt:
{prompt}

Output format: {{"prompt": prompt, "questions": [questions...]}}
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates evaluation questions."},
            {"role": "user", "content": question}
        ],
    )
    
    answer = response.choices[0].message.content.strip()
    return json.loads(answer)


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs for GenEval prompts")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with prompts")
    parser.add_argument("--output", type=str, default="gpt_qa_dict.json", help="Output JSON file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="GPT model to use")
    args = parser.parse_args()
    
    # Load existing QA dict if it exists
    qa_dict = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as fp:
            qa_dict = json.load(fp)
        print(f"Loaded {len(qa_dict)} existing QA pairs")
    
    # Load prompts
    with open(args.input) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    # Generate QA pairs for new prompts
    for metadata in tqdm(metadatas, desc="Generating QA pairs"):
        prompt = metadata['prompt']
        
        if prompt in qa_dict:
            continue
        
        try:
            answer = query_gpt_qa(prompt, model=args.model)
            qa_dict[prompt] = answer['questions']
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            continue
    
    # Save updated QA dict
    with open(args.output, "w") as fp:
        json.dump(qa_dict, fp, indent=2)
    
    print(f"Saved {len(qa_dict)} QA pairs to {args.output}")


if __name__ == "__main__":
    main()