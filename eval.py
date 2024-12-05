
import json
import evaluate
import argparse
from anthropic import Anthropic
import os
import re

from IPython import embed

parser = argparse.ArgumentParser("")
parser.add_argument("--result_file", type=str, default="None")
parser.add_argument("--model", type=str, default="None")
parser.add_argument("--anthropic_key", type=str, default="None")
parser.add_argument("--output_log", type=str, default="evaluation_results.jsonl")
args = parser.parse_args()


def compute_exact_match(predictions, references):
    em_metric = evaluate.load('exact_match')
    return em_metric.compute(predictions=predictions, references=references)


def compute_bleu(predictions, references):
    bleu_metric = evaluate.load('bleu')
    return bleu_metric.compute(predictions=predictions, references=references)


def compute_rouge(predictions, references):
    rouge_metric = evaluate.load('rouge')
    return rouge_metric.compute(predictions=predictions, references=references)


def Claude3score(predictions, references, questions):
    client = Anthropic(api_key=args.anthropic_key)
    eval_prompt = "Question:{} \nModel prediction: {} \nGround truth: {}. \nPlease help me judge if the model prediction is correct or not given the question and ground truth answer. Please use one word (Yes or No) to answer. Do not explain."

    res = []
    for pred, ref, question in zip(predictions, references, questions):
        x = eval_prompt.format(question, pred, ref)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
            messages=[
                {"role": "user", "content": x},
            ],
            system="You are a generative language model evaluator. Respond with Yes or No.",
            temperature=0.01
        )
        Claude_score = response.content[0].text.strip()
        try:
            assert Claude_score in ['Yes', 'No']
        except:
            print(f"Unexpected response: {Claude_score}")
            embed()
        if Claude_score == 'Yes':
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res)


def read_json(file):
    results = []
    preds = []
    gts = []
    questions = []
    with open(file) as f:
        readin = f.readlines()
        for line in readin:
            try:
                tmp = json.loads(line)
                if all(key in tmp for key in ['model_answer', 'gt_answer', 'question']):
                    results.append(tmp)
                    preds.append(tmp['model_answer'])
                    gts.append(tmp['gt_answer'])
                    questions.append(tmp['question'])
                else:
                    print(f"Skipping entry due to missing keys: {tmp}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON line: {line}")
    return results, preds, gts, questions


def sanitize_filename(filename):
    # Remove path and extension, replace non-alphanumeric characters
    base_filename = os.path.basename(filename)
    sanitized = re.sub(r'[^\w\-_\.]', '_', base_filename)
    return os.path.splitext(sanitized)[0]


def log_results(output_log, model, result_file, results):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_log), exist_ok=True)

    # Sanitize the input filename to use as a key
    input_file_key = sanitize_filename(result_file)

    # Prepare the result dictionary
    log_entry = {
        input_file_key: {
            'model': model,
            **results
        }
    }

    # Append to the log file
    with open(output_log, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    print(f"Results logged to {output_log}")


# Main execution
results, preds, gts, questions = read_json(args.result_file)
preds = [pred if pred != None else '' for pred in preds]

# Compute metrics
em_score = compute_exact_match(preds, gts)
bleu_score = compute_bleu(preds, gts)
rouge_score = compute_rouge(preds, gts)
claude_score = Claude3score(preds, gts, questions)

# Print results
result_output = f"{args.model} || EM: {em_score['exact_match']} | Bleu: {bleu_score['bleu']} | Rouge1: {rouge_score['rouge1']} | Rouge2: {rouge_score['rouge2']} | RougeL: {rouge_score['rougeL']} | RougeLSum: {rouge_score['rougeLsum']} | Claude3score: {claude_score}"
print(result_output)

# Log results to file
log_results(
    args.output_log,
    args.model,
    args.result_file,
    {
        'exact_match': em_score['exact_match'],
        'bleu': bleu_score['bleu'],
        'rouge1': rouge_score['rouge1'],
        'rouge2': rouge_score['rouge2'],
        'rougeL': rouge_score['rougeL'],
        'rougeLsum': rouge_score['rougeLsum'],
        'claude_score': claude_score
    }
)
