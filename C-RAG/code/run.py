import os
from tqdm import tqdm
import logging
import argparse
import jsonlines
import datetime
from GraphAgent import GraphAgent
from tools.retriever import NODE_TEXT_KEYS
from graph_prompts import graph_agent_prompt, graph_agent_prompt_zeroshot

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str, default="dblp")
parser.add_argument("--anthropic_api_key", type=str, default="xxx")
parser.add_argument("--path", type=str)
parser.add_argument("--save_file", type=str)
parser.add_argument("--embedder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
parser.add_argument("--faiss_gpu", type=bool, default=False)
parser.add_argument("--embed_cache", type=bool, default=True)
parser.add_argument("--max_steps", type=int, default=15)
parser.add_argument("--zero_shot", type=bool, default=False)
parser.add_argument("--ref_dataset", type=str, default=None)

parser.add_argument("--llm_version", type=str, default="claude-3-haiku-20240307")
args = parser.parse_args()

args.embed_cache_dir = args.path
args.graph_dir = os.path.join(args.path, "graph.json")
args.data_dir = os.path.join(args.path, "data.json")
args.node_text_keys = NODE_TEXT_KEYS[args.dataset]
args.ref_dataset = args.dataset if not args.ref_dataset else args.ref_dataset

assert args.llm_version in [
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307'
]


def remove_fewshot(prompt):
    """Remove few-shot examples from the prompt"""
    if isinstance(prompt, list):
        # If it's a list of messages
        if hasattr(prompt[0], 'content'):
            content = prompt[0].content
        else:
            content = prompt[0]['content']
    elif isinstance(prompt, str):
        # If it's a direct string
        content = prompt
    else:
        # Try to get content in other ways
        content = str(prompt)

    # Check if 'Here are some examples:' exists
    if 'Here are some examples:' in content:
        prefix = content.split('Here are some examples:')[0]
        suffix = content.split('(END OF EXAMPLES)')[1]
        return prefix.strip('\n').strip() + '\n' + suffix.strip('\n').strip()
    else:
        # If no examples found, return the entire content
        return content.strip('\n')


def main():
    # Set Anthropic API key
    os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key

    # Load data
    with open(args.data_dir, 'r') as f:
        contents = []
        for item in jsonlines.Reader(f):
            contents.append(item)

    # Prepare output directories
    output_file_path = args.save_file

    parent_folder = os.path.dirname(output_file_path)
    parent_parent_folder = os.path.dirname(parent_folder)

    # Create directories if they don't exist
    os.makedirs(parent_parent_folder, exist_ok=True)
    os.makedirs(parent_folder, exist_ok=True)

    logs_dir = os.path.join(parent_folder, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Select prompt based on zero-shot setting
    agent_prompt = graph_agent_prompt if not args.zero_shot else graph_agent_prompt_zeroshot

    # Initialize agent
    agent = GraphAgent(args, agent_prompt)

    # Tracking variables
    unanswered_questions = []
    correct_logs = []
    halted_logs = []
    incorrect_logs = []
    generated_text = []





    # Process each question
    for i in tqdm(range(len(contents))):
        # Set the question and answer before running
        agent.set_qa(contents[i]['question'], contents[i]['answer'])

        # Run agent on the question
        agent.run(contents[i]['question'], contents[i]['answer'])

        print(f'Ground Truth Answer: {agent.key}')
        print(f'Model Answer: {agent.answer}')  # Add this to print the model's answer
        print('---------')

        # Prepare log
        log = f"Question: {contents[i]['question']}\n"

        # Modify log generation to handle both zero-shot and few-shot scenarios
        if not args.zero_shot:
            # For few-shot, remove the examples from the prompt
            full_prompt = agent._build_agent_prompt()[1].content
            prefix = full_prompt.split('Here are some examples:')[0]
            suffix = full_prompt.split('(END OF EXAMPLES)')[1]
            log += prefix.strip('\n') + '\n' + suffix.strip('\n')
        else:
            # For zero-shot, use the entire prompt
            log += agent._build_agent_prompt()[0].content

        log += f'\nCorrect answer: {agent.key}\n\n'

        # Write individual question log
        with open(os.path.join(logs_dir, contents[i]['qid'] + '.txt'), 'w') as f:
            f.write(log)

        # Ensure a single, consistent model answer
        model_answer = agent.answer if agent.answer else "No answer generated"

        # Store generated text
        generated_text.append({
            "question": contents[i]["question"],
            "model_answer": model_answer,  # Use the last generated answer
            "gt_answer": contents[i]['answer']
        })

        # Categorize logs
        if agent.is_correct():
            correct_logs.append(log)
        elif agent.is_halted():
            halted_logs.append(log)
        elif agent.is_finished() and not agent.is_correct():
            incorrect_logs.append(log)
        else:
            raise ValueError('Something went wrong!')

    # Write generated text to output file
    with jsonlines.open(output_file_path, 'w') as writer:
        for row in generated_text:
            writer.write(row)

    # Print summary
    print(
        f'Finished Trial {len(contents)}, Correct: {len(correct_logs)}, Incorrect: {len(incorrect_logs)}, Halted: {len(halted_logs)}')
    print('Unanswered questions {}: {}'.format(len(unanswered_questions), unanswered_questions))

if __name__ == '__main__':
    main()