import os
import re
import json
import jsonlines
import logging
import torch
import gc
import psutil
from transformers import AutoTokenizer
import asyncio
from anthropic import AsyncAnthropic
from retriever import Retriever, NODE_TEXT_KEYS
import argparse

import time

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



def load_questions_from_file(file_path):
    """
    Load all questions from the specified JSONLines file.

    Args:
        file_path: Path to the JSONLines file.

    Returns:
        List of all questions from the file.
    """
    logger = logging.getLogger(__name__)
    all_questions = []

    with open(file_path, 'r') as f:
        reader = jsonlines.Reader(f)
        all_questions = list(reader)  # Load all questions into a list

    logger.info(f"Loaded {len(all_questions)} questions from the file.")
    return all_questions


async def dispatch_anthropic_request(args, messages, model, temperature=0.01, max_tokens=4096, retries=5, delay=5):
    client = AsyncAnthropic(api_key=args.anthropic_key)

    for attempt in range(retries):
        try:
            system_prompt = next(msg["content"] for msg in messages if msg["role"] == "system")
            user_message = next(msg["content"] for msg in messages if msg["role"] == "user")

            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            return response

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed after {retries} attempts: {str(e)}")
                raise

            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            await asyncio.sleep(delay * (attempt + 1))

def truncate_string(input_text, tokenizer, max_len):
    return tokenizer.decode(
        tokenizer.encode(input_text, add_special_tokens=False, truncation=True, max_length=max_len))


def print_memory_usage():
    """Print current CPU and GPU memory usage."""
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    gpu_cache = torch.cuda.memory_reserved() / 1024 / 1024
    logger.info(f"CPU Memory: {cpu_mem:.2f} MB")
    logger.info(f"GPU Memory allocated: {gpu_mem:.2f} MB")
    logger.info(f"GPU Memory cached: {gpu_cache:.2f} MB")


def clear_memory():
    """Clear unused memory."""
    gc.collect()
    torch.cuda.empty_cache()


class MemoryEfficientRetriever:
    def __init__(self, args, graph):
        self.args = args
        self.graph = graph
        self.batch_size = 8  # Smaller batch size for processing

    def process_chunks(self, chunks):
        """Process chunks in smaller batches."""
        results = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            # Process batch
            results.extend(batch)
            if i % (self.batch_size * 4) == 0:
                clear_memory()
        return results


async def process_single_question(args, question_item, tokenizer, retriever):
    """Process a single question with memory monitoring."""
    try:
        print_memory_usage()

        system_message = "You are an AI assistant to answer questions. Please use your own knowledge and the given context to answer the questions. If you do not know the answer, please guess a most probable answer. Only include the answer in your response. Do not explain."



        context = truncate_string(
            retriever.search_single(query=question_item["question"],
                                    hop=args.retrieve_graph_hop,
                                    topk=1),
            tokenizer,
            args.graph_max_len
        )

        message = question_item["question"] + '\nContext: ' + context
        query_message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]

        response = await dispatch_anthropic_request(
            args,
            messages=query_message,
            model=args.model,
            temperature=0.01,
            max_tokens=args.max_len,
        )

        clear_memory()
        return {
            "question": question_item["question"],
            "context": context,
            "model_answer": response.content[0].text,
            "gt_answer": question_item["answer"]
        }

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        await asyncio.sleep(60)
        return {
            "question": question_item["question"],
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser("")

    parser.add_argument("--dataset", type=str, default="maple")
    parser.add_argument("--model", type=str, default="claude-3-sonnet-20240229")
    parser.add_argument("--graph_dir", type=str, default="None")
    parser.add_argument("--path", type=str, default="None")
    parser.add_argument("--save_file", type=str, default="None")
    parser.add_argument("--anthropic_key", type=str, default="None")
    parser.add_argument("--embedder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--faiss_gpu", type=bool, default=False)
    parser.add_argument("--embed_cache", type=bool, default=True)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--retrieve_graph_hop", type=int, default=1)
    parser.add_argument("--graph_max_len", type=int, default=3000)
    args = parser.parse_args()

    args.embed_cache_dir = args.path
    args.graph_dir = os.path.join(args.path, "graph.json")
    args.data_dir = os.path.join(args.path, "data.json")
    args.retrieval_context_dir = os.path.join(args.path, f"retrieval_context_{args.retrieve_graph_hop}.json")
    args.node_text_keys = NODE_TEXT_KEYS[args.dataset]

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.8)

    print_memory_usage()
    logger.info("Starting process...")

    # Updated question loading
    questions = load_questions_from_file(args.data_dir)
    print_memory_usage()
    logger.info(f"Loaded {len(questions)} questions total...")



    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-13b-chat-hf',
        use_fast=False,
        local_files_only=True
    )

    logger.info('Loading the graph...')
    with open(args.graph_dir) as f:
        graph = json.load(f)

    print_memory_usage()

    logger.info('Initializing retriever...')
    retriever = Retriever(args, graph)
    retriever.encode_batch_size = 32

    del graph
    clear_memory()

    print_memory_usage()
    logger.info('Processing questions...')

    async def process_all_questions():
        results = []
        for i, question in enumerate(questions):
            try:
                task_result = await process_single_question(args, question, tokenizer, retriever)
                results.append(task_result)

                # More aggressive rate limiting
                wait_time = 1 + (i // 40) * 1  # Increasing wait time
                logger.info(f"Waiting {wait_time} seconds between requests...")
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                results.append({
                    "question": question["question"],
                    "error": str(e)
                })

        return results

    results = asyncio.run(process_all_questions())

    output_file_path = args.save_file
    parent_folder = os.path.dirname(output_file_path)
    parent_parent_folder = os.path.dirname(parent_folder)

    os.makedirs(parent_parent_folder, exist_ok=True)
    os.makedirs(parent_folder, exist_ok=True)

    with jsonlines.open(output_file_path, 'w') as writer:
        for result in results:
            writer.write(result)
            logger.info(f"Processed result: {result}")

    print_memory_usage()
    logger.info("Process completed successfully")


if __name__ == '__main__':
    main()