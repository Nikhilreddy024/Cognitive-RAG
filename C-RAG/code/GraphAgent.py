import re
import string
import os
import json
import logging
from typing import List, Union, Literal
from enum import Enum

import tiktoken
import time

# Replace langchain with Anthropic
import anthropic

from graph_prompts import GRAPH_DEFINITION
from graph_fewshots import EXAMPLES
from tools import graph_funcs, retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphAgent:
    def __init__(self,
                 args,
                 agent_prompt,
                 ) -> None:

        self.max_steps = args.max_steps
        self.agent_prompt = agent_prompt
        self.examples = EXAMPLES[args.ref_dataset]

        # Anthropic client setup
        self.client = anthropic.Anthropic()
        self.model = args.llm_version or "claude-3-sonnet-20240229"

        # Tokenization
        self.enc = tiktoken.encoding_for_model("text-davinci-003")

        self.graph_definition = GRAPH_DEFINITION[args.dataset]
        self.load_graph(args.graph_dir)
        self.graph_funcs = graph_funcs.graph_funcs(self.graph)
        self.node_retriever = retriever.Retriever(args, self.graph)

        self.__reset_agent()

    def load_graph(self, graph_dir):
        logger.info('Loading the graph...')
        self.graph = json.load(open(graph_dir))

    def run(self, question, answer, reset=True) -> None:
        if reset:
            self.__reset_agent()

        self.question = question
        self.key = answer

        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.prompt_agent()
        self.scratchpad += ' ' + thought
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = thought
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        action_list = get_action_list(action)
        for tmp_action in action_list:
            try:
                action_type, argument = parse_action(tmp_action)
            except:
                self.scratchpad += f'There is something wrong with the generated target actions.'
                continue

            if action_type == 'Finish':
                try:
                    self.answer = eval(argument)
                except:
                    self.answer = argument
                if self.is_correct():
                    self.scratchpad += 'Answer is CORRECT'
                else:
                    self.scratchpad += 'Answer is INCORRECT'
                self.finished = True
                self.step_n += 1
                return

            elif action_type == 'RetrieveNode':
                try:
                    idd, node = self.node_retriever.search_single(argument, 1)
                    self.scratchpad += f"The ID of this retrieval target node is {idd}."
                except Exception as e:
                    self.scratchpad += f'Error in node retrieval: {str(e)}'

            elif action_type == 'NeighbourCheck':
                try:
                    node_id, neighbor_type = argument.split(', ')
                    node_id = remove_quotes(node_id)
                    neighbor_type = remove_quotes(neighbor_type)
                    self.scratchpad += f"The {neighbor_type} neighbors of {node_id} are: " + str(
                        self.graph_funcs.check_neighbours(node_id, neighbor_type)) + '. '
                except Exception as e:
                    self.scratchpad += f'Error in neighbor checking: {str(e)}'

            elif action_type == 'NodeFeature':
                try:
                    node_id, feature_name = argument.split(', ')
                    node_id = remove_quotes(node_id)
                    feature_name = remove_quotes(feature_name)
                    self.scratchpad += f"The {feature_name} feature of {node_id} are: " + self.graph_funcs.check_nodes(
                        node_id, feature_name) + '. '
                except Exception as e:
                    self.scratchpad += f'Error in node feature checking: {str(e)}'

            elif action_type == 'NodeDegree':
                try:
                    node_id, neighbor_type = argument.split(', ')
                    node_id = remove_quotes(node_id)
                    neighbor_type = remove_quotes(neighbor_type)
                    self.scratchpad += f"The {neighbor_type} neighbor node degree of {node_id} are: " + self.graph_funcs.check_degree(
                        node_id, neighbor_type) + '. '
                except Exception as e:
                    self.scratchpad += f'Error in node degree checking: {str(e)}'

            else:
                self.scratchpad += 'Invalid Action. Valid Actions are RetrieveNode[<Content>] NeighbourCheck[<Node>] NodeFeature[<Node>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        try:
            # Correctly use system parameter
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system="You are a helpful AI assistant specialized in graph-based reasoning.",
                messages=[
                    {
                        "role": "user",
                        "content": self._build_agent_prompt()[1].content
                    }
                ]
            )
            return anthropic_format_step(response)
        except Exception as e:
            logger.error(f"Error in prompting agent: {e}")
            return ""

    def _build_agent_prompt(self):
        # Check if question and key attributes exist
        if not hasattr(self, 'question'):
            raise AttributeError("Question not set. Use run() or set_qa() method first.")

        # Use the prompt template to format messages
        formatted_messages = self.agent_prompt.format_messages(
            examples=self.examples,
            question=self.question,
            scratchpad=getattr(self, 'scratchpad', ''),  # Use empty string if scratchpad doesn't exist
            graph_definition=self.graph_definition
        )

        # Return the formatted messages
        return formatted_messages

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt()[1].content)) > 10000)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.answer = ''
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


# Utility functions remain the same as in the original script
def split_checks(input_string):
    pattern = r'\w+\[.*?\]'
    return re.findall(pattern, input_string)


def get_action_list(string):
    if string[:len('Finish')] == 'Finish':
        return [string]
    else:
        return split_checks(string)


def remove_quotes(s):
    if s.startswith(("'", '"')) and s.endswith(("'", '"')):
        return s[1:-1]
    return s


def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    else:
        return None


def anthropic_format_step(response) -> str:
    return response.content[0].text.strip('\n').strip().replace('\n', '')


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))