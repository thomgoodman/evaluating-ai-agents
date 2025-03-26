import phoenix as px
import os
import json
from tqdm import tqdm
from phoenix.evals import (
    TOOL_CALLING_PROMPT_TEMPLATE, 
    llm_classify,
    OpenAIModel
)
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from openinference.instrumentation import suppress_tracing

import nest_asyncio
nest_asyncio.apply()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import run_agent, start_main_span, tools
from helper import get_phoenix_endpoint

## Running the agent with a set of test questions

# agent_questions = [
#     "What was the most popular product SKU?",
#     "What was the total revenue across all stores?",
#     "Which store had the highest sales volume?",
#     "Create a bar chart showing total sales by store",
#     "What percentage of items were sold on promotion?",
#     "What was the average transaction value?"
# ]

# for question in tqdm(agent_questions, desc="Processing questions"):
#     try:
#         ret = start_main_span([{"role": "user", "content": question}])
#     except Exception as e:
#         print(f"Error processing question: {question}")
#         print(e)
#         continue

## Router evals using LLM-as-a-Judge

PROJECT_NAME = "evaluating-agent"

query = SpanQuery().where(
    # Filter for the `LLM` span kind.
    # The filter condition is a string of valid Python boolean expression.
    "span_kind == 'LLM'",
).select(
    question="input.value",
    tool_call="llm.tools"
)

# The Phoenix Client can take this query and return the dataframe.
tool_calls_df = px.Client().query_spans(query, project_name=PROJECT_NAME, timeout=None)
tool_calls_df = tool_calls_df.dropna(subset=["tool_call"])

#print(tool_calls_df.head())

## Evaluating tool calling
with suppress_tracing():
    tool_call_eval = llm_classify(
        dataframe = tool_calls_df,
        template = TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace("{tool_definitions}", 
                                                                             json.dumps(tools).replace("{", '"').replace("}", '"')),
        rails = ['correct', 'incorrect'],
        model=OpenAIModel(model="gpt-4o"),
        provide_explanation=True
    )

tool_call_eval['score'] = tool_call_eval.apply(lambda x: 1 if x['label']=='correct' else 0, axis=1)

print(tool_call_eval.head())

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Tool Calling Eval", dataframe=tool_call_eval),
)