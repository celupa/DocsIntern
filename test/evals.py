import os
from dotenv import load_dotenv
import ast
from datetime import datetime
import random
import pandas as pd
from openai import OpenAI
from helpers import cfg as c
from helpers import custom_functions as cfoos
from helpers import model_selector as ms
from tests import docs as td


# log into OpenAI
load_dotenv()
os.getenv("OPENAI_API_KEY")
EVAL_MODEL = 'gpt-4o'
openai = OpenAI()
# load model
app = ms.ModelSelector(c.MODEL_CONFIG)


def prompt_gpt(user_prompt: str) -> str:
    """Prompt GPT to evaluate APP model on stress questions."""
    sys_prompt = """You will receive a python dictionary contanining pairs of 
    stress questions and responses coming from an LLM.
    The LLM has been instructed to be helpful and brief in its responses.
    For each item in the dictionary assess the quality of the answer from 0 to 100.
    If the LLM hasn't responded or doesn't know the answer rate the item 0.
    If the LLM only corrects the user without providing additional information, rate the answer lower.
    Answers that are more precise and factually correct, should be rated higher.
    Finally, you return the full dictionary in string format with no additonal text and 
    "answer_precision" filled with your evaluations.
    """
    output = openai.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
        ]
    )
    response = output.choices[0].message.content
    return response


# set-up test
def test_capacity_latency(csv_results: str=None) -> None:
    """
    Test model capacity and latency.
    Here, we use  misspelled and/or factually-incorrect questions.
    If csv_results
    """
    # set path for results
    evaluation_df_path = c.TESTS_PATH / f"test_capacity_latency_{app.model_name}_{datetime.now()}.csv"
    # if results were previously stored, retrieve them
    if csv_results:
        try:
            print(f"---Loading previous results: {csv_results}")
            saved_results = pd.read_csv(csv_results)
            results = saved_results.to_dict(orient="list")
        except Exception:
            print("---Failed to load previous results. Check file/path.")
    else:
        results = {
             "general_questions": [],
             "llm_answers": [],
             "response_time_sec": [],
             "answer_precision": []
             }
        print("---Prompting APP Model...")
        for i, question in enumerate(td.stress_questions):
            print(f"---Q{i + 1}: {question}")
            results["general_questions"].append(question)
            start = datetime.now()
            results["llm_answers"].append(app.prompt_llm(question))
            end = datetime.now()
            latency = (end - start).total_seconds() / 60
            results["response_time_sec"].append(latency)
        # save unrated results in case of unforseen problems
        df = pd.DataFrame.from_dict(results, orient="index").T
        df.to_csv(evaluation_df_path, index=False)

    print("---Evaluating responses...")
    evaluator_response = prompt_gpt(str(results))

    evaluator_dict = ast.literal_eval(evaluator_response)
    final_df = pd.DataFrame.from_dict(evaluator_dict)
    final_df.to_csv(evaluation_df_path / "_EVALED", index=False)
    final_df["response_time_sec"] = final_df["response_time_sec"].astype("float")
    final_df["answer_precision"] = final_df["answer_precision"].astype("float")

    print("---Capacity & latency results (stress questions):")
    print(f"---Model Average Latency: {final_df['response_time_sec'].mean()}")
    print(f"---Model Average Precision: {final_df['answer_precision'].mean()}")
