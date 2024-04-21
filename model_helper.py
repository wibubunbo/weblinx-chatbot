import torch
from functools import partial
from transformers import AutoTokenizer
from weblinx.processing import group_record_to_dict
from weblinx.processing.prompt import build_input_records_from_selected_turns, select_candidates_for_turn
from modeling.dmr.processing import build_records_for_single_turn, build_formatters
from modeling.dmr.eval import verify_queries_are_all_the_same, run_model_and_update_groups, get_ranks_from_scores
from modeling.llama.processing import build_prompt_records_for_llama_truncated, build_formatter_for_multichoice, insert_formatted_chat_into_records
import re
import requests
from dotenv import load_dotenv
import os
load_dotenv()

API_URL_DMR = os.getenv("API_URL_DMR")
API_URL_ACTION = os.getenv("API_URL_ACTION")
headers_dmr = {
	"Accept" : "application/json",
	"Content-Type": "application/json" 
}
headers_action = {
    "Accept" : "application/json",
    "Content-Type": "application/json" 
}

def query_dmr(payload):
	response = requests.post(API_URL_DMR, headers=headers_dmr, json=payload)
	return response.json()

def query_action(payload):
    response = requests.post(API_URL_ACTION, headers=headers_action, json=payload)
    return response.json()

def load_formatters():
    tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/Llama-2-7b-chat-weblinx", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    template_tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/Llama-2-7b-chat-weblinx")

    format_intent_input, _ = build_formatters()
    format_intent = build_formatter_for_multichoice()
    build_prompt_records_fn = partial(
        build_prompt_records_for_llama_truncated,
        format_intent=format_intent,
        tokenizer=tokenizer,
    )

    return format_intent_input, format_intent, build_prompt_records_fn, tokenizer, template_tokenizer


def predict_answer(state, current_turn, replay, format_intent_input, format_intent, build_prompt_records_fn, tokenizer, template_tokenizer):
    if state is not None:
        demo_record = build_records_for_single_turn(
            turn=current_turn,
            replay=replay,
            format_intent_input=format_intent_input,
            uid_key="data-webtasks-id",
            max_neg=None,
            only_allow_valid_uid=False,
            num_utterances=5
        )
        input_grouped = group_record_to_dict(
            demo_record, keys=["demo_name", "turn_index"], remove_keys=False
        )
        # Verify that queries are all the same within each group
        error_msg = "Queries are not all the same within each group"
        assert verify_queries_are_all_the_same(input_grouped), error_msg

        for k, group in input_grouped.items():
            group = input_grouped[k]
            query = group[0]["query"]
            docs = [r["doc"] for r in group]

            scores = query_dmr({
                "inputs": {
                    "sentences": docs,
                    "source_sentence": query,
                    "parameters": {}
                }
            })["similarities"]

            for i, r in enumerate(group):
                r["score"] = scores[i]

            for group in input_grouped.values():
                scores = {r["uid"]: r["score"] for r in group}
                ranks = get_ranks_from_scores(scores)
                for r in group:
                    r["rank"] = ranks[r["uid"]]

        cands_turn = select_candidates_for_turn(
            candidates=input_grouped,
            turn=current_turn,
            num_candidates=10
        )
        selected_turns = [dict(
            replay=replay,
            turn=current_turn,
            cands_turn=cands_turn,
        )]
    else:
        selected_turns = [dict(
            replay=replay,
            turn=current_turn,
            cands_turn=None,
        )]

    input_records = build_input_records_from_selected_turns(
        selected_turns=selected_turns,
        format_intent=format_intent,
        build_prompt_records_fn=build_prompt_records_fn,
        format_prompt_records_fn=None,
    )
    insert_formatted_chat_into_records(
        records=input_records,
        tokenizer=template_tokenizer,
        include_output_target=False,
    )
    out = query_action({
        "inputs": input_records[0]['text'],
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False,
            "pad_token_id": tokenizer.eos_token_id
        }
    })[0]['generated_text']

    answer = re.findall('\w+\([^)]*\)', out)[0] 
    return answer
