import os
import json
import re
import requests
import torch
import datetime as dt
import weblinx as wl
from functools import partial
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score
from weblinx.processing import group_record_to_dict
from weblinx.processing.prompt import build_input_records_from_selected_turns, select_candidates_for_turn
from modeling.dmr.processing import build_records_for_single_turn, build_formatters
from modeling.dmr.eval import verify_queries_are_all_the_same, run_model_and_update_groups, get_ranks_from_scores
from modeling.llama.processing import build_prompt_records_for_llama_truncated, build_formatter_for_multichoice, insert_formatted_chat_into_records
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import uuid
import streamlit as st
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

def add_temporary_action_to_replay(file_path, state, viewport_width, viewport_height, initial_timestamp, replay_data):
    text_dict = {
        "type": "browser",
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "state": {
            "page": state
        },
        "action": {
            "intent": "load",
            "arguments": {
                "metadata": {
                    "url": "https://www.google.com/",
                    "mouseX": 0,
                    "mouseY": 0,
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height
                },
                "properties": {
                    "url": "https://www.google.com/"
                }
            }
        }
    }
    replay_data['data'].append(text_dict)
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

def add_say_to_replay(file_path, speaker, utterance, initial_timestamp, replay_data):
    text_dict = {
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "speaker": speaker,
        "utterance": utterance,
        "type": "chat",
    }
    replay_data['data'][-1] = text_dict
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

def add_load_to_replay(file_path, state, url, viewport_width, viewport_height, initial_timestamp, replay_data):
    text_dict = {
        "type": "browser",
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "state": {
            "page": state
        },
        "action": {
            "intent": "load",
            "arguments": {
                "metadata": {
                    "url": url,
                    "mouseX": 0,
                    "mouseY": 0,
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height
                },
                "properties": {
                    "url": url
                }
            }
        }
    }
    replay_data['data'][-1] = text_dict
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

def add_click_to_replay(file_path, state, url, viewport_width, viewport_height, initial_timestamp, mouseX, mouseY, attributes, bbox, tagName, replay_data):
    text_dict = {
        "type": "browser",
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "state": {
            "page": state
        },
        "action": {
            "intent": "click",
            "arguments": {
                "metadata": {
                    "url": url,
                    "mouseX": mouseX,
                    "mouseY": mouseY,
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height
                },
                "element": {
                    "url": url,
                    "attributes": attributes,
                    "bbox": bbox,
                    "tagName": tagName
                }
            }
        }
    }
    replay_data['data'][-1] = text_dict
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

def add_textInput_to_replay(file_path, state, url, viewport_width, viewport_height, initial_timestamp, mouseX, mouseY, attributes, bbox, tagName, text, replay_data):
    text_dict = {
        "type": "browser",
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "state": {
            "page": state
        },
        "action": {
            "intent": "textInput",
            "arguments": {
                "metadata": {
                    "url": url,
                    "mouseX": mouseX,
                    "mouseY": mouseY,
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height
                },
                "element": {
                    "url": url,
                    "attributes": attributes,
                    "bbox": bbox,
                    "tagName": tagName
                },
                "text": text
            }
        }
    }
    replay_data['data'][-1] = text_dict
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

def add_scroll_to_replay(file_path, state, url, viewport_width, viewport_height, initial_timestamp, scrollX, scrollY, replay_data):
    text_dict = {
        "type": "browser",
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "state": {
            "page": state
        },
        "action": {
            "intent": "scroll",
            "arguments": {
                "metadata": {
                    "url": url,
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height
                },
                "scrollX": scrollX,
                "scrollY": scrollY
            }
        }
    }
    replay_data['data'][-1] = text_dict
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

def add_submit_to_replay(file_path, state, url, viewport_width, viewport_height, initial_timestamp, mouseX, mouseY, attributes, bbox, tagName, replay_data):
    text_dict = {
        "type": "browser",
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "state": {
            "page": state
        },
        "action": {
            "intent": "submit",
            "arguments": {
                "metadata": {
                    "url": url,
                    "mouseX": mouseX,
                    "mouseY": mouseY,
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height
                },
                "element": {
                    "url": url,
                    "attributes": attributes,
                    "bbox": bbox,
                    "tagName": tagName
                }
            }
        }
    }
    replay_data['data'][-1] = text_dict
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

def add_change_to_replay(file_path, state, url, viewport_width, viewport_height, initial_timestamp, mouseX, mouseY, attributes, bbox, tagName, value, replay_data):
    text_dict = {
        "type": "browser",
        "timestamp": (dt.datetime.now() - dt.datetime.strptime(initial_timestamp, "%H:%M:%S")).seconds,
        "state": {
            "page": state
        },
        "action": {
            "intent": "change",
            "arguments": {
                "metadata": {
                    "url": url,
                    "mouseX": mouseX,
                    "mouseY": mouseY,
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height
                },
                "element": {
                    "url": url,
                    "attributes": attributes,
                    "bbox": bbox,
                    "tagName": tagName
                },
                "value": value
            }
        }
    }
    replay_data['data'][-1] = text_dict
    with open(file_path, 'w') as f:
        json.dump(replay_data, f)

os.makedirs('./live_data', exist_ok=True)
os.makedirs('./live_data/bboxes', exist_ok=True)
os.makedirs('./live_data/pages', exist_ok=True)

# Delete screenshot if it exists
if os.path.exists('./screenshot.png'):
    os.remove('./screenshot.png')

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

API_URL_DMR = "https://tz9o4fzbt55lqfsl.us-east-1.aws.endpoints.huggingface.cloud"
headers_dmr = {
	"Accept" : "application/json",
	"Content-Type": "application/json" 
}

def query_dmr(payload):
	response = requests.post(API_URL_DMR, headers=headers_dmr, json=payload)
	return response.json()

API_URL_ACTION = "https://z2wzbni4hb7rcrb5.us-east-1.aws.endpoints.huggingface.cloud"
headers_action = {
    "Accept" : "application/json",
    "Content-Type": "application/json" 
}

def query_action(payload):
    response = requests.post(API_URL_ACTION, headers=headers_action, json=payload)
    return response.json()

# Set the window size
viewport_width = 1600
viewport_height = 900
# driver.set_window_size(viewport_width, viewport_height)

st.set_page_config(layout="wide")  # Set the layout to wide
st.title("WebLINX Chatbot")

# Description
st.sidebar.write("""
This chatbot can help you navigate the web by performing actions such as clicking buttons, scrolling,
submitting forms, etc., directly on the webpage. Start by typing your message below and pressing Enter.
Type 'Continue' to proceed with actions, and 'Quit' to stop the chatbot.
""")

# Initialize session state for messages if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'turn_index' not in st.session_state:
    st.session_state.turn_index = -1

if 'html_and_bboxes_index' not in st.session_state:
    st.session_state.html_and_bboxes_index = 0

if 'initial_timestamp' not in st.session_state:
    st.session_state.initial_timestamp = dt.datetime.now().strftime("%H:%M:%S")

if 'driver' not in st.session_state:
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    st.session_state.driver = webdriver.Chrome(options=chrome_options)
    st.session_state.driver.set_window_size(1600, 900)

if 'replay_data' not in st.session_state:
    st.session_state.replay_data = {"data": []}
    with open("./live_data/replay.json", "w") as f:
        f.write('{"data": []}\n')

with st.sidebar:
    if user_chat := st.chat_input("Type your message here:"):
        if user_chat == "Quit":
            add_temporary_action_to_replay('./live_data/replay.json', None, viewport_height, viewport_width, st.session_state.initial_timestamp, st.session_state.replay_data)
            st.session_state.turn_index += 1
            add_say_to_replay('./live_data/replay.json', "instructor", "That's all", st.session_state.initial_timestamp, st.session_state.replay_data)
            st.session_state.driver.quit()
            st.session_state.messages = []
        elif user_chat != "Continue":
            st.session_state.messages.append({"role": "user", "content": user_chat})
            add_temporary_action_to_replay('./live_data/replay.json', None, viewport_height, viewport_width, st.session_state.initial_timestamp, st.session_state.replay_data)
            st.session_state.turn_index += 1
            add_say_to_replay('./live_data/replay.json', "instructor", user_chat, st.session_state.initial_timestamp, st.session_state.replay_data)

            state = f"page-{st.session_state.html_and_bboxes_index - 1}-{0}.html" if st.session_state.html_and_bboxes_index > 0 else None
            add_temporary_action_to_replay('./live_data/replay.json', state, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
            st.session_state.turn_index += 1

            demo = wl.Demonstration('live_data', base_dir='.')
            replay = wl.Replay.from_demonstration(demo)
            current_turn = wl.Turn.from_replay(replay, st.session_state.turn_index)
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
            print(answer)
            
            if answer.startswith('say'):
                add_say_to_replay('./live_data/replay.json', "navigator", answer.split('utterance="')[1][:-2], st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.messages.append({"role": "assistant", "content": answer.split('utterance="')[1][:-2]})

            elif answer.startswith('load'):
                url = re.findall('url="([^"]*)"', answer)[0]
                st.session_state.driver.get(url)
                time.sleep(10)
                # Generate unique IDs for each element in the DOM
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}

                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                
                add_load_to_replay('./live_data/replay.json', state, url, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('click'):
                target_uid = re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element                
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_click_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                target_element.click()
                time.sleep(10)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }
                
                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('text_input'):
                text, target_uid = re.findall('text="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_textInput_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, text, st.session_state.replay_data)
                target_element.send_keys(text)
                time.sleep(5)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }
                
                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('scroll'):
                scrollX, scrollY = re.findall('x="([^"]*)"', answer)[0], re.findall('y="([^"]*)"', answer)[0]
                st.session_state.driver.execute_script(f"window.scrollTo({scrollX}, {scrollY});")
                time.sleep(5)
                add_scroll_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, scrollX, scrollY, st.session_state.replay_data)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('submit'):
                target_uid = re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_submit_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                target_element.submit()
                time.sleep(20)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width" : rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('change'):
                value, target_uid = re.findall('value="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_change_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, value, st.session_state.replay_data)
                target_element.send_keys(value)
                time.sleep(5)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width" : rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            else:
                raise ValueError(f"Invalid action: {answer}")

        elif user_chat == "Continue":
            state = f"page-{st.session_state.html_and_bboxes_index - 1}-{0}.html" if st.session_state.html_and_bboxes_index > 0 else None
            add_temporary_action_to_replay('./live_data/replay.json', state, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
            st.session_state.turn_index += 1

            demo = wl.Demonstration('live_data', base_dir='.')
            replay = wl.Replay.from_demonstration(demo)
            current_turn = wl.Turn.from_replay(replay, st.session_state.turn_index)
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
            print(answer)
            
            if answer.startswith('say'):
                add_say_to_replay('./live_data/replay.json', "navigator", answer.split('utterance="')[1][:-2], st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.messages.append({"role": "assistant", "content": answer.split('utterance="')[1][:-2]})

            elif answer.startswith('load'):
                url = re.findall('url="([^"]*)"', answer)[0]
                st.session_state.driver.get(url)
                time.sleep(10)
                # Generate unique IDs for each element in the DOM
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}

                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                
                add_load_to_replay('./live_data/replay.json', state, url, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('click'):
                target_uid = re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element                
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_click_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                target_element.click()
                time.sleep(10)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }
                
                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('text_input'):
                text, target_uid = re.findall('text="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_textInput_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, text, st.session_state.replay_data)
                target_element.send_keys(text)
                time.sleep(5)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }
                
                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('scroll'):
                scrollX, scrollY = re.findall('x="([^"]*)"', answer)[0], re.findall('y="([^"]*)"', answer)[0]
                st.session_state.driver.execute_script(f"window.scrollTo({scrollX}, {scrollY});")
                time.sleep(5)
                add_scroll_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, scrollX, scrollY, st.session_state.replay_data)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width": rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('submit'):
                target_uid = re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_submit_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                target_element.submit()
                time.sleep(20)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width" : rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif answer.startswith('change'):
                value, target_uid = re.findall('value="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                target_element = st.session_state.driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{target_uid}"]')
                # Obtain the bounding rectangle of the target element
                attrs = ["class", "title", "href", "aria-label", "d", "src"]
                attrs_dict = {}
                for attr in attrs:
                    if target_element.get_attribute(attr) is not None:
                        attrs_dict[attr] = target_element.get_attribute(attr)
                attrs_dict["data-webtasks-id"] = target_uid
                rect = target_element.rect
                bbox = {
                    "x": rect['x'],
                    "y": rect['y'],
                    "width": rect['width'],
                    "height": rect['height'],
                    "top": rect['y'],
                    "right": rect['x'] + rect['width'],
                    "bottom": rect['y'] + rect['height'],
                    "left": rect['x']
                }
                add_change_to_replay('./live_data/replay.json', state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, rect['x'], rect['y'], attrs_dict, bbox, target_element.tag_name, value, st.session_state.replay_data)
                target_element.send_keys(value)
                time.sleep(5)
                elements = st.session_state.driver.find_elements(By.XPATH, "//*")
                element_data = {}
                for element in elements:
                    uid = element.get_attribute('data-webtasks-id')
                    if not uid:
                        uid = str(uuid.uuid4())[:18]
                        st.session_state.driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                    
                    # Obtain the bounding rectangle of the element
                    rect = element.rect
                    element_data[uid] = {
                        "x": rect['x'],
                        "y": rect['y'],
                        "width" : rect['width'],
                        "height": rect['height'],
                        "top": rect['y'],
                        "right": rect['x'] + rect['width'],
                        "bottom": rect['y'] + rect['height'],
                        "left": rect['x']
                    }

                # Save HTML content to pages folder
                with open(f'./live_data/pages/page-{st.session_state.html_and_bboxes_index}-{0}.html', 'w') as f:
                    f.write(st.session_state.driver.page_source)
                # Save the bounding boxes to bboxes folder
                with open(f'./live_data/bboxes/bboxes-{st.session_state.html_and_bboxes_index}.json', 'w') as f:
                    json.dump(element_data, f)
                st.session_state.html_and_bboxes_index += 1
                st.session_state.driver.save_screenshot('./screenshot.png')
                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                raise ValueError(f"Invalid action: {answer}")

# Main area: Chat history and screenshot display
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Chat History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.header("Web Page View")
    if os.path.exists('./screenshot.png'):
        st.image('screenshot.png', caption="Current Web Page", use_column_width=True)
    else:
        st.image('microphonecat.png', caption="No Screenshot Available", use_column_width=True)
