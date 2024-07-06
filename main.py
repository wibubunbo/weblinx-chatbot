import os
import re
import datetime as dt
import weblinx as wl
import time
import streamlit as st
import logging
logging.basicConfig(level=logging.WARNING)
from typing import Dict, Any

import replay_helper
import model_helper
import browser_helper

# Constants
DATA_DIR = './live_data'
VIEWPORT_WIDTH = 1600
VIEWPORT_HEIGHT = 900
SCREENSHOT_PATH = './screenshot.png'

logging.basicConfig(level=logging.WARNING)

def initialize_session_state():
    if 'initialized' not in st.session_state:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'pages'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'bboxes'), exist_ok=True)

        for folder in ['pages', 'bboxes']:
            full_path = os.path.join(DATA_DIR, folder)
            for file in os.listdir(full_path):
                os.remove(os.path.join(full_path, file))

        if os.path.exists(SCREENSHOT_PATH):
            os.remove(SCREENSHOT_PATH)

        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.turn_index = -1
        st.session_state.html_and_bboxes_index = 0
        st.session_state.initial_timestamp = dt.datetime.now().strftime("%H:%M:%S")
        st.session_state.driver = browser_helper.initialize_browser(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
        st.session_state.replay_data = {"data": []}
        with open(os.path.join(DATA_DIR, 'replay.json'), 'w') as f:
            f.write('{"data": []}\n')
        st.session_state.format_intent_input, st.session_state.format_intent, st.session_state.build_prompt_records_fn, st.session_state.tokenizer, st.session_state.template_tokenizer = model_helper.load_formatters()

def handle_user_input(user_chat: str):
    if user_chat == "Quit":
        handle_quit()
    else:
        handle_new_message(user_chat)

def handle_quit():
    replay_helper.add_temporary_action_to_replay(get_replay_file_path(), None, VIEWPORT_HEIGHT, VIEWPORT_WIDTH, st.session_state.initial_timestamp, st.session_state.replay_data)
    st.session_state.turn_index += 1
    replay_helper.add_say_to_replay(get_replay_file_path(), "instructor", "That's all", st.session_state.initial_timestamp, st.session_state.replay_data)
    st.session_state.driver.quit()
    st.session_state.messages = []

def handle_new_message(user_chat: str):
    st.session_state.messages.append({"role": "user", "content": user_chat})
    replay_helper.add_temporary_action_to_replay(get_replay_file_path(), None, VIEWPORT_HEIGHT, VIEWPORT_WIDTH, st.session_state.initial_timestamp, st.session_state.replay_data)
    st.session_state.turn_index += 1
    replay_helper.add_say_to_replay(get_replay_file_path(), "instructor", user_chat, st.session_state.initial_timestamp, st.session_state.replay_data)
    process_model_response()

def handle_continue():
    process_model_response()

def process_model_response():
    state = get_current_state()
    replay_helper.add_temporary_action_to_replay(get_replay_file_path(), state, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, st.session_state.initial_timestamp, st.session_state.replay_data)
    st.session_state.turn_index += 1

    demo = wl.Demonstration(DATA_DIR, base_dir='.')
    replay = wl.Replay.from_demonstration(demo)
    current_turn = wl.Turn.from_replay(replay, st.session_state.turn_index)

    answer = model_helper.predict_answer(state, current_turn, replay, st.session_state.format_intent_input, st.session_state.format_intent, st.session_state.build_prompt_records_fn, st.session_state.tokenizer, st.session_state.template_tokenizer)
    
    handle_model_action(answer)

def handle_model_action(answer: str):
    action_handlers = {
        'say': handle_say_action,
        'load': handle_load_action,
        'click': handle_click_action,
        'text_input': handle_text_input_action,
        'scroll': handle_scroll_action,
        'submit': handle_submit_action,
        'change': handle_change_action
    }

    action_type = answer.split('(')[0]
    handler = action_handlers.get(action_type)

    if handler:
        handler(answer)
    else:
        raise ValueError(f"Invalid action: {answer}")

def handle_say_action(answer: str):
    utterance = re.findall('utterance="([^"]*)"', answer)[0]
    replay_helper.add_say_to_replay(get_replay_file_path(), "navigator", utterance, st.session_state.initial_timestamp, st.session_state.replay_data)
    st.session_state.messages.append({"role": "assistant", "content": utterance})

def handle_load_action(answer: str):
    url = re.findall('url="([^"]*)"', answer)[0]
    st.session_state.driver.get(url)
    time.sleep(2)
    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, DATA_DIR)
    replay_helper.add_load_to_replay(get_replay_file_path(), get_current_state(), url, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, st.session_state.initial_timestamp, st.session_state.replay_data)
    st.session_state.html_and_bboxes_index += 1
    st.session_state.driver.save_screenshot(SCREENSHOT_PATH)
    st.session_state.messages.append({"role": "assistant", "content": answer})

def handle_click_action(answer: str):
    uid = re.findall('uid="([^"]*)"', answer)[0]
    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
    replay_helper.add_click_to_replay(get_replay_file_path(), get_current_state(), st.session_state.driver.current_url, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
    target_element.click()
    time.sleep(2)
    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, DATA_DIR)
    st.session_state.html_and_bboxes_index += 1
    st.session_state.driver.save_screenshot(SCREENSHOT_PATH)
    st.session_state.messages.append({"role": "assistant", "content": answer})

def handle_text_input_action(answer: str):
    text, uid = re.findall('text="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
    replay_helper.add_textInput_to_replay(get_replay_file_path(), get_current_state(), st.session_state.driver.current_url, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, text, st.session_state.replay_data)
    target_element.send_keys(text)
    time.sleep(2)
    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, DATA_DIR)
    st.session_state.html_and_bboxes_index += 1
    st.session_state.driver.save_screenshot(SCREENSHOT_PATH)
    st.session_state.messages.append({"role": "assistant", "content": answer})

def handle_scroll_action(answer: str):
    scrollX, scrollY = re.findall('x="([^"]*)"', answer)[0], re.findall('y="([^"]*)"', answer)[0]
    st.session_state.driver.execute_script(f"window.scrollTo({scrollX}, {scrollY});")
    time.sleep(2)
    replay_helper.add_scroll_to_replay(get_replay_file_path(), get_current_state(), st.session_state.driver.current_url, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, st.session_state.initial_timestamp, scrollX, scrollY, st.session_state.replay_data)
    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, DATA_DIR)
    st.session_state.html_and_bboxes_index += 1
    st.session_state.driver.save_screenshot(SCREENSHOT_PATH)
    st.session_state.messages.append({"role": "assistant", "content": answer})

def handle_submit_action(answer: str):
    uid = re.findall('uid="([^"]*)"', answer)[0]
    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
    replay_helper.add_submit_to_replay(get_replay_file_path(), get_current_state(), st.session_state.driver.current_url, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
    target_element.submit()
    time.sleep(2)
    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, DATA_DIR)
    st.session_state.html_and_bboxes_index += 1
    st.session_state.driver.save_screenshot(SCREENSHOT_PATH)
    st.session_state.messages.append({"role": "assistant", "content": answer})

def handle_change_action(answer: str):
    value, uid = re.findall('value="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
    replay_helper.add_change_to_replay(get_replay_file_path(), get_current_state(), st.session_state.driver.current_url, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, value, st.session_state.replay_data)
    target_element.send_keys(value)
    time.sleep(2)
    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, DATA_DIR)
    st.session_state.html_and_bboxes_index += 1
    st.session_state.driver.save_screenshot(SCREENSHOT_PATH)
    st.session_state.messages.append({"role": "assistant", "content": answer})

def get_current_state() -> str:
    return f"page-{st.session_state.html_and_bboxes_index - 1}-{0}.html" if st.session_state.html_and_bboxes_index > 0 else None

def get_replay_file_path() -> str:
    return os.path.join(DATA_DIR, 'replay.json')

def main():
    st.set_page_config(layout="wide")
    st.title("WebLINX Chatbot")

    initialize_session_state()

    st.sidebar.write("""
    This chatbot can help you navigate the web by performing actions such as clicking buttons, scrolling,
    submitting forms, etc., directly on the webpage. Start by typing your message below and pressing Enter.
    Click button Continue to continue proceeding with actions, and type 'Quit' to stop the chatbot.
    """)

    with st.sidebar:
        user_chat = st.chat_input("Type your message here:")
        if user_chat:
            handle_user_input(user_chat)
        
        if st.button("Continue"):
            handle_continue()

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("Chat History")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with col2:
        st.header("Web Page View")
        if os.path.exists(SCREENSHOT_PATH):
            st.image(SCREENSHOT_PATH, caption="Current Web Page", use_column_width=True)
        else:
            st.image('microphonecat.png', caption="No Screenshot Available", use_column_width=True)

if __name__ == "__main__":
    main()
