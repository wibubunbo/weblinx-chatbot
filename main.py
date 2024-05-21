import os
import re
import datetime as dt
import weblinx as wl
import time
import streamlit as st
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

# Modularized Functions
import replay_helper
import model_helper
import browser_helper

def main():
    data_dir = './live_data'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'pages'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'bboxes'), exist_ok=True)
    replay_file_path = os.path.join(data_dir, 'replay.json')

    # Delete screenshot if it exists
    if os.path.exists('./screenshot.png'):
        os.remove('./screenshot.png')

    # Set the window size
    viewport_width = 1600
    viewport_height = 900

    st.set_page_config(layout="wide")  # Set the layout to wide
    st.title("WebLINX Chatbot")

    # Description
    st.sidebar.write("""
    This chatbot can help you navigate the web by performing actions such as clicking buttons, scrolling,
    submitting forms, etc., directly on the webpage. Start by typing your message below and pressing Enter.
    Type 'Continue' to continue proceeding with actions, and 'Quit' to stop the chatbot.
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
        st.session_state.driver = browser_helper.initialize_browser(viewport_width, viewport_height)

    if 'replay_data' not in st.session_state:
        st.session_state.replay_data = {"data": []}
        with open(replay_file_path, 'w') as f:
            f.write('{"data": []}\n')

    if 'model_init' not in st.session_state:
        st.session_state.format_intent_input, st.session_state.format_intent, st.session_state.build_prompt_records_fn, st.session_state.tokenizer, st.session_state.template_tokenizer = model_helper.load_formatters()

    with st.sidebar:
        if user_chat := st.chat_input("Type your message here:"):
            if user_chat == "Quit":
                replay_helper.add_temporary_action_to_replay(replay_file_path, None, viewport_height, viewport_width, st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.turn_index += 1
                replay_helper.add_say_to_replay(replay_file_path, "instructor", "That's all", st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.driver.quit()
                st.session_state.messages = []
            elif user_chat != "Continue":
                st.session_state.messages.append({"role": "user", "content": user_chat})
                replay_helper.add_temporary_action_to_replay(replay_file_path, None, viewport_height, viewport_width, st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.turn_index += 1
                replay_helper.add_say_to_replay(replay_file_path, "instructor", user_chat, st.session_state.initial_timestamp, st.session_state.replay_data)

                state = f"page-{st.session_state.html_and_bboxes_index - 1}-{0}.html" if st.session_state.html_and_bboxes_index > 0 else None
                replay_helper.add_temporary_action_to_replay(replay_file_path, state, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.turn_index += 1

                demo = wl.Demonstration(data_dir, base_dir='.')
                replay = wl.Replay.from_demonstration(demo)
                current_turn = wl.Turn.from_replay(replay, st.session_state.turn_index)

                answer = model_helper.predict_answer(state, current_turn, replay, st.session_state.format_intent_input, st.session_state.format_intent, st.session_state.build_prompt_records_fn, st.session_state.tokenizer, st.session_state.template_tokenizer)
            
                print(answer)
                
                if answer.startswith('say'):
                    replay_helper.add_say_to_replay(replay_file_path, "navigator", answer.split('utterance="')[1][:-2], st.session_state.initial_timestamp, st.session_state.replay_data)
                    st.session_state.messages.append({"role": "assistant", "content": answer.split('utterance="')[1][:-2]})

                elif answer.startswith('load'):
                    url = re.findall('url="([^"]*)"', answer)[0]
                    st.session_state.driver.get(url)
                    time.sleep(3)
                    # Generate unique IDs for each element in the DOM
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    replay_helper.add_load_to_replay(replay_file_path, state, url, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('click'):
                    uid = re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_click_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                    target_element.click()
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('text_input'):
                    text, uid = re.findall('text="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_textInput_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, text, st.session_state.replay_data)
                    target_element.send_keys(text)
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('scroll'):
                    scrollX, scrollY = re.findall('x="([^"]*)"', answer)[0], re.findall('y="([^"]*)"', answer)[0]
                    st.session_state.driver.execute_script(f"window.scrollTo({scrollX}, {scrollY});")
                    time.sleep(3)
                    replay_helper.add_scroll_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, scrollX, scrollY, st.session_state.replay_data)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('submit'):
                    uid = re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_submit_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                    target_element.submit()
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('change'):
                    value, uid = re.findall('value="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_change_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, value, st.session_state.replay_data)
                    target_element.send_keys(value)
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                else:
                    raise ValueError(f"Invalid action: {answer}")
                
            elif user_chat == "Continue":
                state = f"page-{st.session_state.html_and_bboxes_index - 1}-{0}.html" if st.session_state.html_and_bboxes_index > 0 else None
                replay_helper.add_temporary_action_to_replay(replay_file_path, state, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
                st.session_state.turn_index += 1

                demo = wl.Demonstration(data_dir, base_dir='.')
                replay = wl.Replay.from_demonstration(demo)
                current_turn = wl.Turn.from_replay(replay, st.session_state.turn_index)

                answer = model_helper.predict_answer(state, current_turn, replay, st.session_state.format_intent_input, st.session_state.format_intent, st.session_state.build_prompt_records_fn, st.session_state.tokenizer, st.session_state.template_tokenizer)
                print(answer)

                if answer.startswith('say'):
                    replay_helper.add_say_to_replay(replay_file_path, "navigator", answer.split('utterance="')[1][:-2], st.session_state.initial_timestamp, st.session_state.replay_data)
                    st.session_state.messages.append({"role": "assistant", "content": answer.split('utterance="')[1][:-2]})
                elif answer.startswith('load'):
                    url = re.findall('url="([^"]*)"', answer)[0]
                    st.session_state.driver.get(url)
                    time.sleep(3)
                    # Generate unique IDs for each element in the DOM
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    
                    replay_helper.add_load_to_replay(replay_file_path, state, url, viewport_width, viewport_height, st.session_state.initial_timestamp, st.session_state.replay_data)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('click'):
                    uid = re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_click_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                    target_element.click()
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('text_input'):
                    text, uid = re.findall('text="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_textInput_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, text, st.session_state.replay_data)
                    target_element.send_keys(text)
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('scroll'):
                    scrollX, scrollY = re.findall('x="([^"]*)"', answer)[0], re.findall('y="([^"]*)"', answer)[0]
                    st.session_state.driver.execute_script(f"window.scrollTo({scrollX}, {scrollY});")
                    time.sleep(3)
                    replay_helper.add_scroll_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, scrollX, scrollY, st.session_state.replay_data)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('submit'):
                    uid = re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_submit_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, st.session_state.replay_data)
                    target_element.submit()
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
                    st.session_state.html_and_bboxes_index += 1
                    st.session_state.driver.save_screenshot('./screenshot.png')
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                elif answer.startswith('change'):
                    value, uid = re.findall('value="([^"]*)"', answer)[0], re.findall('uid="([^"]*)"', answer)[0]
                    target_element, attrs_dict, bbox = browser_helper.get_element_rect_and_attr(st.session_state.driver, uid)
                    replay_helper.add_change_to_replay(replay_file_path, state, st.session_state.driver.current_url, viewport_width, viewport_height, st.session_state.initial_timestamp, round(bbox["x"]), round(bbox["y"]), attrs_dict, bbox, target_element.tag_name, value, st.session_state.replay_data)
                    target_element.send_keys(value)
                    time.sleep(3)
                    browser_helper.save_pages_bbox(st.session_state.driver, st.session_state.html_and_bboxes_index, data_dir)
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

if __name__ == "__main__":
    main()
