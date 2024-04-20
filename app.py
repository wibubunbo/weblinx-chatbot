import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import time

st.set_page_config(layout="wide")  # Set the layout to wide
st.title("WebLINX Chatbot")

# Description
st.sidebar.write("""
This chatbot can help you navigate the web by performing actions such as clicking buttons, scrolling,
submitting forms, etc., directly on the webpage. Start by typing your message below and pressing Enter.
Type 'Continue' to proceed with actions, and 'That's all' to stop the chatbot.
""")

# Initialize session state for messages if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'driver' not in st.session_state:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    st.session_state.driver = webdriver.Chrome(options=chrome_options)

# Function to take screenshots
def take_screenshot(driver, url="https://www.example.com"):
    driver.get(url)
    time.sleep(2)  # Wait for the page to load
    screenshot_path = 'screenshot.png'
    driver.save_screenshot(screenshot_path)
    return screenshot_path

# Sidebar for chat input
with st.sidebar:
    user_input = st.text_input("Type your message here:", key="chat_input")
    if user_input:
        if user_input.lower() == "that's all":
            st.session_state.driver.quit()
            del st.session_state.driver
            st.session_state.messages = []
        elif user_input.lower() != "continue":
            st.session_state.messages.append({"role": "user", "content": user_input})
            screenshot_path = take_screenshot(st.session_state.driver)
            st.session_state.messages.append({"role": "assistant", "content": "load(uid=444)"})

# Main area: Chat history and screenshot display
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Chat History")
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.info(message["content"])
            else:
                st.success(message["content"])

with col2:
    st.header("Web Page View")
    if os.path.exists('screenshot.png'):
        st.image('screenshot.png', caption="Latest Screenshot", use_column_width=True)
