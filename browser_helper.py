from selenium import webdriver 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.chrome.options import Options 
import time  
import json
import uuid
import os

def initialize_browser(viewport_width, viewport_height): 
    chrome_options = Options()
    chrome_options.add_argument("--headless") # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(viewport_width, viewport_height)
    return driver


def save_pages_bbox(driver, html_and_bboxes_index, file_path):
    elements = driver.find_elements(By.XPATH, "//*")
    element_data = {}
    for element in elements:
        uid = element.get_attribute('data-webtasks-id')
        if not uid:
            uid = str(uuid.uuid4())[:18]
            driver.execute_script("arguments[0].setAttribute('data-webtasks-id', arguments[1]);", element, uid)
                
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
    pages_directory = os.path.join(file_path, 'pages')
    bboxes_directory = os.path.join(file_path, 'bboxes')
    html_file_path = os.path.join(pages_directory, f'page-{html_and_bboxes_index}-{0}.html')
    bboxes_file_path = os.path.join(bboxes_directory, f'bboxes-{html_and_bboxes_index}.json')
    # Save HTML content to pages folder
    with open(html_file_path, 'w') as f:
        f.write(driver.page_source)
    # Save the bounding boxes to bboxes folder
    with open(bboxes_file_path, 'w') as f:
        json.dump(element_data, f)

def get_element_rect_and_attr(driver, uid):
    # Obtain the bounding rectangle of the target element                
    target_element = driver.find_element(By.XPATH, f'//*[@data-webtasks-id="{uid}"]')
    # Obtain the bounding rectangle of the target element                
    attrs = ["class", "title", "href", "aria-label", "d", "src"]
    attrs_dict = {}
    for attr in attrs:
        if target_element.get_attribute(attr) is not None:
            attrs_dict[attr] = target_element.get_attribute(attr)
    attrs_dict["data-webtasks-id"] = uid
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
    return target_element, attrs_dict, bbox
