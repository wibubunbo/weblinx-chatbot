import json
import datetime as dt


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
