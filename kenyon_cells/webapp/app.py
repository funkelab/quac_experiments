"""
Create a web application that has grid of items. 

Each item has a name and a set of checkboxes, for the following columns: 
- psd darker
- wider cleft
- longer cleft

The clicks are persisted to file, and the state is loaded from file on startup.
"""

import os
import json
from pathlib import Path
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)


# Define the custom filter
@app.template_filter("zip")
def zip_filter(a, b):
    return zip(a, b)


STATE_FILE = "annotation_state.json"
SOURCE_DIR = "assets/visualizations"
DESCRIPTION_FILE = "descriptions.json"

# The names of the items
ITEM_NAMES = list(f.name for f in Path(SOURCE_DIR).iterdir() if str(f).endswith(".png"))
IMAGE_URLS = [f"assets/visualizations/{item}.png" for item in ITEM_NAMES]
# Number of items
NUM_ITEMS = len(ITEM_NAMES)

# Annotation column names
# Load column names and descriptions from file
with open(DESCRIPTION_FILE, "r") as file:
    descriptions = json.load(file)
    COLUMN_NAMES = list(descriptions.keys())
    column_descriptions = list(descriptions.values())
    NUM_COLUMNS = len(COLUMN_NAMES)

# Define the directory containing the images
IMAGE_DIR = os.path.join(app.root_path, "assets", "visualizations")


@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, f"{filename}")


# The state of the grid
state = {}

# Load the state from file
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as file:
        state = json.load(file)

        # Check if the state has the correct number of items and columns
        if len(state) != NUM_ITEMS:
            state = {}
        else:
            for item in state.values():
                if len(item) != NUM_COLUMNS:
                    state = {}
                    break

# If the state is empty, initialize it
if not state:
    state = {item: [False] * NUM_COLUMNS for item in ITEM_NAMES}


def save_state():
    with open(STATE_FILE, "w") as file:
        json.dump(state, file)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Update the state based on the form data
        for item in ITEM_NAMES:
            for i in range(NUM_COLUMNS):
                checkbox_name = f"state_{item}_{i}"
                state[item][i] = checkbox_name in request.form

        # Debug: Print updated state
        print(f"Updated state: {state}")

        # Save the state to file
        save_state()

    return render_template(
        "index.html",
        items=state,
        columns=COLUMN_NAMES,
        descriptions=column_descriptions,
    )


if __name__ == "__main__":
    app.run(debug=True)