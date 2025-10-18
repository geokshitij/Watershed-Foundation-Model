import io
import sys
import threading
from contextlib import redirect_stdout
from flask import Flask, request, render_template_string, jsonify

# Import the necessary training function from the lightly library
# Note: We use the internal '_train_cli' function for more direct control
from lightly.cli.train_cli import _train_cli
from lightly.cli.config.get_config import get_lightly_config

# --- Configuration ---

# A list of models supported by Lightly's command-line interface.
# This can be expanded if more models are supported in the future.
SUPPORTED_MODELS = [
    "resnet-9",
    "resnet-18",
    "resnet-34",
    "resnet-50",
    "resnet-101",
    "resnet-152",
]

# --- Flask Web Application ---

app = Flask(__name__)

# Global state to hold training logs and status
training_log_capture = io.StringIO()
training_status = {"running": False, "log": ""}


def run_training_thread(config):
    """
    This function runs the training in a separate thread and captures its output.
    """
    global training_status, training_log_capture
    
    # Reset log and set status
    training_log_capture = io.StringIO()
    training_status["running"] = True
    training_status["log"] = "Starting training...\n"

    try:
        # Redirect stdout to capture print statements from the training function
        with redirect_stdout(training_log_capture):
            print("--- Lightly Training Configuration ---")
            for key, value in config.items():
                print(f"{key}: {value}")
            print("------------------------------------")
            
            # Call the training function
            # is_cli_call=False prevents it from trying to parse command-line args
            checkpoint_path = _train_cli(config, is_cli_call=False)
            
            print(f"\n--- Training Finished ---")
            print(f"Checkpoint saved to: {checkpoint_path}")

    except Exception as e:
        # Capture and log any errors during training
        print(f"\n--- An Error Occurred ---")
        print(str(e))
    finally:
        # Update status when finished
        training_status["running"] = False
        training_status["log"] = training_log_capture.getvalue()


@app.route("/", methods=["GET"])
def index():
    """
    Renders the main HTML page with the training form.
    """
    # HTML and CSS for the user interface, embedded as a string
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lightly Training UI</title>
        <style>
            body { font-family: sans-serif; margin: 2em; background-color: #f4f4f9; color: #333; }
            h1 { color: #4a4a4a; }
            .container { max-width: 800px; margin: auto; background: white; padding: 2em; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 1.5em; }
            label { display: block; margin-bottom: 0.5em; font-weight: bold; }
            input[type="text"], input[type="number"], select {
                width: 100%;
                padding: 0.8em;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 0.8em 1.2em;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1em;
            }
            button:disabled { background-color: #ccc; cursor: not-allowed; }
            .log-container {
                margin-top: 2em;
                background-color: #282c34;
                color: #abb2bf;
                padding: 1em;
                border-radius: 4px;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 500px;
                overflow-y: auto;
            }
            .status { font-style: italic; color: #555; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Train a Self-Supervised Model with Lightly</h1>
            <form id="training-form">
                <div class="form-group">
                    <label for="input_dir">Dataset Directory Path:</label>
                    <input type="text" id="input_dir" name="input_dir" placeholder="/path/to/your/images" required>
                </div>
                <div class="form-group">
                    <label for="model_name">Backbone Model:</label>
                    <select id="model_name" name="model_name">
                        {% for model in models %}
                            <option value="{{ model }}" {% if model == 'resnet-18' %}selected{% endif %}>{{ model }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label for="max_epochs">Number of Epochs:</label>
                    <input type="number" id="max_epochs" name="max_epochs" value="10" min="1" required>
                </div>
                <div class="form-group">
                    <label for="batch_size">Batch Size:</label>
                    <input type="number" id="batch_size" name="batch_size" value="64" min="1" required>
                </div>
                <div class="form-group">
                    <label for="num_workers">Number of Workers:</label>
                    <input type="number" id="num_workers" name="num_workers" value="-1" min="-1" required>
                    <small>Set to -1 to use all available CPU cores.</small>
                </div>
                <button type="submit" id="train-button">Start Training</button>
            </form>
            <div id="status-container">
                <p class="status">Status: <span id="status-text">Idle</span></p>
                <pre id="log-output" class="log-container">Training logs will appear here...</pre>
            </div>
        </div>

        <script>
            const form = document.getElementById('training-form');
            const trainButton = document.getElementById('train-button');
            const statusText = document.getElementById('status-text');
            const logOutput = document.getElementById('log-output');
            let intervalId;

            form.addEventListener('submit', function(event) {
                event.preventDefault();
                trainButton.disabled = true;
                statusText.textContent = 'Starting...';
                logOutput.textContent = 'Sending training request to the server...';

                const formData = new FormData(form);
                fetch('/train', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        statusText.textContent = 'Running...';
                        // Start polling for logs
                        intervalId = setInterval(fetchLogs, 2000);
                    } else {
                        logOutput.textContent = 'Error: ' + data.message;
                        trainButton.disabled = false;
                        statusText.textContent = 'Error';
                    }
                });
            });

            function fetchLogs() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        logOutput.textContent = data.log;
                        // Scroll to the bottom of the log
                        logOutput.scrollTop = logOutput.scrollHeight;
                        if (!data.running) {
                            clearInterval(intervalId);
                            trainButton.disabled = false;
                            statusText.textContent = 'Finished';
                        } else {
                            statusText.textContent = 'Running...';
                        }
                    });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, models=SUPPORTED_MODELS)


@app.route("/train", methods=["POST"])
def train():
    """
    Handles the form submission to start a training job.
    """
    global training_status
    if training_status["running"]:
        return jsonify({"status": "error", "message": "A training job is already running."}), 400

    # Get default configuration from lightly
    config = get_lightly_config()

    # Update config with user inputs from the form
    try:
        config["input_dir"] = request.form["input_dir"]
        config["model"]["name"] = request.form["model_name"]
        config["trainer"]["max_epochs"] = int(request.form["max_epochs"])
        config["loader"]["batch_size"] = int(request.form["batch_size"])
        config["loader"]["num_workers"] = int(request.form["num_workers"])
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid form data: {e}"}), 400

    # Start the training in a background thread
    training_thread = threading.Thread(target=run_training_thread, args=(config,))
    training_thread.start()

    return jsonify({"status": "started"})


@app.route("/status", methods=["GET"])
def status():
    """
    Provides the current status and logs of the training job.
    """
    global training_status, training_log_capture
    # Update the log in the status dictionary before sending
    training_status["log"] = training_log_capture.getvalue()
    return jsonify(training_status)


if __name__ == "__main__":
    print("Starting Lightly Training UI...")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    # The reloader is disabled to prevent issues with the background training thread
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
