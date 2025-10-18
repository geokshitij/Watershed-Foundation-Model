import io
import os
import sys
import threading
import traceback
from contextlib import redirect_stdout
from flask import Flask, request, render_template_string, jsonify

# --- Dependencies Check ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import timm
    import tifffile
    from lightly.models.modules.masked_vision_transformer_timm import MaskedVisionTransformerTIMM
    from lightly.models.modules.masked_autoencoder_timm import MAEDecoderTIMM
    from lightly.transforms.mae_transform import MAETransform
    from lightly.models import utils
    # Import torchvision transforms directly for the new inference transform
    from torchvision import transforms as T
except ImportError as e:
    print(f"Error: A required library is missing: {e}")
    print("Please install all dependencies with:")
    print("pip install lightly timm flask tifffile numpy imagecodecs torchvision")
    sys.exit(1)

# --- Part 1: Custom Dataset for TIFF Catchment Images ---

class TiffCatchmentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.tiff', '.tif'))]
        if not self.image_files:
            raise RuntimeError(f"No .tiff or .tif files found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = tifffile.imread(img_path)
        except Exception as e:
            print(f"Warning: Could not read file {img_path}. Error: {e}", file=sys.stderr)
            return None

        if image.ndim == 2: image = np.expand_dims(image, axis=0)
        elif image.ndim == 3 and image.shape[2] < image.shape[0]: image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.from_numpy(image.astype(np.float32))
        if self.transform:
            image_tensor = self.transform(image_tensor)[0]
        return image_tensor

def collate_fn_skip_corrupt(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Part 2: MAE Model for Catchment Encoding ---

class CatchmentMAE(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', in_chans=1, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        vit = timm.create_model(vit_model_name, pretrained=False, in_chans=in_chans)
        self.encoder = MaskedVisionTransformerTIMM(vit=vit)
        
        decoder_embed_dim = 512
        num_patches = self.encoder.vit.patch_embed.num_patches
        self.decoder = MAEDecoderTIMM(
            num_patches=num_patches,
            patch_size=vit.patch_embed.patch_size[0],
            in_chans=in_chans,
            embed_dim=decoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
        )
        self.decoder.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        decoder_pos_embed = utils.get_2d_sincos_pos_embed(
            self.decoder.decoder_pos_embed.shape[-1], int(num_patches ** .5), cls_token=False
        )
        self.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        self.decoder_embed = nn.Linear(vit.embed_dim, decoder_embed_dim, bias=True)

    def forward_encoder(self, images, idx_keep):
        return self.encoder.encode(images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, ids_restore):
        x_encoded = self.decoder_embed(x_encoded)
        x_full = torch.cat([
            x_encoded,
            self.decoder.mask_token.repeat(x_encoded.shape[0], ids_restore.shape[1] - x_encoded.shape[1], 1)
        ], dim=1)
        x_unshuffled = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))
        x_unshuffled = x_unshuffled + self.decoder.decoder_pos_embed
        decoded = self.decoder.decode(x_unshuffled)
        return self.decoder.predict(decoded)

    def forward(self, images):
        batch_size = images.shape[0]
        num_patches = self.encoder.vit.patch_embed.num_patches
        num_masked = int(self.mask_ratio * num_patches)
        
        ids_shuffle = torch.randperm(num_patches, device=images.device).unsqueeze(0).expand(batch_size, -1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :num_patches - num_masked]
        
        latent = self.forward_encoder(images, idx_keep=ids_keep)
        
        predictions = self.forward_decoder(latent, ids_restore)
        
        return predictions, ids_restore

# --- Part 3: Training Logic Helper ---

def patchify(imgs, patch_size):
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
    return x
    
def get_inference_transform(image_size, norm_mean, norm_std):
    """Creates a transform for inference that resizes without cropping."""
    # Note: Using torchvision.transforms (T) imported at the top
    return T.Compose([
        T.Resize(size=(image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

# --- Part 4: Flask Web Application ---
app = Flask(__name__)
training_log_capture = io.StringIO()
training_status = {"running": False, "log": "", "error": False}
trained_model_config = {}

def run_training_thread(config):
    """Runs the training in a separate thread and captures its output."""
    global training_status, training_log_capture, trained_model_config
    
    training_log_capture = io.StringIO()
    training_status.update({"running": True, "log": "Starting training...\n", "error": False})

    try:
        with redirect_stdout(training_log_capture):
            print("--- Catchment MAE Training Configuration ---")
            for key, value in config.items(): print(f"{key}: {value}")
            print("------------------------------------------\n")

            normalize_dict = {"mean": config['norm_mean'], "std": config['norm_std']}
            transform = MAETransform(input_size=config['image_size'], normalize=normalize_dict)
            dataset = TiffCatchmentDataset(root_dir=config['input_dir'], transform=transform)
            if len(dataset) == 0:
                raise RuntimeError("No valid images found in the dataset directory.")
            
            dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], collate_fn=collate_fn_skip_corrupt)
            print(f"Found {len(dataset)} images. Starting training...")

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}\n")
            model = CatchmentMAE(vit_model_name=config['vit_model_name'], in_chans=config['in_chans'], mask_ratio=config['mask_ratio']).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
            
            patch_size = model.encoder.vit.patch_embed.patch_size[0]

            for epoch in range(config['max_epochs']):
                total_loss = 0
                num_batches = 0
                for batch in dataloader:
                    if batch is None: continue
                    images = batch.to(device)
                    
                    predictions, ids_restore = model(images)
                    target_patches = patchify(images, patch_size)
                    
                    # === THIS IS THE CORRECTED LOGIC ===
                    # 1. Determine which patches were masked
                    num_patches = model.encoder.vit.patch_embed.num_patches
                    num_masked = int(model.mask_ratio * num_patches)
                    num_visible = num_patches - num_masked
                    ids_masked = torch.argsort(ids_restore, dim=1)[:, num_visible:]
                    
                    # 2. Select the predicted patches that were masked
                    masked_preds = torch.gather(predictions, 1, ids_masked.unsqueeze(-1).expand(-1, -1, predictions.shape[2]))
                    
                    # 3. Select the original patches that were masked
                    masked_patches_gt = torch.gather(target_patches, 1, ids_masked.unsqueeze(-1).expand(-1, -1, target_patches.shape[2]))
                    
                    # 4. Calculate loss only on the masked patches
                    loss = F.mse_loss(masked_preds, masked_patches_gt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch {epoch+1}/{config['max_epochs']}, Average Loss: {avg_loss:.4f}")

            print("\n--- Training Finished Successfully ---")
            output_path = 'catchment_foundation_model.pth'
            torch.save(model.state_dict(), output_path)
            print(f"Full MAE model saved to: {output_path}")
            trained_model_config = config

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(traceback.format_exc())
        training_status["error"] = True
    finally:
        training_status["running"] = False
        training_status["log"] = training_log_capture.getvalue()

@app.route("/", methods=["GET"])
def index():
    """Renders the main HTML page."""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Train & Use Catchment Foundation Model</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background-color: #f8f9fa; color: #212529; }
            h1, h2 { color: #343a40; }
            .container { max-width: 900px; margin: auto; background: white; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }
            .form-group { margin-bottom: 1em; }
            label { display: block; margin-bottom: 0.5em; font-weight: 600; }
            input, select { width: 100%; padding: 0.75em; border: 1px solid #ced4da; border-radius: 4px; box-sizing: border-box; }
            small { color: #6c757d; }
            button { background-color: #007bff; color: white; padding: 0.75em 1.5em; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; transition: background-color 0.2s; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #6c757d; cursor: not-allowed; }
            .log-container { margin-top: 1em; background-color: #282c34; color: #abb2bf; padding: 1em; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; font-family: "SF Mono", "Fira Code", monospace; }
            .status { font-style: italic; color: #6c757d; }
            #inference-section { display: none; margin-top: 2em; border-top: 1px solid #dee2e6; padding-top: 2em; }
            #embedding-output { background-color: #e9ecef; padding: 1em; border-radius: 4px; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Train a Catchment Foundation Model (MAE)</h1>
            <p>This UI uses a Masked Autoencoder (MAE) to learn spatial representations from your catchment TIFF images.</p>
            <form id="training-form">
                <h2>1. Training</h2>
                <div class="grid">
                    <div class="form-group">
                        <label for="input_dir">Dataset Directory Path (TIFFs):</label>
                        <input type="text" id="input_dir" name="input_dir" placeholder="/path/to/your/catchments" required>
                    </div>
                    <div class="form-group">
                        <label for="vit_model_name">ViT Model:</label>
                        <select id="vit_model_name" name="vit_model_name">
                            <option value="vit_small_patch16_224" selected>ViT-Small</option>
                            <option value="vit_base_patch16_224">ViT-Base</option>
                            <option value="vit_large_patch16_224">ViT-Large</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="image_size">Image Size (pixels):</label>
                        <input type="number" id="image_size" name="image_size" value="224" min="32" required>
                    </div>
                    <div class="form-group">
                        <label for="in_chans">Number of Image Channels:</label>
                        <input type="number" id="in_chans" name="in_chans" value="1" min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="norm_mean">Normalization Mean:</label>
                        <input type="text" id="norm_mean" name="norm_mean" value="0.5" required>
                        <small>For 1 channel, use a single number. For 3, use '0.1,0.2,0.3'.</small>
                    </div>
                    <div class="form-group">
                        <label for="norm_std">Normalization Std:</label>
                        <input type="text" id="norm_std" name="norm_std" value="0.5" required>
                        <small>For 1 channel, use a single number. For 3, use '0.1,0.2,0.3'.</small>
                    </div>
                    <div class="form-group">
                        <label for="max_epochs">Number of Epochs:</label>
                        <input type="number" id="max_epochs" name="max_epochs" value="50" min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="batch_size">Batch Size:</label>
                        <input type="number" id="batch_size" name="batch_size" value="32" min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="mask_ratio">Mask Ratio:</label>
                        <input type="number" id="mask_ratio" name="mask_ratio" value="0.75" min="0.1" max="0.9" step="0.05" required>
                    </div>
                    <div class="form-group">
                        <label for="num_workers">Number of Workers:</label>
                        <input type="number" id="num_workers" name="num_workers" value="4" min="0" required>
                    </div>
                </div>
                <button type="submit" id="train-button">Start Training</button>
            </form>
            <div id="status-container">
                <p class="status">Status: <span id="status-text">Idle</span></p>
                <pre id="log-output" class="log-container">Training logs will appear here...</pre>
            </div>

            <div id="inference-section">
                <h2>2. Get Embedding from Trained Model</h2>
                <form id="inference-form">
                    <div class="form-group">
                        <label for="image_path">Path to a single TIFF image:</label>
                        <input type="text" id="image_path" name="image_path" placeholder="/path/to/single/catchment.tif" required>
                    </div>
                    <button type="submit" id="embed-button">Get Embedding</button>
                </form>
                <p class="status">Result:</p>
                <pre id="embedding-output" class="log-container">Embedding vector will appear here...</pre>
            </div>
        </div>

        <script>
            const trainForm = document.getElementById('training-form');
            const trainButton = document.getElementById('train-button');
            const statusText = document.getElementById('status-text');
            const logOutput = document.getElementById('log-output');
            const inferenceSection = document.getElementById('inference-section');
            let trainingIntervalId;

            trainForm.addEventListener('submit', function(event) {
                event.preventDefault();
                trainButton.disabled = true;
                statusText.textContent = 'Starting...';
                logOutput.textContent = 'Sending training request...';
                inferenceSection.style.display = 'none';

                const formData = new FormData(trainForm);
                fetch('/train', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        statusText.textContent = 'Running...';
                        trainingIntervalId = setInterval(fetchLogs, 2000);
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
                        logOutput.scrollTop = logOutput.scrollHeight;
                        if (!data.running) {
                            clearInterval(trainingIntervalId);
                            trainButton.disabled = false;
                            if (data.error) {
                                statusText.textContent = 'Error';
                                statusText.style.color = 'red';
                            } else {
                                statusText.textContent = 'Finished Successfully';
                                statusText.style.color = 'green';
                                inferenceSection.style.display = 'block';
                            }
                        } else {
                            statusText.textContent = 'Running...';
                            statusText.style.color = '#6c757d';
                        }
                    });
            }

            const inferenceForm = document.getElementById('inference-form');
            const embedButton = document.getElementById('embed-button');
            const embeddingOutput = document.getElementById('embedding-output');

            inferenceForm.addEventListener('submit', function(event) {
                event.preventDefault();
                embedButton.disabled = true;
                embeddingOutput.textContent = 'Generating embedding...';

                const formData = new FormData(inferenceForm);
                fetch('/embed', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        embeddingOutput.textContent = 'Embedding Shape: ' + data.shape + '\\n\\n';
                        embeddingOutput.textContent += 'Vector (first 10 values):\\n' + JSON.stringify(data.embedding.slice(0, 10), null, 2);
                    } else {
                        embeddingOutput.textContent = 'Error: ' + data.message;
                    }
                    embedButton.disabled = false;
                });
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

def parse_list_from_string(s):
    return [float(item.strip()) for item in s.split(',')]

@app.route("/train", methods=["POST"])
def train():
    """Handles the form submission to start a training job."""
    global training_status
    if training_status["running"]:
        return jsonify({"status": "error", "message": "A training job is already running."}), 400

    try:
        config = {
            "input_dir": request.form["input_dir"],
            "vit_model_name": request.form["vit_model_name"],
            "image_size": int(request.form["image_size"]),
            "in_chans": int(request.form["in_chans"]),
            "norm_mean": parse_list_from_string(request.form["norm_mean"]),
            "norm_std": parse_list_from_string(request.form["norm_std"]),
            "max_epochs": int(request.form["max_epochs"]),
            "batch_size": int(request.form["batch_size"]),
            "mask_ratio": float(request.form["mask_ratio"]),
            "num_workers": int(request.form["num_workers"]),
        }
        if len(config['norm_mean']) != config['in_chans'] or len(config['norm_std']) != config['in_chans']:
            raise ValueError("Number of normalization values must match number of image channels.")
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid form data: {e}"}), 400

    if not os.path.isdir(config['input_dir']):
        return jsonify({"status": "error", "message": f"Dataset directory not found: {config['input_dir']}"}), 400

    training_thread = threading.Thread(target=run_training_thread, args=(config,))
    training_thread.start()
    return jsonify({"status": "started"})

@app.route("/status", methods=["GET"])
def status():
    """Provides the current status and logs of the training job."""
    global training_status, training_log_capture
    training_status["log"] = training_log_capture.getvalue()
    return jsonify(training_status)

@app.route("/embed", methods=["POST"])
def embed():
    """Loads the trained model and returns an embedding for a single image."""
    global trained_model_config
    if not trained_model_config:
        return jsonify({"status": "error", "message": "No model has been trained yet."}), 400

    image_path = request.form.get("image_path")
    if not image_path or not os.path.isfile(image_path):
        return jsonify({"status": "error", "message": f"Image file not found: {image_path}"}), 400

    try:
        model = CatchmentMAE(
            vit_model_name=trained_model_config['vit_model_name'],
            in_chans=trained_model_config['in_chans']
        )
        state_dict = torch.load('catchment_foundation_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

        normalize_dict = {"mean": trained_model_config['norm_mean'], "std": trained_model_config['norm_std']}
        inference_transform = get_inference_transform(
            image_size=trained_model_config['image_size'],
            norm_mean=normalize_dict['mean'],
            norm_std=normalize_dict['std'],
        )
        
        image = tifffile.imread(image_path)
        if image.ndim == 2: image = np.expand_dims(image, axis=0)
        elif image.ndim == 3 and image.shape[2] < image.shape[0]: image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image.astype(np.float32))
        transformed_image = inference_transform(image_tensor).unsqueeze(0)

        with torch.no_grad():
            embedding = model.encoder(transformed_image)
        
        embedding_list = embedding.squeeze().tolist()
        return jsonify({
            "status": "success",
            "embedding": embedding_list,
            "shape": list(embedding.squeeze().shape)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("Starting Catchment MAE Training UI...")
    print("Please install imagecodecs if you have LZW compressed TIFFs: pip install imagecodecs")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)