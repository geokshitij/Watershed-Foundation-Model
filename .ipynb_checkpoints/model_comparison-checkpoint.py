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
    import lightly
    from lightly.loss import BarlowTwinsLoss, NTXentLoss
    from lightly.data.multi_view_collate import MultiViewCollate
    from lightly.models.modules.heads import BarlowTwinsProjectionHead, SimCLRProjectionHead
    from lightly.models.modules.masked_vision_transformer_timm import MaskedVisionTransformerTIMM
    from lightly.models.modules.masked_autoencoder_timm import MAEDecoderTIMM
    from lightly.transforms import MAETransform, SimCLRTransform
    from lightly.transforms.multi_view_transform import MultiViewTransform
    from lightly.transforms.utils import IMAGENET_NORMALIZE
    from lightly.transforms.gaussian_blur import GaussianBlur
    from lightly.transforms.solarize import RandomSolarization
    from lightly.models import utils
    from torchvision import transforms as T
    import pytorch_lightning as pl
except ImportError as e:
    print(f"Error: A required library is missing: {e}")
    print("Please install all dependencies with:")
    print("pip install lightly timm flask tifffile numpy imagecodecs torchvision pytorch-lightning")
    sys.exit(1)

# --- Part 1: Custom Dataset for TIFF Catchment Images ---

class TiffCatchmentDataset(Dataset):
    def __init__(self, root_dir, transform=None, for_contrastive=False):
        self.root_dir = root_dir
        self.transform = transform
        self.for_contrastive = for_contrastive
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.tiff', '.tif'))]
        if not self.image_files:
            raise RuntimeError(f"No .tiff or .tif files found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = tifffile.imread(img_path)
            
            if self.for_contrastive:
                if image.ndim == 2: image = np.stack((image,)*3, axis=-1)
                if image.dtype != np.uint8:
                    max_val = image.max()
                    if max_val > 1: image = (image / max_val * 255)
                    else: image = (image * 255)
                    image = image.astype(np.uint8)
                image = T.ToPILImage()(image)
            else:
                if image.ndim == 2: image = np.expand_dims(image, axis=0)
                elif image.ndim == 3 and image.shape[2] < image.shape[0]: image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image.astype(np.float32))
        except Exception as e:
            print(f"Warning: Could not read or convert file {img_path}. Error: {e}", file=sys.stderr)
            return None, None, None

        if self.transform:
            image = self.transform(image)
            
        return image, 0, self.image_files[idx]

def collate_fn_skip_corrupt_multiview(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: return None
    return MultiViewCollate()(batch)

def collate_fn_skip_corrupt_singleview(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: return None, None, None
    images = torch.stack([item[0][0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    fnames = [item[2] for item in batch]
    return images, labels, fnames

class BarlowTwinsTransform(MultiViewTransform):
    def __init__(self, input_size=224, normalize=IMAGENET_NORMALIZE):
        view_transform = T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(kernel_size=23, sigmas=(0.1, 2.0))], p=1.0),
            T.RandomSolarization(threshold=128, p=0.2),
            T.ToTensor(),
            T.Normalize(mean=normalize['mean'], std=normalize['std'])
        ])
        super().__init__(transforms=[view_transform, view_transform])

# --- Part 2: Model Definitions ---
# ... (Model definitions are correct and unchanged, so they are omitted for brevity)
# (Paste MAEModel and ContrastiveModel classes here from the previous response)
class MAEModel(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', in_chans=1, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        vit = timm.create_model(vit_model_name, pretrained=False, in_chans=in_chans)
        self.encoder = MaskedVisionTransformerTIMM(vit=vit)
        decoder_embed_dim = 512
        num_patches = self.encoder.vit.patch_embed.num_patches
        self.decoder = MAEDecoderTIMM(
            num_patches=num_patches, patch_size=vit.patch_embed.patch_size[0], in_chans=in_chans,
            embed_dim=decoder_embed_dim, decoder_embed_dim=decoder_embed_dim,
            decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0
        )
        self.decoder.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        decoder_pos_embed = utils.get_2d_sincos_pos_embed(
            self.decoder.decoder_pos_embed.shape[-1], int(num_patches ** .5), cls_token=False
        )
        self.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.decoder_embed = nn.Linear(vit.embed_dim, decoder_embed_dim, bias=True)

    def forward(self, images):
        batch_size = images.shape[0]
        num_patches = self.encoder.vit.patch_embed.num_patches
        num_masked = int(self.mask_ratio * num_patches)
        ids_shuffle = torch.randperm(num_patches, device=images.device).unsqueeze(0).expand(batch_size, -1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_patches - num_masked]
        latent = self.encoder.encode(images, idx_keep=ids_keep)
        latent = self.decoder_embed(latent)
        x_full = torch.cat([latent, self.decoder.mask_token.repeat(latent.shape[0], ids_restore.shape[1] - latent.shape[1], 1)], dim=1)
        x_unshuffled = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))
        x_unshuffled = x_unshuffled + self.decoder.decoder_pos_embed
        decoded = self.decoder.decode(x_unshuffled)
        predictions = self.decoder.predict(decoded)
        return predictions, ids_restore

class ContrastiveModel(nn.Module):
    def __init__(self, resnet_model_name='resnet18', method='simclr'):
        super().__init__()
        resnet = timm.create_model(resnet_model_name, pretrained=False, num_classes=0, in_chans=3)
        self.backbone = resnet
        num_ftrs = self.backbone.num_features
        if method == 'simclr': self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs, 128)
        elif method == 'barlow_twins': self.projection_head = BarlowTwinsProjectionHead(num_ftrs, 8192, 8192)
        else: raise ValueError(f"Unknown contrastive method: {method}")

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(features)


# --- Part 3: PyTorch Lightning Training Module ---

class LightningTrainer(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.method = config['method']
        if self.method == 'mae':
            self.criterion = nn.MSELoss()
            self.patch_size = self.model.encoder.vit.patch_embed.patch_size[0]
        elif self.method == 'simclr': self.criterion = NTXentLoss(temperature=0.1)
        elif self.method == 'barlow_twins': self.criterion = BarlowTwinsLoss()

    def training_step(self, batch, batch_idx):
        if self.method == 'mae':
            images, _, _ = batch
            images = images[0] if isinstance(images, list) else images
            
            # === FIX for Batch Norm Error ===
            if images.shape[0] < 2:
                self.log('train_loss', 0.0) # Log a zero loss
                print("Skipping batch with size 1 to avoid BatchNorm error.")
                return None # Skip update
            
            predictions, ids_restore = self.model(images)
            patches = patchify(images, self.patch_size)
            loss = self.criterion(predictions, patches)
        else: # Contrastive methods
            views, _, _ = batch
            # === FIX for Batch Norm Error ===
            if views[0].shape[0] < 2:
                self.log('train_loss', 0.0)
                print("Skipping batch with size 1 to avoid BatchNorm error.")
                return None
            
            z1 = self.model(views[0])
            z2 = self.model(views[1])
            loss = self.criterion(z1, z2)
            
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1.5e-4, weight_decay=0.05)


def patchify(imgs, patch_size):
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
    return x

# --- Part 4: Flask Web Application ---
app = Flask(__name__)
# ... (The rest of the Flask app is unchanged, paste it here from the previous response)
# ... (The HTML/JS is also unchanged)
training_log_capture = io.StringIO()
training_status = {"running": False, "log": "", "error": False}
trained_model_config = {}

def run_training_thread(config):
    global training_status, training_log_capture, trained_model_config
    training_log_capture = io.StringIO()
    training_status.update({"running": True, "log": "Starting training...\n", "error": False})
    try:
        with redirect_stdout(training_log_capture):
            print("--- Training Configuration ---")
            for key, value in config.items(): print(f"{key}: {value}")
            print("----------------------------\n")

            is_contrastive = config['method'] != 'mae'
            if is_contrastive:
                transform = SimCLRTransform(input_size=config['image_size']) if config['method'] == 'simclr' else BarlowTwinsTransform(input_size=config['image_size'])
                collate = collate_fn_skip_corrupt_multiview
                in_chans = 3
            else:
                transform = MAETransform(input_size=config['image_size'])
                # FIX: Use the correct single-view collate function for MAE
                collate = collate_fn_skip_corrupt_singleview
                in_chans = config['in_chans']

            dataset = TiffCatchmentDataset(root_dir=config['input_dir'], transform=transform, for_contrastive=is_contrastive)
            if len(dataset) == 0: raise RuntimeError("No valid images found.")
            
            # Add drop_last=True to avoid single-item batches
            dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], collate_fn=collate, drop_last=True)
            if len(dataloader) == 0:
                print("\nWARNING: Dataset is smaller than the batch size. No batches to process.")
                print("Please add more images to your dataset or reduce the batch size.")
                raise RuntimeError("Training cannot proceed with zero batches.")

            print(f"Found {len(dataset)} images. Starting training with {len(dataloader)} batches per epoch...")

            if config['architecture'] == 'vit':
                model = MAEModel(vit_model_name=config['model_name'], in_chans=in_chans, mask_ratio=config['mask_ratio'])
            else:
                model = ContrastiveModel(resnet_model_name=config['model_name'], method=config['method'])
            
            pl_model = LightningTrainer(model, config)
            trainer = pl.Trainer(max_epochs=config['max_epochs'], accelerator='auto', devices=1)
            trainer.fit(pl_model, dataloader)

            print("\n--- Training Finished Successfully ---")
            output_path = f"{config['method']}_foundation_model.pth"
            torch.save(model.state_dict(), output_path)
            print(f"Model saved to: {output_path}")
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
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Train Foundation Model</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background-color: #f8f9fa; color: #212529; }
            h1, h2 { color: #343a40; }
            .container { max-width: 900px; margin: auto; background: white; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }
            .form-group { margin-bottom: 1em; }
            label { display: block; margin-bottom: 0.5em; font-weight: 600; }
            input, select { width: 100%; padding: 0.75em; border: 1px solid #ced4da; border-radius: 4px; box-sizing: border-box; }
            small { color: #6c757d; }
            button { background-color: #007bff; color: white; padding: 0.75em 1.5em; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; }
            button:disabled { background-color: #6c757d; cursor: not-allowed; }
            .log-container { margin-top: 1em; background-color: #282c34; color: #abb2bf; padding: 1em; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; font-family: "SF Mono", monospace; }
            .status { font-style: italic; color: #6c757d; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Train a Foundation Model</h1>
            <p>Choose an architecture and self-supervised method to train on your TIFF images.</p>
            <form id="training-form">
                <h2>1. Model & Data Configuration</h2>
                <div class="grid">
                    <div class="form-group">
                        <label for="architecture">Architecture:</label>
                        <select id="architecture" name="architecture">
                            <option value="vit" selected>Transformer (ViT)</option>
                            <option value="resnet">Convolutional (ResNet)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="method">Method:</label>
                        <select id="method" name="method"></select>
                    </div>
                </div>
                <div class="grid">
                     <div class="form-group">
                        <label for="input_dir">Dataset Directory Path (TIFFs):</label>
                        <input type="text" id="input_dir" name="input_dir" placeholder="/path/to/your/catchments" required>
                    </div>
                    <div class="form-group">
                        <label for="model_name">Backbone Model:</label>
                        <select id="model_name" name="model_name"></select>
                    </div>
                    <div class="form-group">
                        <label for="image_size">Image Size (pixels):</label>
                        <input type="number" id="image_size" name="image_size" value="224" min="32" required>
                    </div>
                    <div class="form-group param-group mae-param">
                        <label for="in_chans">Number of Image Channels (for MAE):</label>
                        <input type="number" id="in_chans" name="in_chans" value="1" min="1">
                    </div>
                    <div class="form-group param-group mae-param">
                        <label for="mask_ratio">Mask Ratio:</label>
                        <input type="number" id="mask_ratio" name="mask_ratio" value="0.75" min="0.1" max="0.9" step="0.05">
                    </div>
                </div>

                <h2>2. Training Parameters</h2>
                <div class="grid">
                    <div class="form-group">
                        <label for="max_epochs">Number of Epochs:</label>
                        <input type="number" id="max_epochs" name="max_epochs" value="50" min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="batch_size">Batch Size:</label>
                        <input type="number" id="batch_size" name="batch_size" value="32" min="1" required>
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
        </div>

        <script>
            const archSelect = document.getElementById('architecture');
            const methodSelect = document.getElementById('method');
            const modelSelect = document.getElementById('model_name');

            const options = {
                vit: {
                    methods: ['mae'],
                    models: ['vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224']
                },
                resnet: {
                    methods: ['simclr', 'barlow_twins'],
                    models: ['resnet18', 'resnet34', 'resnet50']
                }
            };

            function updateForm() {
                const arch = archSelect.value;
                const availableMethods = options[arch].methods;
                const availableModels = options[arch].models;

                methodSelect.innerHTML = '';
                availableMethods.forEach(method => {
                    const option = document.createElement('option');
                    option.value = method;
                    option.textContent = method.charAt(0).toUpperCase() + method.slice(1).replace('_', ' ');
                    methodSelect.appendChild(option);
                });

                modelSelect.innerHTML = '';
                availableModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });

                document.querySelectorAll('.param-group').forEach(el => el.classList.add('hidden'));
                document.querySelectorAll(`.${methodSelect.value}-param`).forEach(el => el.classList.remove('hidden'));
            }
            
            archSelect.addEventListener('change', updateForm);
            methodSelect.addEventListener('change', updateForm);
            
            document.addEventListener('DOMContentLoaded', updateForm);

            const trainForm = document.getElementById('training-form');
            const trainButton = document.getElementById('train-button');
            const statusText = document.getElementById('status-text');
            const logOutput = document.getElementById('log-output');
            let trainingIntervalId;

            trainForm.addEventListener('submit', function(event) {
                event.preventDefault();
                trainButton.disabled = true;
                statusText.textContent = 'Starting...';
                logOutput.textContent = 'Sending training request...';

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
                            statusText.textContent = data.error ? 'Error' : 'Finished with Error';
                            statusText.style.color = data.error ? 'red' : 'green';
                        } else {
                            statusText.textContent = 'Running...';
                            statusText.style.color = '#6c757d';
                        }
                    });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route("/train", methods=["POST"])
def train():
    global training_status
    if training_status["running"]:
        return jsonify({"status": "error", "message": "A training job is already running."}), 400
    try:
        config = {
            "input_dir": request.form["input_dir"],
            "architecture": request.form["architecture"],
            "method": request.form["method"],
            "model_name": request.form["model_name"],
            "image_size": int(request.form["image_size"]),
            "max_epochs": int(request.form["max_epochs"]),
            "batch_size": int(request.form["batch_size"]),
            "num_workers": int(request.form["num_workers"]),
            "mask_ratio": float(request.form.get("mask_ratio", 0.75)),
            "in_chans": int(request.form.get("in_chans", 1)),
        }
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid form data: {e}"}), 400

    if not os.path.isdir(config['input_dir']):
        return jsonify({"status": "error", "message": f"Dataset directory not found: {config['input_dir']}"}), 400

    training_thread = threading.Thread(target=run_training_thread, args=(config,))
    training_thread.start()
    return jsonify({"status": "started"})


@app.route("/status", methods=["GET"])
def status():
    global training_status, training_log_capture
    training_status["log"] = training_log_capture.getvalue()
    return jsonify(training_status)


if __name__ == "__main__":
    print("Starting Model Comparison UI...")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)