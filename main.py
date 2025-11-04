import io
import os
import sys
import json
import threading
import traceback
import random
import math
import re
import base64
import argparse
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
    from lightly.models import utils
    from torchvision import transforms as T
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend for server
except ImportError as e:
    print(f"Error: A required library is missing: {e}")
    print("Please install all dependencies with:")
    print("pip install torch torchvision timm lightly flask tifffile numpy imagecodecs matplotlib")
    sys.exit(1)

# --- Part 1: Data Splitting, Custom Dataset & Transform ---

def create_data_splits(root_dir, output_dir, train_split=0.8, val_split=0.1, seed=42):
    print("Creating data splits...")
    all_files = [f for f in os.listdir(root_dir) if f.endswith(('.tiff', '.tif'))]
    if not all_files:
        raise RuntimeError(f"No .tiff or .tif files found in {root_dir} to create splits.")
    random.seed(seed)
    random.shuffle(all_files)
    num_files = len(all_files)
    test_split_ratio = max(0.0, 1.0 - train_split - val_split)
    num_val = int(num_files * val_split)
    num_test = int(num_files * test_split_ratio)
    if num_files > 5:
        if val_split > 0 and num_val == 0: num_val = 1
        if test_split_ratio > 0 and num_test == 0: num_test = 1
    num_train = num_files - num_val - num_test
    train_files = all_files[:num_train]
    val_files = all_files[num_train : num_train + num_val]
    test_files = all_files[num_train + num_val:]
    for name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        path = os.path.join(output_dir, f"{name}.txt")
        with open(path, 'w') as f: f.write('\n'.join(file_list))
        print(f"  - {len(file_list)} files for {name} set. List saved to {path}")
    print("Data splits created successfully.\n")
    return train_files, val_files, test_files

class CustomCatchmentTransform:
    def __init__(self, image_size, norm_mean, norm_std):
        self.transform = T.Compose([
            T.Resize(size=(image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(mean=norm_mean, std=norm_std),
        ])
    def __call__(self, image_tensor): return [self.transform(image_tensor)]

class TiffCatchmentDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None, nodata_value=-9999.0, image_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.nodata_value = nodata_value
        self.image_size = image_size
        self.image_files = file_list
        if not self.image_files: print(f"Warning: Dataset for directory {root_dir} initialized with an empty file list.", file=sys.stderr)
        self.mask_transform = T.Resize(size=(image_size, image_size), interpolation=T.InterpolationMode.NEAREST)
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = tifffile.imread(img_path).astype(np.float32)
            original_shape = image.shape[-2:]
            valid_mask_np = (image != self.nodata_value)
            if valid_mask_np.any():
                mean_val = image[valid_mask_np].mean()
                image[~valid_mask_np] = mean_val
            else: image.fill(0)
        except Exception as e:
            print(f"Warning: Could not read file {img_path}. Error: {e}", file=sys.stderr)
            return None, None, None
        if image.ndim == 2: image = np.expand_dims(image, axis=0)
        image_tensor = torch.from_numpy(image)
        if self.transform: image_tensor = self.transform(image_tensor)[0]
        mask_tensor = torch.from_numpy(valid_mask_np.astype(np.float32)).unsqueeze(0)
        mask_tensor = self.mask_transform(mask_tensor)
        metadata = torch.tensor(original_shape, dtype=torch.float32)
        return image_tensor, metadata, mask_tensor

def collate_fn_skip_corrupt(batch):
    batch = list(filter(lambda x: x is not None and all(item is not None for item in x), batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Part 2: Scale-Aware MAE Model ---
class CatchmentMAE(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', in_chans=1, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        vit = timm.create_model(vit_model_name, pretrained=False, in_chans=in_chans)
        self.encoder = MaskedVisionTransformerTIMM(vit=vit)
        decoder_embed_dim = 512
        num_patches = self.encoder.vit.patch_embed.num_patches
        self.decoder = MAEDecoderTIMM(
            num_patches=num_patches, patch_size=vit.patch_embed.patch_size[0], in_chans=in_chans,
            embed_dim=decoder_embed_dim, decoder_embed_dim=decoder_embed_dim, decoder_depth=8,
            decoder_num_heads=16, mlp_ratio=4.0,
        )
        self.decoder.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        decoder_pos_embed = utils.get_2d_sincos_pos_embed(self.decoder.decoder_pos_embed.shape[-1], int(num_patches ** .5), cls_token=False)
        self.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.decoder_embed = nn.Linear(vit.embed_dim, decoder_embed_dim, bias=True)
        self.scale_embed = nn.Sequential(nn.Linear(2, vit.embed_dim), nn.GELU(), nn.Linear(vit.embed_dim, vit.embed_dim))
    def forward_encoder(self, images, metadata, idx_keep):
        tokens = self.encoder.images_to_tokens(images)
        scale_embedding = self.scale_embed(torch.log(metadata + 1e-6))
        tokens = tokens + scale_embedding.unsqueeze(1)
        tokens = self.encoder.prepend_prefix_tokens(tokens)
        tokens = self.encoder.add_pos_embed(tokens)
        if idx_keep is not None: tokens = torch.gather(tokens, dim=1, index=idx_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        encoded = self.encoder.vit.norm_pre(tokens)
        encoded = self.encoder.vit.blocks(encoded)
        encoded = self.encoder.vit.norm(encoded)
        return encoded
    def forward_decoder(self, x_encoded, ids_restore):
        x_encoded = self.decoder_embed(x_encoded)
        x_full = torch.cat([x_encoded, self.decoder.mask_token.repeat(x_encoded.shape[0], ids_restore.shape[1] - x_encoded.shape[1], 1)], dim=1)
        x_unshuffled = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))
        x_unshuffled = x_unshuffled + self.decoder.decoder_pos_embed
        decoded = self.decoder.decode(x_unshuffled)
        return self.decoder.predict(decoded)
    def forward(self, images, metadata):
        batch_size = images.shape[0]
        num_patches = self.encoder.vit.patch_embed.num_patches
        num_masked = int(self.mask_ratio * num_patches)
        ids_shuffle = torch.randperm(num_patches, device=images.device).unsqueeze(0).expand(batch_size, -1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep_patches = ids_shuffle[:, :num_patches - num_masked]
        ids_keep_full = torch.cat([torch.zeros(batch_size, 1, device=images.device, dtype=torch.long), ids_keep_patches + 1], dim=1)
        latent_full = self.forward_encoder(images, metadata, idx_keep=ids_keep_full)
        latent_patches = latent_full[:, 1:, :]
        predictions = self.forward_decoder(latent_patches, ids_restore)
        return predictions, ids_restore

# --- Part 3: Utility Functions ---
def patchify(imgs, patch_size):
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
    return x

def find_latest_checkpoint(output_dir, base_name):
    if not os.path.isdir(output_dir): return None, -1
    pattern = re.compile(rf"{re.escape(base_name)}_epoch_(\d+)\.pth")
    checkpoints = []
    for f in os.listdir(output_dir):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, os.path.join(output_dir, f)))
    if not checkpoints: return None, -1
    latest_epoch, latest_path = max(checkpoints, key=lambda item: item[0])
    return latest_path, latest_epoch

def plot_history(history_path, output_dir):
    epochs, train_losses, val_losses = [], [], []
    try:
        with open(history_path, 'r') as f:
            next(f) # Skip header
            for line in f:
                epoch, train_loss, val_loss = line.strip().split(',')
                epochs.append(int(epoch))
                train_losses.append(float(train_loss))
                val_losses.append(float(val_loss))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
        ax.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='x')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Over Epochs')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'loss_plot.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved loss plot to: {plot_path}")
    except Exception as e:
        print(f"Could not generate loss plot. Error: {e}", file=sys.stderr)

# --- Part 4: Main Training Function ---
def start_training(config, stop_event=None):
    try:
        output_dir = config['output_dir']
        base_name = config['base_name']
        os.makedirs(output_dir, exist_ok=True)
        print(f"All outputs will be saved to: {output_dir}\n")

        print("--- Catchment MAE Training Configuration ---")
        for key, value in config.items(): print(f"{key}: {value}")
        print("------------------------------------------\n")

        train_files, val_files, test_files = create_data_splits(root_dir=config['input_dir'], output_dir=output_dir, seed=config['seed'])
        transform = CustomCatchmentTransform(image_size=config['image_size'], norm_mean=config['norm_mean'], norm_std=config['norm_std'])
        train_dataset = TiffCatchmentDataset(root_dir=config['input_dir'], file_list=train_files, transform=transform, image_size=config['image_size'])
        val_dataset = TiffCatchmentDataset(root_dir=config['input_dir'], file_list=val_files, transform=transform, image_size=config['image_size'])
        test_dataset = TiffCatchmentDataset(root_dir=config['input_dir'], file_list=test_files, transform=transform, image_size=config['image_size'])
        if len(train_dataset) == 0: raise RuntimeError("Training set is empty.")

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], collate_fn=collate_fn_skip_corrupt)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], collate_fn=collate_fn_skip_corrupt)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], collate_fn=collate_fn_skip_corrupt)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CatchmentMAE(vit_model_name=config['vit_model_name'], in_chans=config['in_chans'], mask_ratio=config['mask_ratio']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
        patch_size = model.encoder.vit.patch_embed.patch_size[0]
        start_epoch = 0
        
        history_path = os.path.join(output_dir, 'history.csv')

        if config.get('continue_training'):
            latest_checkpoint_path, latest_epoch = find_latest_checkpoint(output_dir, base_name)
            if latest_checkpoint_path:
                print(f"Found latest checkpoint: {latest_checkpoint_path}")
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch}.\n")
            else:
                print("No checkpoint found. Starting training from scratch.\n")
                with open(history_path, 'w') as f: f.write("epoch,train_loss,val_loss\n")
        else:
            with open(history_path, 'w') as f: f.write("epoch,train_loss,val_loss\n")
        
        print(f"Starting training from epoch {start_epoch}...")
        print(f"Using device: {device}\n")

        for epoch in range(start_epoch, config['max_epochs']):
            if stop_event and stop_event.is_set():
                print(f"\n--- Training stopped by user at epoch {epoch} ---")
                final_model_path = os.path.join(output_dir, f"{base_name}.pth")
                torch.save(model.state_dict(), final_model_path)
                print(f"Final model state saved to: {final_model_path}")
                return

            model.train()
            total_train_loss, num_train_batches = 0, 0
            for batch in train_dataloader:
                if batch is None: continue
                images, metadata, masks = batch
                images, metadata, masks = images.to(device), metadata.to(device), masks.to(device)
                predictions, ids_restore = model(images, metadata)
                target_patches = patchify(images, patch_size)
                mask_patches = (patchify(masks, patch_size).mean(dim=-1) > 0.5).float().unsqueeze(-1)
                loss = F.mse_loss(predictions, target_patches, reduction='none')
                masked_loss = loss * mask_patches
                final_loss = masked_loss.sum() / (mask_patches.sum() + 1e-8)
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
                total_train_loss += final_loss.item()
                num_train_batches += 1
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0

            model.eval()
            total_val_loss, num_val_batches = 0, 0
            with torch.no_grad():
                for batch in val_dataloader:
                    if batch is None: continue
                    images, metadata, masks = batch
                    images, metadata, masks = images.to(device), metadata.to(device), masks.to(device)
                    predictions, ids_restore = model(images, metadata)
                    target_patches = patchify(images, patch_size)
                    mask_patches = (patchify(masks, patch_size).mean(dim=-1) > 0.5).float().unsqueeze(-1)
                    loss = F.mse_loss(predictions, target_patches, reduction='none')
                    masked_loss = loss * mask_patches
                    final_loss = masked_loss.sum() / (mask_patches.sum() + 1e-8)
                    total_val_loss += final_loss.item()
                    num_val_batches += 1
            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
            
            print(f"Epoch {epoch+1}/{config['max_epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            with open(history_path, 'a') as f: f.write(f"{epoch+1},{avg_train_loss},{avg_val_loss}\n")

            checkpoint_path = os.path.join(output_dir, f"{base_name}_epoch_{epoch+1:04d}.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")

        if not (stop_event and stop_event.is_set()):
            print("\n--- Training Finished Successfully ---")
            print("Running final evaluation on the test set...")
            model.eval()
            total_test_loss, num_test_batches = 0, 0
            with torch.no_grad():
                for batch in test_dataloader:
                    if batch is None: continue
                    images, metadata, masks = batch
                    images, metadata, masks = images.to(device), metadata.to(device), masks.to(device)
                    predictions, ids_restore = model(images, metadata)
                    target_patches = patchify(images, patch_size)
                    mask_patches = (patchify(masks, patch_size).mean(dim=-1) > 0.5).float().unsqueeze(-1)
                    loss = F.mse_loss(predictions, target_patches, reduction='none')
                    masked_loss = loss * mask_patches
                    final_loss = masked_loss.sum() / (mask_patches.sum() + 1e-8)
                    total_test_loss += final_loss.item()
                    num_test_batches += 1
            avg_test_loss = total_test_loss / num_test_batches if num_test_batches > 0 else 0
            print(f"\n--- FINAL TEST LOSS: {avg_test_loss:.4f} ---")

            final_model_path = os.path.join(output_dir, f"{base_name}.pth")
            torch.save(model.state_dict(), final_model_path)
            print(f"Final model state saved to: {final_model_path}")
            
            config_path = os.path.join(output_dir, f"{base_name}.json")
            with open(config_path, 'w') as f: json.dump(config, f, indent=4)
            print(f"Model configuration saved to: {config_path}")
            
            plot_history(history_path, output_dir)

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise e

# --- Part 5: Flask Web Application ---
app = Flask(__name__)

# Global variables for Flask app state
training_log_capture = io.StringIO()
training_status = {"running": False, "log": "", "error": False, "stopped": False}
stop_training_event = threading.Event()

def run_training_thread_wrapper(config):
    global training_status, training_log_capture
    # Reset log and status for the new run
    training_log_capture = io.StringIO()
    training_status.update({"running": True, "log": "Starting training...\n", "error": False, "stopped": False})
    try:
        with redirect_stdout(training_log_capture):
            start_training(config, stop_event=stop_training_event)
    except Exception:
        print(f"\n--- AN ERROR OCCURRED IN TRAINING THREAD ---")
        print(traceback.format_exc())
        training_status["error"] = True
    finally:
        training_status["running"] = False
        training_status["log"] = training_log_capture.getvalue()

@app.route("/", methods=["GET"])
def index():
    # The HTML template remains the same
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Train & Use Catchment Foundation Model</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background-color: #f8f9fa; color: #212529; }
            h1, h2, h3 { color: #343a40; }
            .container { max-width: 900px; margin: auto; background: white; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }
            .form-group { margin-bottom: 1em; }
            label { display: block; margin-bottom: 0.5em; font-weight: 600; }
            input, select { width: 100%; padding: 0.75em; border: 1px solid #ced4da; border-radius: 4px; box-sizing: border-box; }
            small { color: #6c757d; }
            button { background-color: #007bff; color: white; padding: 0.75em 1.5em; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; transition: background-color 0.2s; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #6c757d; cursor: not-allowed; }
            #stop-button { background-color: #dc3545; }
            #stop-button:hover { background-color: #c82333; }
            .log-container { margin-top: 1em; background-color: #282c34; color: #abb2bf; padding: 1em; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; font-family: "SF Mono", "Fira Code", monospace; }
            .status { font-style: italic; color: #6c757d; }
            .section { margin-top: 2em; border-top: 1px solid #dee2e6; padding-top: 2em; }
            #inference-section, #plotting-section { display: none; }
            #embedding-output { background-color: #e9ecef; padding: 1em; border-radius: 4px; font-size: 0.9em; }
            #plot-container img { max-width: 100%; border-radius: 4px; margin-top: 1em; }
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
                        <label for="output_dir">Output Directory Name:</label>
                        <input type="text" id="output_dir" name="output_dir" value="catchment_model_outputs" required>
                        <small>A folder with this name will be created to store all outputs (models, logs, plots).</small>
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
                        <input type="text" id="norm_mean" name="norm_mean" value="756.5" required>
                        <small>For 1 channel, use a single number. For 3, use '0.1,0.2,0.3'.</small>
                    </div>
                    <div class="form-group">
                        <label for="norm_std">Normalization Std:</label>
                        <input type="text" id="norm_std" name="norm_std" value="688.0" required>
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
                        <input type="number" id="num_workers" name="num_workers" value="16" min="0" required>
                    </div>
                </div>
                <div class="form-group">
                    <label><input type="checkbox" id="continue_training" name="continue_training"> Continue from latest checkpoint</label>
                    <small>Looks for `..._epoch_*.pth` files in the specified Output Directory.</small>
                </div>
                <button type="submit" id="train-button">Start Training</button>
                <button type="button" id="stop-button" disabled>Stop Training</button>
            </form>
            <div id="status-container">
                <p class="status">Status: <span id="status-text">Idle</span></p>
                <pre id="log-output" class="log-container">Training logs will appear here...</pre>
            </div>
            <div id="inference-section" class="section">
                <h2>2. Get Embedding from Trained Model</h2>
                <form id="inference-form">
                    <div class="grid">
                        <div class="form-group">
                            <label for="model_path">Path to trained model (.pth):</label>
                            <input type="text" id="model_path" name="model_path" placeholder="catchment_model_outputs/catchment_model_outputs.pth" required>
                        </div>
                        <div class="form-group">
                            <label for="image_path">Path to a single TIFF image:</label>
                            <input type="text" id="image_path" name="image_path" placeholder="/path/to/single/catchment.tif" required>
                        </div>
                    </div>
                    <button type="submit" id="embed-button">Get Embedding</button>
                </form>
                <p class="status">Result:</p>
                <pre id="embedding-output" class="log-container">Embedding vector will appear here...</pre>
            </div>
            <div id="plotting-section" class="section">
                <h3>3. Plot Training History</h3>
                <form id="plot-form">
                    <div class="form-group">
                        <label for="plot_output_dir">Output Directory:</label>
                        <input type="text" id="plot_output_dir" name="plot_output_dir" placeholder="catchment_model_outputs" required>
                        <small>The directory containing the `history.csv` file.</small>
                    </div>
                    <button type="submit" id="plot-button">Generate Plot</button>
                </form>
                <div id="plot-container"></div>
            </div>
        </div>
        <script>
            const trainForm = document.getElementById('training-form');
            const trainButton = document.getElementById('train-button');
            const stopButton = document.getElementById('stop-button');
            const statusText = document.getElementById('status-text');
            const logOutput = document.getElementById('log-output');
            const inferenceSection = document.getElementById('inference-section');
            const plottingSection = document.getElementById('plotting-section');
            let trainingIntervalId;

            trainForm.addEventListener('submit', function(event) {
                event.preventDefault();
                trainButton.disabled = true;
                stopButton.disabled = false;
                statusText.textContent = 'Starting...';
                logOutput.textContent = 'Sending training request...';
                inferenceSection.style.display = 'none';
                plottingSection.style.display = 'none';
                document.getElementById('plot-container').innerHTML = '';
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
                        stopButton.disabled = true;
                        statusText.textContent = 'Error';
                    }
                });
            });

            stopButton.addEventListener('click', function() {
                stopButton.disabled = true;
                statusText.textContent = 'Stopping...';
                fetch('/stop', { method: 'POST' });
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
                            stopButton.disabled = true;
                            if (data.error) {
                                statusText.textContent = 'Error';
                                statusText.style.color = 'red';
                            } else if (data.stopped) {
                                statusText.textContent = 'Stopped by user';
                                statusText.style.color = '#ffc107';
                            } else {
                                statusText.textContent = 'Finished Successfully';
                                statusText.style.color = 'green';
                            }
                            inferenceSection.style.display = 'block';
                            plottingSection.style.display = 'block';
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

            const plotForm = document.getElementById('plot-form');
            const plotButton = document.getElementById('plot-button');
            const plotContainer = document.getElementById('plot-container');
            plotForm.addEventListener('submit', function(event) {
                event.preventDefault();
                plotButton.disabled = true;
                plotContainer.innerHTML = '<p>Generating plot...</p>';
                const formData = new FormData(plotForm);
                fetch('/plot', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        plotContainer.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Training History Plot">`;
                    } else {
                        plotContainer.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
                    }
                    plotButton.disabled = false;
                });
            });
        </script>
    </body>
    </html>
    """)

def parse_list_from_string(s):
    return [float(item.strip()) for item in s.split(',')]

@app.route("/train", methods=["POST"])
def train():
    global training_status, stop_training_event
    if training_status["running"]: return jsonify({"status": "error", "message": "A training job is already running."}), 400
    stop_training_event.clear()
    try:
        output_dir = request.form["output_dir"]
        base_name = os.path.basename(output_dir)
        config = {
            "input_dir": request.form["input_dir"], "output_dir": output_dir, "base_name": base_name,
            "vit_model_name": request.form["vit_model_name"], "image_size": int(request.form["image_size"]),
            "in_chans": int(request.form["in_chans"]), "norm_mean": parse_list_from_string(request.form["norm_mean"]),
            "norm_std": parse_list_from_string(request.form["norm_std"]), "max_epochs": int(request.form["max_epochs"]),
            "batch_size": int(request.form["batch_size"]), "mask_ratio": float(request.form["mask_ratio"]),
            "num_workers": int(request.form["num_workers"]), "seed": 42,
            "continue_training": "continue_training" in request.form,
        }
        if len(config['norm_mean']) != config['in_chans'] or len(config['norm_std']) != config['in_chans']:
            raise ValueError("Normalization values must match channel count.")
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid form data: {e}"}), 400
    if not os.path.isdir(config['input_dir']): return jsonify({"status": "error", "message": f"Dataset directory not found: {config['input_dir']}"}), 400
    
    training_thread = threading.Thread(target=run_training_thread_wrapper, args=(config,))
    training_thread.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop_training():
    global training_status, stop_training_event
    if not training_status["running"]: return jsonify({"status": "error", "message": "No training job is currently running."}), 400
    stop_training_event.set()
    return jsonify({"status": "stopping"})

@app.route("/status", methods=["GET"])
def status():
    global training_status, training_log_capture
    training_status["log"] = training_log_capture.getvalue()
    return jsonify(training_status)

@app.route("/embed", methods=["POST"])
def embed():
    model_path = request.form.get("model_path")
    image_path = request.form.get("image_path")
    nodata_value = -9999.0
    if not model_path or not os.path.isfile(model_path): return jsonify({"status": "error", "message": f"Model file not found: {model_path}"}), 400
    if not image_path or not os.path.isfile(image_path): return jsonify({"status": "error", "message": f"Image file not found: {image_path}"}), 400
    
    output_dir = os.path.dirname(model_path)
    base_name = os.path.basename(model_path).replace('.pth', '')
    config_path = os.path.join(output_dir, f"{base_name}.json")
    
    if not os.path.isfile(config_path): return jsonify({"status": "error", "message": f"Model config file not found at inferred path: {config_path}"}), 400
    try:
        with open(config_path, 'r') as f: config = json.load(f)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CatchmentMAE(vit_model_name=config['vit_model_name'], in_chans=config['in_chans']).to(device)
        saved_data = torch.load(model_path, map_location=device)
        if isinstance(saved_data, dict) and 'model_state_dict' in saved_data: model.load_state_dict(saved_data['model_state_dict'])
        else: model.load_state_dict(saved_data)
        model.eval()
        inference_transform = T.Compose([
            T.Resize(size=(config['image_size'], config['image_size']), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(mean=config['norm_mean'], std=config['norm_std']),
        ])
        image = tifffile.imread(image_path).astype(np.float32)
        original_shape = image.shape[-2:]
        valid_mask_np = (image != nodata_value)
        if valid_mask_np.any():
            mean_val = image[valid_mask_np].mean()
            image[~valid_mask_np] = mean_val
        else: image.fill(0)
        if image.ndim == 2: image = np.expand_dims(image, axis=0)
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        transformed_image = inference_transform(image_tensor)
        with torch.no_grad():
            metadata = torch.tensor(original_shape, dtype=torch.float32).unsqueeze(0).to(device)
            encoded_tokens = model.forward_encoder(transformed_image, metadata, idx_keep=None)
            embedding = encoded_tokens[:, 0]
        return jsonify({"status": "success", "embedding": embedding.squeeze().tolist(), "shape": list(embedding.squeeze().shape)})
    except Exception as e:
        return jsonify({"status": "error", "message": traceback.format_exc()}), 500

@app.route("/plot", methods=["POST"])
def plot():
    output_dir = request.form.get("plot_output_dir")
    history_path = os.path.join(output_dir, 'history.csv')
    if not os.path.isfile(history_path):
        return jsonify({"status": "error", "message": f"History file not found: {history_path}"}), 400
    try:
        buf = io.BytesIO()
        plot_history(history_path, output_dir)
        plot_path = os.path.join(output_dir, 'loss_plot.png')
        with open(plot_path, 'rb') as f: buf.write(f.read())
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"status": "success", "image": image_base64})
    except Exception as e:
        return jsonify({"status": "error", "message": traceback.format_exc()}), 500

def run_web_server():
    print("Starting Catchment MAE Training UI...")
    print("Please install imagecodecs if you have LZW compressed TIFFs: pip install imagecodecs")
    print("Navigate to http://127.0.0.1:5001 in your browser.")
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

def run_cli():
    parser = argparse.ArgumentParser(description="Train a Catchment Foundation Model (MAE) from the command line.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing TIFF images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory where all outputs will be saved.')
    parser.add_argument('--vit_model_name', type=str, default='vit_small_patch16_224', help='Name of the Vision Transformer model from timm.')
    parser.add_argument('--image_size', type=int, default=224, help='Size to which images will be resized.')
    parser.add_argument('--in_chans', type=int, default=1, help='Number of input channels for the model.')
    parser.add_argument('--norm_mean', type=lambda s: [float(item) for item in s.split(',')], default=[756.5], help='Mean for normalization, comma-separated for multiple channels.')
    parser.add_argument('--norm_std', type=lambda s: [float(item) for item in s.split(',')], default=[688.0], help='Standard deviation for normalization, comma-separated.')
    parser.add_argument('--max_epochs', type=int, default=50, help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per batch.')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Proportion of patches to mask.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--continue_training', action='store_true', help='Flag to continue training from the latest checkpoint in the output directory.')
    
    args = parser.parse_args(sys.argv[2:])
    config = vars(args)
    config['base_name'] = os.path.basename(config['output_dir'])
    
    start_training(config)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        run_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == 'web':
        run_web_server()
    else:
        # Default to web server if no command or an unknown command is given
        if len(sys.argv) > 1:
            print(f"Unknown command: '{sys.argv[1]}'. Starting web server by default.")
        run_web_server()