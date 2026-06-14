"""
Standalone serving app for the Visual Product Search + Near-Duplicate Detector.

This is the "online" half of the system: it loads precomputed CLIP embeddings
and product metadata (produced by Visual_Search_Duplicate_Detection.ipynb,
section 9 "Save artifacts for the deployable demo") and serves search requests
via Gradio. The "offline" half (downloading data, computing embeddings,
building indices, ablation study, Grad-CAM) lives in the notebook.

Expected files in the same directory:
  - embeddings.npz   (arrays: image, text, multimodal)
  - products.pkl     (pandas DataFrame with columns: item_id, title,
                       product_type, image_path)

Run locally:
    pip install -r requirements.txt
    python app.py

Deploy on Hugging Face Spaces:
    Push this file + embeddings.npz + products.pkl + the image files +
    requirements.txt to a Spaces repo with the Gradio SDK.
"""

import numpy as np
import pandas as pd
import torch
import open_clip
import faiss
import gradio as gr
from PIL import Image

THRESHOLD = 0.97  # near-duplicate cosine similarity cut-off, tune against labelled pairs

print("Loading embeddings and product metadata...")
data = np.load("embeddings.npz")
image_embeds = data["image"].astype("float32")
df = pd.read_pickle("products.pkl")

print("Building FAISS index...")
image_index = faiss.IndexFlatIP(image_embeds.shape[1])
image_index.add(image_embeds)

print("Loading CLIP model (ViT-B-32, laion2b)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()
print(f"Ready on {device}.")


def search_and_explain(uploaded_image, top_k=5):
    if uploaded_image is None:
        return [], "Upload an image first."

    img = uploaded_image.convert("RGB")
    with torch.no_grad():
        img_t = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.cpu().numpy().astype("float32")

    scores, indices = image_index.search(feat, top_k)

    results, flags = [], []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        caption = f"{row['title'][:45]} (sim={score:.2f})"
        results.append((Image.open(row["image_path"]), caption))
        if score > THRESHOLD:
            flags.append(
                f"Potential duplicate/counterfeit match: "
                f"'{row['title'][:50]}' (sim={score:.3f})"
            )

    flag_text = "\n".join(flags) if flags else "No near-duplicates detected above threshold."
    return results, flag_text


with gr.Blocks(title="Visual Product Search & Duplicate Detector") as demo:
    gr.Markdown("## Multimodal Visual Search + Near-Duplicate / Counterfeit Listing Detector")
    gr.Markdown(
        "Upload a product photo to find visually similar catalogue listings "
        "and flag potential duplicate/counterfeit matches. "
        f"Catalogue size: {len(df)} products."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Query image")
        with gr.Column():
            gallery = gr.Gallery(label="Top matches", columns=5)
            flags_box = gr.Textbox(label="Duplicate / Counterfeit Flags", lines=4)
    btn = gr.Button("Search")
    btn.click(search_and_explain, inputs=inp, outputs=[gallery, flags_box])

if __name__ == "__main__":
    demo.launch()
