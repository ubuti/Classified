import torchvision
import torch
import warnings
import sys
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

warnings.filterwarnings("ignore")

sys.path.insert(1, '/Users/inuit/Documents/Gitrepos/Classified/datasets')
from transform_imagenet100 import load_image100_data
dataloader, labels_dict = load_image100_data(train=False)
imgs, labels = next(iter(dataloader))

sys.path.insert(1, '/Users/inuit/Documents/Gitrepos/Classified/models')
from use_model import classify_sample
from load_model import load_model

MODELS = ["resnet50", "resnet18", "resnet34", "alexnet", "googlenet", "inception_v3"]
WEIGHTS = ["IMAGENET1K_V1", "IMAGENET1K_V2"]

guesses = 1

# Global variables
current_model = None
# keep current_img as a torch Tensor (C,H,W) and current_label as a Tensor/int
current_img, current_label = imgs[0].clone(), labels[0].clone()

def load_model_wrapper(model_name, weights):
    global current_model
    current_model = load_model(model_name, weights)
    return f"Loaded {model_name} with {weights}"

def next_image():
    """
    Load next sample from dataloader, store tensor in current_img and return a matplotlib Figure for display.
    Do not convert current_img to numpy permanently because classify_sample expects a torch tensor.
    """
    global current_img, current_label
    imgs, labels = next(iter(dataloader))
    # keep tensor for classification (on CPU). clone to avoid in-place changes.
    current_img, current_label = imgs[0].clone().detach(), labels[0].clone().detach()

    # Convert tensor to numpy for plotting only (channels first -> HWC, un-normalize)
    img_np = current_img.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_np)
    # use int() to convert tensor label to plain int
    try:
        title = labels_dict[int(current_label)]
    except Exception:
        title = str(int(current_label.item()) if hasattr(current_label, "item") else int(current_label))
    ax.set_title(f"True Label: {title}")
    ax.axis('off')
    plt.close(fig)
    return fig

def classify_with_loaded_model(num_guesses: int = 1):
    """
    Call user's classify_sample using the stored torch tensor current_img.
    num_guesses: number of top classes to return (passed to classify_sample as topk).
    Returns (fig, probs_text) for Gradio.
    """
    if current_model is None:
        return None, "Please load a model first."

    # ensure integer and >=1
    topk = max(1, int(num_guesses))

    # classify_sample expects a torch tensor; pass current_img (C,H,W) which your function handles
    res, display_img, label = classify_sample(
        current_model, current_img, labels_dict=labels_dict,
        true_label=current_label, topk=topk
    )

    # display_img may be a numpy array or PIL; ensure we create a matplotlib figure for Gradio plot
    if isinstance(display_img, torch.Tensor):
        disp = display_img.cpu().numpy().transpose(1, 2, 0)
    else:
        disp = display_img

    # prepare figure
    if isinstance(disp, np.ndarray):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(np.clip(disp, 0.0, 1.0))
        ax.set_title(f"Label: {label}")
        ax.axis('off')
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.text(0.5, 0.5, str(display_img), ha='center', va='center')
        ax.axis('off')
        plt.close(fig)

    # res is a zip object of (idx, prob) pairs; convert to list to iterate safely
    res_list = list(res)
    probs_text = "\n".join([f"{labels_dict[int(k) if not isinstance(k, (list,tuple)) else int(k[0]) if isinstance(k, (list,tuple)) else int(k)]}: {float(v):.4f}" 
                             if not isinstance(k, (list,tuple)) else f"{labels_dict[int(k[0])]}: {float(k[1]):.4f}"
                             for k,v in res_list]) if res_list else ""

    # simpler, robust formatting if res contains (idx,prob) tuples
    try:
        probs_text = "\n".join([f"{labels_dict[int(idx)]}: {prob:.4f}" for idx, prob in res_list])
    except Exception:
        # fallback: stringify pairs
        probs_text = "\n".join([f"{str(p)}" for p in res_list])

    return fig, probs_text

# Gradio app
with gr.Blocks() as demo:
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=MODELS, label="Select Model")
        weights_input = gr.Dropdown(choices=WEIGHTS, label="Weights")
        load_btn = gr.Button("Load")
        load_status = gr.Textbox(label="Model Status", interactive=False)

    with gr.Row():
        next_img_btn = gr.Button("Next Image")
        next_img_status = gr.Textbox(label="Image Status", interactive=False)

    with gr.Row():
        guesses_input = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Guesses (top-k)")
        classify_btn = gr.Button("Classify")

    with gr.Row():
        with gr.Column():
            plot_output = gr.Plot(label="Current Image")
        with gr.Column():
            probs_output = gr.Textbox(label="Class Probabilities", lines=10)

    # Load model on "Load" button click
    load_btn.click(
        fn=load_model_wrapper,
        inputs=[model_dropdown, weights_input],
        outputs=load_status
    )

    # Load next image on "Next Image" button click (display figure)
    next_img_btn.click(
        fn=next_image,
        inputs=None,
        outputs=plot_output
    )

    # Classify using the loaded model and current image; pass guesses_input to the function
    classify_btn.click(
        fn=classify_with_loaded_model,
        inputs=[guesses_input],
        outputs=[plot_output, probs_output]
    )

demo.launch()