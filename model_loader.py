import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from multitask_learning.model import MultiTaskModel

MULTITASK_MODEL_PATH = "./multitask_learning/best_model.pth"


def load_multitask_model():
    model = MultiTaskModel(num_tasks=5)
    state_dict = torch.load(MULTITASK_MODEL_PATH)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    return model


def load_visual_captioning_model(weights_path: str):
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    # model.load_state_dict(torch.load(weights_path))
    return model, processor
