import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForZeroShotObjectDetection,
)

from multitask_learning.model import MultiTaskModel

MULTITASK_MODEL_PATH = "./multitask_learning/best_model.pth"


def load_multitask_model():
    model = MultiTaskModel(num_tasks=5)
    state_dict = torch.load(MULTITASK_MODEL_PATH)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    return model


def load_caption_model():
    model_name = "Salesforce/blip-image-captioning-base"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    state_dict = torch.load("./visual_captioning/best_model.pth")
    model.load_state_dict(state_dict["model_state_dict"])
    return model, processor


def load_grounding_model():
    model_name = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    return model, processor
