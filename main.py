from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    File,
    UploadFile,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi.responses import Response
from torchvision import transforms
from PIL import Image
import numpy as np
import asyncio
import logging
import torch
import time
import json
import io

from model_loader import load_multitask_model, load_caption_model
from connection_manager import ConnectionManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
multitask_model = None
caption_model = None
caption_processor = None
executor = ThreadPoolExecutor(max_workers=1)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.8220, 0.8310, 0.8322], std=[0.2968, 0.2954, 0.2816]
        ),
    ]
)


async def initialize_model():
    """Initialize the model on startup"""
    global multitask_model, caption_model, caption_processor
    loop = asyncio.get_event_loop()
    multitask_model = await loop.run_in_executor(executor, load_multitask_model)
    logger.info("Multitask model loaded successfully")
    caption_model, caption_processor = await loop.run_in_executor(
        executor, load_caption_model
    )
    logger.info("Caption model and processor loaded successfully")


def process_frame_prediction(frame_bytes: bytes):
    """Process frame and return prediction (runs in thread)"""
    try:
        # Convert JPEG bytes to PIL Image
        image = Image.open(io.BytesIO(frame_bytes))

        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms (resize, normalize, etc.)
        tensor = transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Run prediction
        with torch.no_grad():
            predictions = multitask_model(tensor)

        # Process predictions (adjust based on your model output)
        return {
            "timestamp": time.time(),
            "predictions": predictions.tolist(),
            "frame_shape": list(tensor.shape),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


async def process_and_broadcast_prediction(frame_data: bytes):
    """Process frame and broadcast prediction result"""
    try:
        if not multitask_model:
            return

        loop = asyncio.get_event_loop()
        prediction_result = await loop.run_in_executor(
            executor, process_frame_prediction, frame_data
        )

        if prediction_result:
            # [[0.00027007897733710706, 5.195397079660324e-06, 0.9999868869781494, 0.12733076512813568, 1.6550764314615662e-07]]
            # threshold to binary
            predictions = {
                "front_left_door": "open"
                if prediction_result["predictions"][0][0] > 0.5
                else "closed",
                "front_right_door": "open"
                if prediction_result["predictions"][0][1] > 0.5
                else "closed",
                "rear_left_door": "open"
                if prediction_result["predictions"][0][2] > 0.5
                else "closed",
                "rear_right_door": "open"
                if prediction_result["predictions"][0][3] > 0.5
                else "closed",
                "hood": "open"
                if prediction_result["predictions"][0][4] > 0.5
                else "closed",
            }
            prediction_result["predictions"] = predictions
            print(prediction_result)

            await manager.broadcast_prediction(prediction_result)
    except Exception as e:
        logger.error(f"Error in prediction processing: {e}")


def generate_image_caption(image_data: bytes):
    """Generate caption for image (runs in thread)"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image with processor
        encoding = caption_processor(
            image,
            # padding="max_length",
            # truncation=True,
            # max_length=128,
            return_tensors="pt",
        )
        # encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Generate caption using your model
        with torch.no_grad():
            outputs = caption_model.generate(
                **encoding, max_length=128, num_beams=4, do_sample=True
            )
            caption = caption_processor.decode(outputs[0], skip_special_tokens=True)

        return caption.replace(" ' ", "'")
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_model()
    yield


app = FastAPI(title="Video Streaming API Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()


@app.websocket("/ws/producer")
async def websocket_producer(websocket: WebSocket):
    """WebSocket endpoint for video producer (streamer)"""
    await manager.connect_producer(websocket)

    await manager.send_message_to_consumers(
        {"type": "producer_status", "status": "connected"}
    )

    try:
        while True:
            data = await websocket.receive_bytes()
            await manager.broadcast_to_consumers(
                data, process_and_broadcast_prediction if multitask_model else None
            )

    except WebSocketDisconnect:
        manager.disconnect_producer()
        await manager.send_message_to_consumers(
            {"type": "producer_status", "status": "disconnected"}
        )
    except Exception as e:
        logger.error(f"Producer error: {e}")
        manager.disconnect_producer()
        await manager.send_message_to_consumers(
            {"type": "producer_status", "status": "error", "message": str(e)}
        )


@app.websocket("/ws/consumer")
async def websocket_consumer(websocket: WebSocket):
    """WebSocket endpoint for video consumers (viewers)"""
    await manager.connect_consumer(websocket)

    producer_status = "connected" if manager.producer else "disconnected"
    await websocket.send_text(
        json.dumps({"type": "producer_status", "status": producer_status})
    )

    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect_consumer(websocket)
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        manager.disconnect_consumer(websocket)


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """WebSocket endpoint for prediction result listeners"""
    await manager.connect_prediction_listener(websocket)

    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect_prediction_listener(websocket)
    except Exception as e:
        logger.error(f"Prediction listener error: {e}")
        manager.disconnect_prediction_listener(websocket)


@app.get("/api/latest-frame")
async def get_latest_frame():
    """Get the latest frame from video stream"""
    if manager.latest_frame is None:
        return {"error": "No frame available"}

    return Response(
        content=manager.latest_frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/latest-frame-tensor")
async def get_latest_frame_tensor():
    """Get the latest frame as torch-compatible tensor data"""
    if manager.latest_frame is None:
        return {"error": "No frame available"}

    try:
        image = Image.open(io.BytesIO(manager.latest_frame))

        if image.mode != "RGB":
            image = image.convert("RGB")

        np_array = np.array(image)
        tensor = torch.from_numpy(np_array).permute(2, 0, 1).float() / 255.0

        return {
            # "tensor": tensor.tolist(),
            "shape": list(tensor.shape),
            "dtype": "float32",
            "format": "CHW",
            "normalized": True,
        }
    except Exception as e:
        return {"error": f"Failed to process frame: {str(e)}"}


@app.post("/api/caption-image")
async def caption_image(file: UploadFile = File(...)):
    """Generate caption for uploaded image"""
    if not caption_model or not caption_processor:
        raise HTTPException(status_code=503, detail="Caption model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image data
        image_data = await file.read()

        # Generate caption in thread to avoid blocking
        loop = asyncio.get_event_loop()
        caption = await loop.run_in_executor(
            executor, generate_image_caption, image_data
        )

        if caption is None:
            raise HTTPException(status_code=500, detail="Failed to generate caption")

        return {
            "caption": caption,
            "filename": file.filename,
            "content_type": file.content_type,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Error processing image caption request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/broadcast-prediction")
async def broadcast_prediction(prediction: dict):
    """Broadcast prediction result to all connected listeners"""
    await manager.broadcast_prediction(prediction)
    return {"status": "broadcasted", "listeners": len(manager.prediction_listeners)}


@app.get("/api/prediction-status")
async def get_prediction_status():
    """Get prediction broadcasting status"""
    return {
        "prediction_listeners": len(manager.prediction_listeners),
        "websocket_endpoint": "/ws/predictions",
        "broadcast_endpoint": "/api/broadcast-prediction",
        "model_loaded": multitask_model is not None,
    }


# Update the status endpoint to include caption model status
@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        "status": "online",
        "producer_connected": manager.producer is not None,
        "consumer_count": len(manager.consumers),
        "prediction_listeners": len(manager.prediction_listeners),
        "model_loaded": multitask_model is not None,
        "caption_model_loaded": caption_model is not None
        and caption_processor is not None,
        "websocket_endpoints": {
            "producer": "/ws/producer",
            "consumer": "/ws/consumer",
            "predictions": "/ws/predictions",
        },
        "caption_endpoints": {
            "upload": "/api/caption-image",
            "base64": "/api/caption-base64",
        },
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
