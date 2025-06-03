from fastapi import WebSocket
import asyncio
import json
import logging
from typing import Set

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.consumers: Set[WebSocket] = set()
        self.producer: WebSocket = None
        self.latest_frame: bytes = None
        self.prediction_listeners: Set[WebSocket] = set()
        self.is_processing_prediction = False  # Add this flag

    async def connect_producer(self, websocket: WebSocket):
        """Connect the video producer (streamer)"""
        await websocket.accept()
        if self.producer:
            await self.producer.close()
        self.producer = websocket
        logger.info("Producer connected")

    async def connect_consumer(self, websocket: WebSocket):
        """Connect a video consumer (viewer)"""
        await websocket.accept()
        self.consumers.add(websocket)
        logger.info(f"Consumer connected. Total consumers: {len(self.consumers)}")

    async def connect_prediction_listener(self, websocket: WebSocket):
        """Connect a prediction result listener"""
        await websocket.accept()
        self.prediction_listeners.add(websocket)
        logger.info(
            f"Prediction listener connected. Total listeners: {len(self.prediction_listeners)}"
        )

    def disconnect_producer(self):
        """Disconnect the producer"""
        self.producer = None
        logger.info("Producer disconnected")

    def disconnect_consumer(self, websocket: WebSocket):
        """Disconnect a consumer"""
        self.consumers.discard(websocket)
        logger.info(f"Consumer disconnected. Total consumers: {len(self.consumers)}")

    def disconnect_prediction_listener(self, websocket: WebSocket):
        """Disconnect a prediction listener"""
        self.prediction_listeners.discard(websocket)
        logger.info(
            f"Prediction listener disconnected. Total listeners: {len(self.prediction_listeners)}"
        )

    async def broadcast_to_consumers(self, data: bytes, process_prediction_func=None):
        """Broadcast video data to all connected consumers"""
        self.latest_frame = data

        # Only process prediction if not already processing and listeners exist
        if (
            self.prediction_listeners
            and process_prediction_func
            and not self.is_processing_prediction
        ):
            self.is_processing_prediction = True
            asyncio.create_task(
                self._process_prediction_with_flag(process_prediction_func, data)
            )

        # if self.prediction_listeners and process_prediction_func:
        #     asyncio.create_task(process_prediction_func(data))

        if not self.consumers:
            return

        tasks = []
        consumers_to_remove = set()

        for consumer in self.consumers.copy():
            try:
                tasks.append(consumer.send_bytes(data))
            except Exception as e:
                logger.error(f"Error preparing to send to consumer: {e}")
                consumers_to_remove.add(consumer)

        for consumer in consumers_to_remove:
            self.consumers.discard(consumer)

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error broadcasting to consumers: {e}")

    async def _process_prediction_with_flag(self, process_func, frame_data):
        """Process prediction and reset flag when done"""
        try:
            await process_func(frame_data)
        finally:
            self.is_processing_prediction = False

    async def broadcast_prediction(self, prediction_data: dict):
        """Broadcast prediction results to all listeners"""
        if not self.prediction_listeners:
            return

        message = json.dumps(prediction_data)
        tasks = []
        listeners_to_remove = set()

        for listener in self.prediction_listeners.copy():
            try:
                tasks.append(listener.send_text(message))
            except Exception as e:
                logger.error(f"Error preparing to send prediction to listener: {e}")
                listeners_to_remove.add(listener)

        for listener in listeners_to_remove:
            self.prediction_listeners.discard(listener)

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error broadcasting predictions: {e}")

    async def send_message_to_consumers(self, message: dict):
        """Send JSON message to all consumers"""
        if not self.consumers:
            return

        message_str = json.dumps(message)
        tasks = []
        consumers_to_remove = set()

        for consumer in self.consumers.copy():
            try:
                tasks.append(consumer.send_text(message_str))
            except Exception as e:
                logger.error(f"Error preparing to send message to consumer: {e}")
                consumers_to_remove.add(consumer)

        for consumer in consumers_to_remove:
            self.consumers.discard(consumer)

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error sending message to consumers: {e}")
