"""
NeoNetra V2.0 — WebSocket Bridge

Connects the Python Edge Hub to the ESP32 safety arbiter.
Sends SegFormer navigation commands.
Listens for emergency ToF interrupts (like DROPOFF).
"""

import asyncio
import websockets
import threading
import time

class ESP32Bridge:
    def __init__(self, ip: str, port: int = 82, tts_engine=None):
        self.uri = f"ws://{ip}:{port}"
        self.tts = tts_engine
        self.current_cmd = "CLEAR_0"
        self._running = False
        self._thread = None
        
        # Button Request Flags
        self.scene_requested = False
        self.ocr_requested = False
        self.mute_toggled = False
        
    def start(self):
        """Start the async WebSocket loop in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        
    def set_command(self, cmd: str):
        """Update the latest navigation command from SegFormer."""
        self.current_cmd = cmd
        
    def _run_async_loop(self):
        """Create a new event loop for this thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._websocket_worker())
        
    async def _websocket_worker(self):
        """Maintain connection and exchange data."""
        while self._running:
            try:
                print(f"[Bridge] Connecting to {self.uri}...")
                async with websockets.connect(self.uri, ping_interval=None) as ws:
                    print("[Bridge] Connected successfully!")
                    
                    # Start receiver task
                    receiver_task = asyncio.create_task(self._listen(ws))
                    
                    # Start sender loop
                    while self._running:
                        await ws.send(self.current_cmd)
                        await asyncio.sleep(0.25) # Send at 20 Hz
                        
            except (websockets.exceptions.ConnectionClosedError, OSError, Exception) as e:
                print(f"[Bridge] Disconnected: {e}. Retrying in 2s...")
                await asyncio.sleep(2)
                
    async def _listen(self, ws):
        """Listen for interrupts coming from ESP32."""
        try:
            async for message in ws:
                if message == "DROPOFF":
                    print("⚠️ [ESP32] DROPOFF EMERGENCY TRIGGERED!")
                    if self.tts:
                        self.tts.speak_priority("Seedhiyan hain! Rukiye!")
                elif message == "BTN_SCENE":
                    print("[Bridge] Scene Button Pressed!")
                    self.scene_requested = True
                elif message == "BTN_OCR":
                    print("[Bridge] OCR Button Pressed!")
                    self.ocr_requested = True
                elif message == "BTN_MUTE":
                    print("[Bridge] Mute Button Pressed!")
                    self.mute_toggled = True
                    
        except Exception:
            pass
            
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
