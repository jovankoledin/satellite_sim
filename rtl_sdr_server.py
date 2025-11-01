# This script runs on your Raspberry Pi 5.
# It requires 'rtlsdr', 'numpy', and 'websockets'
# Install with: pip install rtlsdr numpy websockets

import asyncio
import json
import numpy as np
import websockets
from rtlsdr import RtlSdr

# --- Global SDR and Task ---
sdr = None
streaming_task = None
connected_clients = set()

# --- SDR Configuration ---
# These can be changed by client messages
sdr_config = {
    'center_freq': 137.5e6,  # 137.5 MHz
    'sample_rate': 1.024e6,  # 1.024 Msps
    'gain': 'auto',
    'fft_bins': 1024
}

def init_sdr():
    """Initializes and configures the RTL-SDR device."""
    global sdr
    try:
        if sdr:
            sdr.close()
        
        sdr = RtlSdr()
        sdr.set_center_freq(sdr_config['center_freq'])
        sdr.set_sample_rate(sdr_config['sample_rate'])
        sdr.set_gain(sdr_config['gain'])
        print(f"SDR initialized:")
        print(f"  Center Freq: {sdr.get_center_freq() / 1e6} MHz")
        print(f"  Sample Rate: {sdr.get_sample_rate() / 1e6} Msps")
        print(f"  Gain: {sdr.get_gain()} dB")
        return True
    except Exception as e:
        print(f"Error initializing SDR: {e}")
        print("Is the RTL-SDR plugged in? Are permissions (udev rules) set correctly?")
        sdr = None
        return False

async def sdr_stream(websocket):
    """
    SDR streaming task. Reads samples, computes FFT, and sends to client.
    """
    global sdr
    if not sdr:
        print("SDR not initialized. Stream cannot start.")
        return

    print("Starting SDR stream...")
    fft_size = sdr_config['fft_bins']
    
    # Simple averaging (boxcar) for smoother output
    avg_count = 5
    fft_buffer = np.zeros((avg_count, fft_size))
    buffer_index = 0

    try:
        # Use sdr.stream() for continuous, non-blocking streaming
        async for samples in sdr.stream(num_samples_or_bytes=fft_size * avg_count):
            
            # --- Perform FFT ---
            # Use a window function
            window = np.hanning(fft_size)
            psd = np.fft.fft(samples * window, fft_size)
            
            # Get magnitude, convert to dB, and normalize
            # We use fftshift to put 0Hz (DC) in the center
            psd_mag = np.abs(np.fft.fftshift(psd))
            psd_db = 20 * np.log10(psd_mag + 1e-9) # 1e-9 to avoid log(0)
            
            # --- Averaging ---
            fft_buffer[buffer_index, :] = psd_db
            buffer_index = (buffer_index + 1) % avg_count
            averaged_psd = np.mean(fft_buffer, axis=0)

            # --- Prepare Data ---
            # The frontend expects a list of dB values
            data = {
                'type': 'spectrum_data',
                'payload': averaged_psd.tolist()
            }
            
            # --- Send to all connected clients ---
            if connected_clients:
                # Use asyncio.gather to concurrently send the message to all clients.
                # The 'return_exceptions=True' handles cases where a client may have
                # disconnected between the check and the send attempt.
                send_tasks = [client.send(json.dumps(data)) for client in connected_clients]
                await asyncio.gather(*send_tasks, return_exceptions=True)

    except asyncio.CancelledError:
        print("SDR stream task cancelled.")
    except Exception as e:
        print(f"Error in SDR stream: {e}")
    finally:
        print("SDR stream stopped.")


async def handler(websocket, path):
    """Handles WebSocket connections and messages."""
    global streaming_task, sdr_config, sdr
    
    # Register client
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Received command: {data.get('command')}")
                
                if data.get('command') == 'start':
                    if not sdr and not init_sdr():
                        await websocket.send(json.dumps({'type': 'error', 'message': 'SDR not found'}))
                        continue
                    
                    if not streaming_task or streaming_task.done():
                        print("Creating new streaming task.")
                        # Start the SDR streaming task
                        streaming_task = asyncio.create_task(sdr_stream(websocket))
                        await websocket.send(json.dumps({'type': 'status', 'message': 'Stream started'}))
                    else:
                        print("Stream already running.")

                elif data.get('command') == 'stop':
                    if streaming_task and not streaming_task.done():
                        streaming_task.cancel()
                        streaming_task = None
                        print("Streaming task cancelled.")
                        await websocket.send(json.dumps({'type': 'status', 'message': 'Stream stopped'}))
                    else:
                        print("Stream already stopped.")
                
                elif data.get('command') == 'update_settings':
                    settings = data.get('settings', {})
                    print(f"Updating settings: {settings}")
                    
                    # Update config
                    sdr_config['center_freq'] = float(settings.get('centerFreq', sdr_config['center_freq']))
                    # The 'span' from the frontend maps to our 'sample_rate'
                    sdr_config['sample_rate'] = float(settings.get('span', sdr_config['sample_rate']))
                    sdr_config['fft_bins'] = int(settings.get('fftBins', sdr_config['fft_bins']))

                    # Re-initialize SDR with new settings
                    if not init_sdr():
                         await websocket.send(json.dumps({'type': 'error', 'message': 'Failed to re-initialize SDR'}))
                    else:
                        await websocket.send(json.dumps({'type': 'status', 'message': 'SDR reconfigured'}))
                        # If the task was running, it will be stopped by init_sdr(),
                        # so we must restart it.
                        if streaming_task and not streaming_task.done():
                            streaming_task.cancel()
                        streaming_task = asyncio.create_task(sdr_stream(websocket))
                        print("Restarting stream with new settings.")


            except json.JSONDecodeError:
                print(f"Received invalid JSON: {message}")
            except Exception as e:
                print(f"Error handling message: {e}")

    finally:
        # Unregister client
        connected_clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")
        # If last client disconnects, stop the stream
        if not connected_clients and streaming_task and not streaming_task.done():
            streaming_task.cancel()
            streaming_task = None
            if sdr:
                sdr.close()
                sdr = None
            print("Last client disconnected, stopping stream and closing SDR.")

async def main():
    host = '0.0.0.0' # Listen on all network interfaces
    port = 8765
    print(f"Starting WebSocket server on ws://{host}:{port}...")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    if not init_sdr():
        print("---!!! FAILED TO INITIALIZE SDR. EXITING. !!!---")
        print("---!!! Please check connection and udev rules.  !!!---")
    else:
        asyncio.run(main())