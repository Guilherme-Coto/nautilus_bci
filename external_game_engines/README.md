# External Game Engines Integration

This folder contains tools and instructions for integrating external game engines (such as Godot, Unity, or Unreal Engine) with the Nautilus BCI BIDS recording suite.

Because many game engines do not have native, out-of-the-box support for the Lab Streaming Layer (LSL), we provide a lightweight **UDP to LSL Bridge**. This allows any game engine to send simple network messages over UDP, which are then translated into LSL markers and perfectly synchronized with the EEG data in the BIDS dataset.

## 1. Running the UDP to LSL Bridge

The bridge script is a standalone Python application that listens for incoming UDP messages and broadcasts them as LSL markers formatted for the BIDS recorder.

To start the bridge, run:

```bash
uv run python udp_to_lsl_bridge.py
```

By default, it listens on `127.0.0.1:9000`. You can configure the port and the LSL stream name:

```bash
uv run python udp_to_lsl_bridge.py --port 9000 --marker-name ExternalGameMarkers
```

## 2. Sending Markers from Your Game Engine

Your game engine simply needs to send a UDP packet containing a JSON string to the bridge's IP and port.

The expected JSON format is:
```json
{
  "name": "Jump_Action",
  "duration": 0.5
}
```
- `name`: (String) The event name you want recorded in your BIDS `_events.tsv` file.
- `duration`: (Float) The duration of the event in seconds. Use `0.0` for instantaneous events.

### Godot 4 Example (GDScript)

Here is a simple Godot 4 script showing how to send these markers:

```gdscript
extends Node

var udp_peer := PacketPeerUDP.new()
var bridge_host := "127.0.0.1"
var bridge_port := 9000

func _ready():
    # Connect to the local UDP bridge
    udp_peer.connect_to_host(bridge_host, bridge_port)
    print("UDP Peer connected to ", bridge_host, ":", bridge_port)

    # Send an initial marker
    send_bci_marker("Game_Started", 0.0)

func send_bci_marker(marker_name: String, duration: float = 0.0):
    # Construct the JSON payload
    var payload = {
        "name": marker_name,
        "duration": duration
    }

    # Convert dictionary to JSON string
    var json_string = JSON.stringify(payload)

    # Send the packet over UDP
    udp_peer.put_packet(json_string.to_utf8_buffer())
    print("Sent marker: ", marker_name)

# Example usage on input action
func _input(event):
    if event.is_action_pressed("ui_accept"):
        # Send a marker indicating a jump event with a 0.5s expected duration
        send_bci_marker("Player_Jump", 0.5)
```

## 3. Recording the Data

1. Start the main `run_bci_suite.py` application.
2. Start the `udp_to_lsl_bridge.py`.
3. In the BCI suite, start the EEG streamer and begin a **BIDS Recording**. The recorder will automatically detect the "ExternalGameMarkers" LSL stream and include it in the multimodal dataset.
4. Launch your external game (e.g., in Godot) and play. The markers will be saved into the `_events.tsv` files alongside the EEG data!
