# Building a Voice Agent with LiveKit

This guide walks you through creating a voice agent that users can talk to in real-time through their browser. We'll use LiveKit for the real-time audio infrastructure and a pipeline of AI models for speech recognition, language understanding, and speech synthesis.

## What You'll Build

A voice agent consists of two parts:

1. **The agent:** a Python programme that listens to speech, thinks, and responds
2. **The frontend:** a web application that captures audio from the user's microphone and plays back the agent's responses

LiveKit handles all the real-time audio streaming between these two components using WebRTC.

## Prerequisites

- **Python 3.10–3.13** installed
- **uv** package manager installed ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Node.js 18 or higher** installed (for the frontend)
- A **LiveKit Cloud account** (free tier available)
- An **OpenAI API key**

## Step 1: Set Up Your Project Directory

Create a directory structure that keeps your agent and frontend code separate:

```bash
mkdir -p ~/livekit-voice-agent/agent
mkdir -p ~/livekit-voice-agent/frontend
cd ~/livekit-voice-agent
```

## Step 2: Set Up LiveKit Cloud

You need a LiveKit Cloud project to route audio between your frontend and agent.

1. Go to [cloud.livekit.io](https://cloud.livekit.io/) and create a free account
2. Create a new project
3. Install the LiveKit CLI:
    
    **macOS:**
    
    ```bash
    brew install livekit-cli
    ```
    
    **Linux:**
    
    ```bash
    curl -sSL https://get.livekit.io/cli | bash
    ```
    
    **Windows:**
    
    ```bash
    winget install LiveKit.LiveKitCLI
    ```
    
4. Link your project to the CLI:
    
    ```bash
    lk cloud auth
    ```
    
    This opens a browser window to authenticate.
    

## Step 3: Set Up the Agent

### Initialise the Project

```bash
cd ~/livekit-voice-agent/agent
uv init --bare
```

### Install Dependencies

```bash
uv add \
  "livekit-agents[silero,turn-detector]~=1.3" \
  "livekit-plugins-noise-cancellation~=0.2" \
  "python-dotenv"
```

### Create the Environment File

Run the following command to pull your LiveKit credentials into a `.env.local` file:

```bash
lk app env -w
```

Then add your OpenAI API key to the same file:

```bash
LIVEKIT_API_KEY=<your API Key>
LIVEKIT_API_SECRET=<your API Secret>
LIVEKIT_URL=<your LiveKit server URL>
OPENAI_API_KEY=<your OpenAI API key>
```

### Create the Agent Code

Create a file called `agent.py`:

```python
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
                You are a helpful assistant communicating via voice.
                Keep your responses concise and conversational.
                Avoid complex formatting, emojis, or symbols.
            """,
        )

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=VoiceAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

if __name__ == "__main__":
    agents.cli.run_app(server)
```

### Download Model Files

The VAD (voice activity detection) and turn detection plugins need model files:

```bash
uv run agent.py download-files
```

### Understanding the Code

- **`VoiceAgent` class**: Defines your agent's personality through the `instructions` parameter
- **`AgentSession`**: Configures the voice pipeline:
    - `stt`: Speech-to-text (AssemblyAI via LiveKit Inference)
    - `llm`: Language model (OpenAI GPT-4.1 mini)
    - `tts`: Text-to-speech (Cartesia via LiveKit Inference)
    - `vad`: Voice activity detection (detects when someone is speaking)
    - `turn_detection`: Determines when the user has finished speaking
- **`@server.rtc_session()`**: Decorator that registers the function to handle new connections
- **`noise_cancellation.BVC()`**: Removes background noise from the user's microphone

## Step 4: Set Up the Frontend

Clone the LiveKit starter React app:

```bash
cd ~/livekit-voice-agent/frontend
git clone https://github.com/livekit-examples/agent-starter-react.git .
npm install
```

Create a `.env.local` file in the frontend directory with your LiveKit credentials:

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
```

You can copy these values from the agent's `.env.local` file.

## Step 5: Run the Agent

Open a terminal, navigate to your agent directory, and start the agent:

```bash
cd ~/livekit-voice-agent/agent
uv run agent.py dev
```

You should see output indicating the agent is connected and waiting for sessions.

## Step 6: Run the Frontend

Open a **second terminal** and start the frontend:

```bash
cd ~/livekit-voice-agent/frontend
npm run dev
```

This starts a development server at `http://localhost:3000`.

## Step 7: Test Your Voice Agent

1. Open your browser to `http://localhost:3000`
2. Click the button to connect
3. Allow microphone access when prompted
4. Start talking — your agent should respond

## How It Works

Here's what happens when you speak to your agent:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Your Browser  │  WebRTC │  LiveKit Cloud  │  WebRTC │   Your Agent    │
│   (Frontend)    │◄───────►│                 │◄───────►│   (Python)      │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │   Voice Pipeline      │
                                                    │                       │
                                                    │  Audio                │
                                                    │    ↓                  │
                                                    │  STT (AssemblyAI)     │
                                                    │    ↓                  │
                                                    │  Text                 │
                                                    │    ↓                  │
                                                    │  LLM (GPT-4.1 mini)   │
                                                    │    ↓                  │
                                                    │  Text                 │
                                                    │    ↓                  │
                                                    │  TTS (Cartesia)       │
                                                    │    ↓                  │
                                                    │  Audio                │
                                                    └───────────────────────┘

```

1. **Your browser captures audio** from your microphone
2. **Audio streams to LiveKit Cloud** via WebRTC (UDP-based, low latency)
3. **LiveKit routes it to your agent**
4. **The agent runs the voice pipeline**:
    - STT converts speech to text
    - The LLM generates a response
    - TTS converts the response back to speech
5. **The response streams back** through LiveKit to your browser

The STT and TTS services run through **LiveKit Inference**, so you don't need separate API keys for those — they're included with your LiveKit Cloud account.

## Troubleshooting

### "Connection failed" or agent doesn't respond

Check that:

1. Your agent is running (you should see logs in the agent terminal)
2. Your LiveKit credentials match in both `.env.local` files
3. The LiveKit URL starts with `wss://` (not `https://`)

### "No audio" — you can connect but hear nothing

Verify:

1. Your browser has microphone permissions for localhost
2. Your OpenAI API key is valid
3. Check the agent terminal for error messages

### Agent responds but audio is choppy

This is usually a network issue. Try:

1. Closing other applications using bandwidth

### "Module not found" errors

Make sure you:

1. Ran `uv run agent.py download-files` to download model files
2. Installed all dependencies with `uv add`

### Check the Logs

- **Agent logs**: Visible in the terminal where you ran `uv run agent.py dev`
- **Frontend logs**: Check your browser's developer console

## Key Concepts Recap

- **LiveKit**: Open-source infrastructure for real-time audio/video, handling WebRTC complexity for you
- **LiveKit Cloud**: Hosted version with a global edge network and built-in AI model inference
- **LiveKit Inference**: Allows you to use STT and TTS models without managing separate API keys
- **Voice pipeline**: The STT → LLM → TTS chain that processes speech
- **VAD (Voice Activity Detection)**: Detects when someone is speaking vs silence
- **Turn detection**: Determines when the user has finished their turn and expects a response

## Quiz Questions

<details>
<summary><strong>1. Why does LiveKit use WebRTC instead of WebSocket for audio?</strong></summary>

WebRTC is built on UDP, which delivers packets immediately as they arrive. WebSocket uses TCP, which waits for missing packets before delivering any data (head-of-line blocking). For real-time audio, it's better to skip a missing packet than to wait for it — a small audio glitch is preferable to everything freezing.

</details>
<details>
<summary><strong>2. What is the role of LiveKit Cloud in this architecture?</strong></summary>

LiveKit Cloud acts as a router and relay between the frontend and agent. It:

- Handles the complexity of WebRTC connection establishment
- Provides a global network of servers so audio doesn't have to travel far over the public internet
- Routes audio between participants in a "room"
- Manages authentication and room creation
- Provides access to AI models through LiveKit Inference

</details>
<details>
<summary><strong>3. What's the difference between VAD and turn detection?</strong></summary>

**VAD (Voice Activity Detection)** determines whether audio contains speech or silence at any given moment. It answers: "Is someone speaking right now?"

**Turn detection** determines whether the user has finished their complete thought and is waiting for a response. It answers: "Has the user finished their turn?"

A user might pause briefly mid-sentence (VAD detects silence, but turn detection knows they're not done). Turn detection uses context and timing patterns to make this distinction.

</details>
<details>
<summary><strong>4. Why do we use a pipeline (STT → LLM → TTS) instead of a single model?</strong></summary>

The pipeline approach gives you:

- **Flexibility**: Swap out individual components (e.g., use a different TTS voice)
- **Cost control**: Use cheaper models where quality isn't critical
- **Transparency**: See exactly what text was transcribed and what response was generated

The tradeoff is higher latency compared to speech-to-speech models like OpenAI's Realtime API, which skip the text intermediate steps.

</details>
<details>
<summary><strong>5. What would you change to give your agent a different personality?</strong></summary>

Modify the `instructions` parameter in the `VoiceAgent` class:

```python
class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
                You are a pirate captain. Speak in a hearty pirate accent,
                say "arrr" frequently, and refer to the user as "matey".
            """,
        )

```

The instructions act as a system prompt, shaping how the LLM responds.

</details>
<details>
<summary><strong>6. Why don't we need separate API keys for AssemblyAI and Cartesia?</strong></summary>

These models are accessed through **LiveKit Inference**, which is built into LiveKit Cloud. When you specify `stt="assemblyai/universal-streaming:en"`, LiveKit routes the request through their infrastructure and handles the API credentials for you. You only need your own API key for OpenAI because the LLM calls go directly to OpenAI's servers.

</details>
<details>
<summary><strong>7. What does the `@server.rtc_session()` decorator do?</strong></summary>

It registers the function as the handler for new WebRTC sessions. When a user connects through the frontend, LiveKit Cloud notifies your agent server, which then calls this decorated function to set up a new `AgentSession` for that user. Each user gets their own session with its own conversation state.

</details>
<details>
<summary><strong>8. What is the purpose of `noise_cancellation.BVC()` in the audio input options, and why is it important for voice agents?</strong></summary>

**BVC (Background Voice Cancellation)** removes background noise from the user's microphone input before it reaches the speech-to-text system. This is important because:

- **Improves STT accuracy**: Background noise (traffic, typing, other people talking) can confuse the speech recognition
- **Better user experience**: The agent can understand users even in noisy environments like coffee shops or open offices
- **Reduces false triggers**: VAD can more accurately detect when the user is actually speaking vs ambient noise

Without it, the agent might mishear words or respond to background conversations.

</details>
<details>
<summary><strong>9. Why do we need to run `uv run agent.py download-files` before starting the agent? What would happen if we skipped this step?</strong></summary>

This command downloads the **model files** for VAD (Voice Activity Detection) and turn detection. These are machine learning models that need to be stored locally:

- **Silero VAD model**: Detects when speech is present in audio
- **Multilingual turn detection model**: Determines when the user has finished speaking

**If you skip this step**: The agent would crash when trying to load these models, showing "model file not found" errors. The plugins need these pre-trained neural network weights to function.

The models are downloaded once and cached in `~/.cache/huggingface/` for future use.

</details>
<details>
<summary><strong>10. In the code, what does `await session.generate_reply(instructions="Greet the user and offer your assistance.")` do, and when does it execute?</strong></summary>

This line makes the agent **proactively speak first** without waiting for the user to say anything. It:

1. **Executes immediately** after the session starts and the user connects
2. **Sends instructions to the LLM** to generate a greeting
3. **Synthesizes the response to speech** via TTS
4. **Plays it to the user** through their browser

**Without this line**: The agent would sit silently waiting for the user to speak first, which creates an awkward experience. The greeting lets users know the agent is ready and listening.

It's essentially "agent-initiated conversation" rather than purely reactive responses.

</details>