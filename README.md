# Exercise 2: Realtime API and Tools

In Exercise 1, you built a voice agent using a pipeline of specialised models: STT → LLM → TTS. It works, but there's a cost. Every step adds latency, and something important gets lost along the way.

In this exercise, we'll switch to OpenAI's Realtime API and add tool use. You'll see why this combination is such a powerful pattern for building voice-first products.

## What Gets Lost in the Pipeline

When speech passes through STT, it becomes text. That text captures *what* you said, but not *how* you said it.

The "how" is called **prosody**:
- Intonation (pitch rising or falling)
- Emphasis ("I didn't say *he* stole it" vs "I didn't say he *stole* it")
- Hesitation, pauses, rhythm
- Emotion: excitement, frustration, uncertainty

A speech-to-speech model hears all of this. When you trail off mid-sentence, uncertain, the model knows. When you ask a rhetorical question, the model can tell. This makes conversations feel remarkably more natural.

## Why Realtime + Tools Matters

Tools give an LLM the ability to take actions: look things up, save information, control systems. But in a traditional pipeline, tool use is clunky:

1. User speaks → STT → text
2. LLM decides to call a tool → waits for result
3. LLM generates response → TTS → audio
4. User hears the response

Each step adds latency. The pause while the tool executes feels awkward.

With a realtime model, tool calls happen mid-stream. The model can acknowledge your request ("Let me check that for you..."), execute the tool, and continue speaking, all in one fluid interaction. It feels like talking to someone who's actually *doing* something, not just reciting information.

This is the pattern behind the next generation of voice products: assistants that don't just talk, but act.

## What You'll Build

You'll update your agent to:
1. Use the OpenAI Realtime API instead of the STT → LLM → TTS pipeline
2. Add a simple "memory" tool that can save and recall notes

The memory tool is deliberately simple. The point is to feel the difference when an agent can take actions in real-time.

## Prerequisites

You'll need an OpenAI API key with access to the Realtime API. Add it to your `.env.local` if you haven't already:

```
OPENAI_API_KEY=your-openai-api-key
```

## Step 1: Update Dependencies

The Realtime API requires the OpenAI plugin:

```bash
cd ~/livekit-voice-agent/agent
uv add "livekit-agents[openai]~=1.3"
```

## Step 2: Update the Agent Code

Replace the contents of `agent.py` with the following:

```python
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, room_io, function_tool
from livekit.plugins import openai, noise_cancellation

load_dotenv(".env.local")

# Simple in-memory storage for notes
memory = {}


class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
                You are a helpful assistant communicating via voice.
                Keep your responses concise and conversational.
                
                You have the ability to remember things for the user.
                When they ask you to remember something, use the save_note tool.
                When they ask what you've saved or to recall something, use the get_notes tool.
            """,
        )

    @function_tool
    async def save_note(self, note: str) -> str:
        """Save a note to memory. Use this when the user asks you to remember something."""
        note_id = len(memory) + 1
        memory[note_id] = note
        return f"Saved note #{note_id}: {note}"

    @function_tool
    async def get_notes(self) -> str:
        """Retrieve all saved notes. Use this when the user asks what you've remembered."""
        if not memory:
            return "No notes saved yet."
        return "\n".join([f"#{id}: {note}" for id, note in memory.items()])


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="alloy",
            model="gpt-realtime-mini",
        )
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
        instructions="Greet the user and let them know you can remember things for them."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
```

## Understanding the Changes

### Realtime Model

```python
session = AgentSession(
    llm=openai.realtime.RealtimeModel(
        voice="alloy",
        model="gpt-realtime-mini",
    )
)
```

This replaces the entire STT → LLM → TTS pipeline with a single speech-to-speech model. We use `gpt-realtime-mini` as it's more cost-effective for learning. The `voice` parameter controls the output voice: options include `alloy`, `coral`, `echo`, `sage`, and others.

### Function Tools

```python
@function_tool
async def save_note(self, note: str) -> str:
    """Save a note to memory. Use this when the user asks you to remember something."""
    note_id = len(memory) + 1
    memory[note_id] = note
    return f"Saved note #{note_id}: {note}"
```

The `@function_tool` decorator exposes a method as a tool the model can call. Note that tool functions must be `async`, even if they don't perform any asynchronous operations. The docstring is important: it tells the model when and how to use the tool.

When you speak to the agent and say "Remember that my meeting is at 3pm", the model will:
1. Recognise this as a request to save information
2. Call `save_note` with the extracted content
3. Incorporate the result into its spoken response

This all happens in a single conversational turn, with minimal latency.

## Step 3: Test Your Agent

Start the agent:

```bash
uv run agent.py dev
```

Start the frontend (in another terminal):

```bash
cd ~/livekit-voice-agent/frontend
npm run dev
```

Open `http://localhost:3000` and try:

- "Remember that I need to buy milk"
- "Also remember my dentist appointment is on Friday"
- "What have you saved for me?"

Notice how the agent acknowledges each request naturally, without the awkward pauses you might expect from a tool call.

## Experiencing Prosody

Now that your agent is running, try these experiments to feel the difference a speech-to-speech model makes:

**Interruption handling:**
Say "Remember that... actually, never mind." Notice how the agent handles your change of intent mid-sentence. A pipeline would have already committed to saving a note.

**Sarcasm:**
Try asking "Oh great, another thing to remember" in a sarcastic tone, then try again with genuine enthusiasm. Does the agent respond differently?

**Uncertainty:**
Ask a question while trailing off: "Could you maybe... I don't know... save something for me?" Compare this to asking the same thing confidently and directly.

**Emotion:**
Tell the agent something exciting ("I just got the job!") versus something disappointing ("I didn't get the job..."). Notice how it adjusts its response.

These nuances are largely invisible to a pipeline that converts everything to flat text first.

## What You Should Notice

1. **Lower latency**: Responses start faster because there's no STT step
2. **More natural conversation**: The agent picks up on tone and pacing
3. **Fluid tool use**: Saving and recalling notes feels like a natural part of the conversation

## Troubleshooting

### "No audio" or agent doesn't respond

1. Check that your `OPENAI_API_KEY` is set in `.env.local`
2. Verify your OpenAI account has access to the Realtime API
3. Check the agent terminal for error messages

### Tools aren't being called

1. Make sure your docstrings clearly describe when to use each tool
2. Try being more explicit: "Please save a note that says..."
3. Check the agent logs to see if the model is attempting tool calls

### "TypeError: object str can't be used in 'await' expression"

Your tool functions need to be `async`. Make sure you have:

```python
@function_tool
async def save_note(self, note: str) -> str:  # Note the 'async' keyword
```

Not:

```python
@function_tool
def save_note(self, note: str) -> str:  # Missing 'async'
```

## Stretch Goal

**Build a prototype for your startup/employer.**

You now have all the pieces. What would be genuinely useful? Some ideas:

- **Personal assistant**: Calendar management, reminders, email summaries
- **Customer service agent**: Answer FAQs, look up order status, process returns
- **Language tutor**: Conversational practice with corrections and encouragement
- **Interview coach**: Mock interviews with feedback on your answers
- **Therapy companion**: Active listening, mood tracking, coping strategies
- **Medical triage**: Symptom assessment, appointment booking, medication reminders
- **Legal assistant**: Document explanation, deadline tracking, case research

Pick something you'd actually use. Change the `instructions` to give your agent a persona. Add tools that connect to real APIs. The pattern is always the same: a clear persona, tools that take actions, and a realtime model that makes it feel human.

## Cost Comparison

The Realtime API is priced per token. Audio tokens work differently from text tokens, but here's the current pricing:

| Model | Input | Output |
|-------|-------|--------|
| gpt-realtime | $32.00 / 1M tokens | $64.00 / 1M tokens |
| gpt-realtime-mini | $10.00 / 1M tokens | $20.00 / 1M tokens |

For comparison, a pipeline using GPT-4.1 mini with separate STT and TTS is roughly 10x cheaper per minute of conversation. The tradeoff is user experience: for products where conversation quality matters (customer service, therapy apps, companionship) the cost is often worth it.

## Quiz Questions

<details>
<summary><strong>1. Why does the Realtime API feel more natural than a pipeline?</strong></summary>

The Realtime API processes speech directly without converting to text first. This preserves prosodic information (tone, emphasis, hesitation, emotion) that gets lost in STT transcription. The model understands not just what you said, but how you said it.

</details>

<details>
<summary><strong>2. What makes tool use different with the Realtime API?</strong></summary>

With a pipeline, tool calls create noticeable pauses: the system must wait for STT, then LLM processing, then the tool, then TTS. With the Realtime API, tool calls happen mid-stream. The model can speak while waiting for results, or incorporate results immediately. The interaction feels continuous rather than turn-based.

</details>

<details>
<summary><strong>3. Why is the docstring important for function tools?</strong></summary>

The docstring tells the model when to use the tool and what it does. The model reads this description to decide whether a tool is appropriate for the user's request. A clear, specific docstring leads to better tool selection.

</details>

<details>
<summary><strong>4. What happens to the saved notes when the agent restarts?</strong></summary>

They're lost. The `memory` dictionary exists only in the Python process's memory. For persistence, you'd need to store notes in a database or file. This is intentional for the exercise: it keeps the code simple while demonstrating the tool pattern.

</details>

<details>
<summary><strong>5. What happens if two users connect at the same time?</strong></summary>

They share the same `memory` dictionary, since it's defined at module level. User A could save a note and User B could retrieve it. In a production system, you'd scope memory per session or per user, perhaps using `ctx.room.name` as a key.

</details>

<details>
<summary><strong>6. How would you add a new tool to this agent?</strong></summary>

Add a new method to the `VoiceAgent` class with the `@function_tool` decorator. Include a clear docstring explaining when to use it. The method should be `async`, take typed parameters, and return a string result that the model can incorporate into its response.

```python
@function_tool
async def get_time(self) -> str:
    """Get the current time. Use when the user asks what time it is."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M")
```

</details>