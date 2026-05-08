import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import asyncio
import chainlit as cl
from src.rag.graph import execute_workflow


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="""\
<div style="text-align:center; padding: 24px 0 8px 0;">
  <h2 style="margin: 0 0 6px 0; font-size: 1.5rem;">📄 ArXiv Paper Chatbot</h2>
  <p style="margin: 0 0 16px 0; color: #888; font-size: 0.9rem;">
    Hybrid search (BM25 + semantic) over 40 AI/ML research papers
  </p>
</div>

**Try asking:**
- How does retrieval augmented generation work with large language models?
- What is instruction fine-tuning and how does it improve LLM behaviour?
- How do diffusion models generate images and what is the denoising process?
- What is chain-of-thought prompting and why does it improve reasoning?
- How does reinforcement learning from human feedback align language models?
""").send()


@cl.on_message
async def on_message(message: cl.Message):
    loop = asyncio.get_event_loop()

    async with cl.Step(name="⚙️  Processing", show_input=False) as parent:
        async with cl.Step(name="✏️  Rewriting query", show_input=False) as step:
            step.output = '<span style="font-size:0.72rem; color:#888;">Improving query for retrieval...</span>'

        async with cl.Step(name="🔍  Retrieving & grading", show_input=False) as step:
            step.output = '<span style="font-size:0.72rem; color:#888;">Searching papers with hybrid search...</span>'

        async with cl.Step(name="⚡  Generating answer", show_input=False) as step:
            result = await loop.run_in_executor(None, execute_workflow, message.content)
            step.output = '<span style="font-size:0.72rem; color:#888;">✅ Done</span>'

        parent.output = '<span style="font-size:0.72rem; color:#888;">Completed</span>'

    await cl.Message(content=
        f'<div style="font-size:0.88rem; line-height:1.7; color:#333;">'
        f'{result["answer"]}'
        f'</div>'
    ).send()

    if result["citation"]:
        citations_html = "".join(
            f'<li style="margin: 4px 0;">'
            f'<span style="font-size:0.78rem; color:#555;">'
            f'<b>{c["title"]}</b> &nbsp;'
            f'<code style="font-size:0.72rem; background:#f0f0f0; padding:1px 5px; border-radius:4px;">{c["arxiv_id"]}</code>'
            f'&nbsp; p.{c["page"]}'
            f'</span></li>'
            for c in result["citation"]
        )
        await cl.Message(content=
            f'<div style="margin-top:8px; border-top: 1px solid #e0e0e0; padding-top:8px;">'
            f'<p style="font-size:0.8rem; font-weight:600; color:#666; margin:0 0 6px 0;">📚 Sources</p>'
            f'<ul style="margin:0; padding-left:16px; list-style:disc;">{citations_html}</ul>'
            f'</div>'
        ).send()
