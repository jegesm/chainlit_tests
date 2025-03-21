from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from typing import *
from langchain.tools import BaseTool
from langchain_core.output_parsers.string import StrOutputParser

import chainlit as cl
from chainlit.sync import run_sync

import os

token = ""
with open("/v/wfct0p/API-tokens/openai-api.token") as f:
    token = f.read().strip()
os.environ['OPENAI_API_KEY'] = token

class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name: str = "human"
    description: str = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["output"]


@cl.on_chat_start
def start():
    ollama_host = "http://wfct0p-ollamaapi:11434/v1"
    model="llama3.1"
    llm_agent = ChatOpenAI(temperature=0, streaming=True, model_name=model, openai_api_base=ollama_host)
    chain = llm_agent | StrOutputParser()
    llm_agent = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4o-mini")
    llm_math_chain = LLMMathChain.from_llm(llm=llm_agent, verbose=True)

    tools = [
        HumanInputChainlit(),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
            coroutine=llm_math_chain.arun,
        ),
    ]
    agent = initialize_agent(
        tools, llm_agent, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True#, handle_parsing_errors=True
    )

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()
