# (base) kevin@4090:~/llama_index_demo$ cat lats_agent_demo3.py
# conda activate ReAct-Llamaindex-demo
#
import asyncio
import os

from llama_index.agent.lats import LATSAgentWorker
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
    Settings,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata



import nest_asyncio
nest_asyncio.apply()


# Define LLM and embedding model
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.3:latest", api_base="http://localhost:11434/v1",format="json",request_timeout=2400,is_chat_model=True)

from llama_index.embeddings.ollama import OllamaEmbedding


embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large:latest",
    #model_name="chatfire/bge-m3:q8_0",
    base_url="http://localhost:11434",
    )

Settings.llm = llm
Settings.embed_model = embed_model


try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/zte"
    )
    zte_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/fh"
    )
    fh_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False


if not index_loaded:
    # load data
    zte_docs = SimpleDirectoryReader(
        input_files=["./data/10k/zte_2023.pdf"]
    ).load_data()
    fh_docs = SimpleDirectoryReader(
        input_files=["./data/10k/fh_2023.pdf"]
    ).load_data()

    # build index
    zte_index = VectorStoreIndex.from_documents(zte_docs, embed_model=Settings.embed_model)
    fh_index = VectorStoreIndex.from_documents(fh_docs, embed_model=Settings.embed_model)

    # persist index
    zte_index.storage_context.persist(persist_dir="./storage/zte")
    fh_index.storage_context.persist(persist_dir="./storage/fh")


zte_engine = zte_index.as_query_engine(similarity_top_k=3)
fh_engine = fh_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=zte_engine,
        metadata=ToolMetadata(
            name="zte_10k",
            description=(
"提供有关中兴通讯2023年财务状况的信息。"
"使用详细的纯文本问题作为工具的输入。"
"该输入将用于驱动一个语义搜索引擎。"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=fh_engine,
        metadata=ToolMetadata(
            name="fh_10k",
            description=(
"提供有关烽火通信公司2023年财务状况的信息。"
"使用详细的纯文本问题作为工具的输入。"
"该输入将用于驱动一个语义搜索引擎。"
            ),
        ),
    ),
]

# Define asynchronous chat function
async def chat_async(agent, message):
    response = await agent.achat(message)
    return response

# Instantiate LATS agent
agent_worker = LATSAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,
    num_expansions=2,
    max_rollouts=3,
    verbose=True,
)
agent = agent_worker.as_agent()

# Define the task
message = (
"鉴于在烽火通信和中兴通讯的2023年报文件中财务数据，"
"哪家公司表现得更好？请使用具体数字来支持您的决定。"
)

# Run the agent asynchronously
response = asyncio.run(chat_async(agent, message))

print(str(response))
