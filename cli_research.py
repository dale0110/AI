#ReAct-Llamaindex-demo) kevin@4090:~/awesome-llm-apps/research_agent$ python cli_research.py 
#(base) kevin@4090:~/awesome-llm-apps/research_agent$ cat cli_research.py 
# Displaying final output format
from IPython.display import display, Markdown, Latex
# LangChain Dependencies
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import END, StateGraph
# For State Graph 
from typing_extensions import TypedDict
#import streamlit as st
import os
# 创建占位符
#result_placeholder = st.empty()
# Defining LLM
def configure_llm():
    #st.sidebar.header("Configure LLM")
    
    # Model Selection
    model_options = ["deepseek-r1:32b"]
    selected_model = "deepseek-r1:32b" #st.sidebar.selectbox("Choose the LLM Model", options=model_options, index=0)
    
    # Temperature Setting
    temperature =  0.7 #st.sidebar.slider("Set the Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
   
    # Create LLM Instances based on user selection
    llama_model = ChatOpenAI(model=selected_model, temperature=temperature,base_url="http://localhost:11434/v1",timeout=800)
    llama_model_json = ChatOpenAI(model=selected_model, model_kwargs={"response_format": {"type": "json_object"}}, temperature=temperature,base_url="http://localhost:11434/v1",timeout=800)
    
    return llama_model, llama_model_json

# Streamlit Application Interface
#st.title("Personal Research Assistant powered By Llama3.2")
llama3, llama3_json=configure_llm()
wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
generate_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    You are an AI assistant for Research Question Tasks. First analyze the web search results, then synthesize them to answer the question.
    Strictly use the following pieces of web search context to answer the question. If you don't know the answer, just say that you don't know.
    Keep the answer concise, but provide all of the details you can in the form of a research report.
    Only make direct references to material if provided in the context.
    
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    
    Question: {question} 
    Web Search Context: {context} 
    Answer: 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

# Chain
generate_chain = generate_prompt | llama3 | StrOutputParser()
router_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|>
    
    You are an expert at routing a user question to either the generation stage or web search.
    By default, prefer using web search to get the most up-to-date and comprehensive information.
    Only skip web search and go straight to generation if the question is purely about common knowledge, basic concepts, or simple logic.
    You do not need to be stringent with the keywords in the question related to these topics.
    Give a binary choice 'web_search' or 'generate' based on the question.
    Return the JSON with a single key 'choice' with no preamble or explanation.
    
    Question to route: {question} 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["question"],
)

# Chain
question_router = router_prompt | llama3_json | JsonOutputParser()

query_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    You are an expert at crafting web search queries for research questions.
    More often than not, a user will ask a basic question that they wish to learn more about, however it might not be in the best format. 
    Reword their query to be the most effective web search string possible.
    Return the JSON with a single key 'query' with no premable or explanation. 
    
    Question to transform: {question} 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["question"],
)

# Chain
query_chain = query_prompt | llama3_json | JsonOutputParser()
# Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search_query: revised question for web search
        context: web_search result
    """
    question : str
    generation : str
    search_query : str
    context : str

# Node - Generate

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print("Step: Generating Final Response")
    question = state["question"]
    context = state["context"]

    # Answer Generation
    generation = generate_chain.invoke({"context": context, "question": question})
    print(generation)
    return {"generation": generation}

# Node - Query Transformation

def transform_query(state):
    """
    Transform user question to web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended search query
    """
    
    print("Step: Optimizing Query for Web Search")
    question = state['question']
    gen_query = query_chain.invoke({"question": question})
    search_query = gen_query["query"]
    return {"search_query": search_query}


# Node - Web Search

def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to context
    """

    search_query = state['search_query']
    print(f'Step: Searching the Web for: "{search_query}"')
    
    # Web search tool call
    search_result = web_search_tool.invoke(search_query)
    print(search_result)
    return {"context": search_result}


# Conditional Edge, Routing

def route_question(state):
    """
    route question to web search or generation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("Step: Routing Query")
    question = state['question']
    output = question_router.invoke({"question": question})
    print(output)
    if output['choice'] == "web_search":
        print("Step: Routing Query to Web Search")
        return "websearch"
    elif output['choice'] == 'generate':
        print("Step: Routing Query to Generation")
        return "generate"
# Build the nodes
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)

# Build the edges
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "websearch")
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

# Compile the workflow
local_agent = workflow.compile()
def run_agent(query):
    output = local_agent.invoke({"question": query})
    print("=======")

    return output["generation"]
user_query = "特斯拉的自动驾驶技术前景"#st.text_input("Enter your research question:", "")

result = run_agent(user_query)
print(result)
'''
if st.button("Run Query"):
    if user_query:
        # 使用占位符更新结果，而不是直接使用st.write
        with result_placeholder.container():
            result = run_agent(user_query)
            st.write(result)
'''
