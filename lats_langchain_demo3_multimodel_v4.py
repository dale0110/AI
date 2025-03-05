#(Reflexion-Langchain-local) kevin@4090:~$ cat langchain_demo/lats_langchain_demo3_multimodel_v4.py
from __future__ import annotations  # noqa: F404
import time
import datetime

import code, traceback, signal
import functools
import inspect


def debug_log(*args, **kwargs):
    # 获取调用者的堆栈帧
    frame = inspect.currentframe().f_back
    # 当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 文件名
    filename = frame.f_code.co_filename
    # 行号
    lineno = frame.f_lineno
    # 调用函数名称
    function_name = frame.f_code.co_name

    # 构造输出信息
    msg = (f"[{current_time}] {filename}:{lineno} in {function_name}() - "
           f"args: {args}, kwargs: {kwargs}")
    print(msg)

def debug_call_func(func):
    """
    装饰器：用于调试函数，输出输入参数和输出结果。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 获取函数参数名和默认值
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        # 输出输入参数
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} call {func.__name__} func:", end=' ')
        for name, value in bound_args.arguments.items():
            print(f" {name} = {value}",end='; ')
        print(f"\n")    
        # 调用原始函数并获取输出
        result = func(*args, **kwargs)

        # 输出输出结果
        print(f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {func.__name__} func return result = {result} ")
        return result
    
    return wrapper

'''处理 JSON 字符串
处理 Unicode 转义序列
处理嵌套的内容结构'''
def _clean_text(text):
    """清理和解码文本内容"""
    if isinstance(text, str):
        try:
            # 尝试解析JSON字符串
            import json
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    # 提取content字段并处理Unicode编码
                    content = data.get('content', '')
                    return content.encode().decode('unicode_escape')
                elif isinstance(data, list):
                    # 如果是列表，处理每个项目
                    contents = []
                    for item in data:
                        if isinstance(item, dict):
                            content = item.get('content', '')
                            contents.append(content.encode().decode('unicode_escape'))
                    return ' '.join(contents)
            except json.JSONDecodeError:
                # 如果不是JSON，直接处理Unicode编码
                return text.encode().decode('unicode_escape')
        except UnicodeError:
            # 如果解码失败，返回原始文本
            return text
    return str(text)


import getpass
import os


def _set_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)


# Optional: Configure tracing to visualize and debug the agent
_set_if_undefined("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LATS"

_set_if_undefined("TAVILY_API_KEY")


#The root of the search tree
#The user input

import math
from collections import deque
from typing import Optional, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain.tools import Tool


class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional[Node] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f""
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children


    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent



#The graph state
from typing_extensions import TypedDict


class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str


#define Language Agent
from langchain_openai import ChatOpenAI

#llm = ChatOpenAI(model="gpt-4o-mini")  ChatOllama  from langchain_ollama import ChatOllama
#llm = ChatOpenAI(model="llama3.1:8b-instruct-q4_1",temperature=0.0,base_url="http://localhost:11434/v1")
from langchain_ollama import ChatOllama
llm = ChatOllama(model="deepseek-r1:32b",temperature=0.7,base_url="http://localhost:11434",timeout=2400)

#tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.prebuilt import ToolNode

search = TavilySearchAPIWrapper()
tavily_search_results_json = TavilySearchResults(api_wrapper=search, max_results=5).run
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tools = [tavily_tool]
tool_executor = ToolExecutor(tools=[
    Tool(
        name="tavily_search_results_json",
        func=tavily_search_results_json,
        description="Search the web for relevant information"
    )
])

#Reflection
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import chain as as_runnable


class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


# 修改反思提示模板，使其更明确要求JSON格式输出
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a reflection assistant. You MUST respond in valid JSON format with these fields:\n"
        "{\n"
        '  "reflections": "your critique of the response quality",\n'
        '  "score": <number between 0-10>,\n'
        '  "found_solution": <true or false>\n'
        "}\n\n"
        "Analyze the response quality focusing on:\n"
        "1. Completeness and accuracy\n"
        "2. Clarity and organization\n"
        "3. Use of provided information\n"
        "DO NOT include any text outside the JSON structure."
    ),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="candidate"),
])

@as_runnable
#@debug_call_func
def reflection_chain(inputs) -> Reflection:
    input_text = inputs["input"]
    candidate_messages = inputs["candidate"]
    
    # 确保 candidate_messages 是正确的消息格式
    if not isinstance(candidate_messages, list):
        candidate_messages = [AIMessage(content=str(candidate_messages))]
    
    # 构建消息历史
    messages = []
    for msg in candidate_messages:
        if isinstance(msg, (AIMessage, HumanMessage)):
            # 移除 <think> 标签之间的内容
            content = msg.content
            import re
            content = re.sub(r'<think>.*?</think>\n*', '', content, flags=re.DOTALL)
            # 清理和解码内容
            content = _clean_text(content)
            messages.append(f"Assistant: {content}")
        elif isinstance(msg, ToolMessage):
            # 清理和解码工具消息内容
            content = _clean_text(msg.content)
            messages.append(f"Tool Response: {content}")
        else:
            # 处理其他类型的消息
            content = _clean_text(str(msg))
            messages.append(f"Message: {content}")
    
    # 将消息历史组合成一个字符串
    candidate_text = "\n".join(messages)
    debug_log(len(candidate_text))
    
    # 修改提示模板的使用方式
    formatted_messages = [
        SystemMessage(content=(
            "You are a reflection assistant. You MUST respond in valid JSON format with these fields:\n"
            "{\n"
            '  "reflections": "your critique of the response quality",\n'
            '  "score": <number between 0-10>,\n'
            '  "found_solution": <true or false>\n'
            "}\n\n"
            "Analyze the response quality focusing on:\n"
            "1. Completeness and accuracy\n"
            "2. Clarity and organization\n"
            "3. Use of provided information\n"
            "DO NOT include any text outside the JSON structure."
        )),
        HumanMessage(content=f"Input question: {input_text}\n\nCandidate response:\n{candidate_text}")
    ]
    
    # 使用 LLM 生成反思
    reflection_llm = ChatOpenAI(
        model="qwen2.5:14b",
        temperature=0.2,  # 降低温度以获得更一致的输出
        base_url="http://localhost:11434/v1",
        timeout=2400
    )
    
    # 添加重试逻辑
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = reflection_llm.invoke(formatted_messages)
            debug_log(response)
            
            # 清理响应内容，确保只包含JSON
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            result = json.loads(content)
            
            # 验证必需的字段
            required_fields = ["reflections", "score", "found_solution"]
            if all(field in result for field in required_fields):
                return Reflection(
                    reflections=result["reflections"],
                    score=result["score"],
                    found_solution=result["found_solution"]
                )
            
            raise ValueError("Missing required fields in JSON response")
            
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries - 1:
                print(f"Final error parsing reflection response: {e}")
                print(f"Raw response: {response.content}")
                return Reflection(
                    reflections="Failed to parse reflection after multiple attempts",
                    score=0,
                    found_solution=False
                )
            formatted_messages.append(HumanMessage(content=(
                "Your previous response was not in the correct JSON format. "
                "Please respond ONLY with a valid JSON object containing exactly these fields:\n"
                "{\n"
                '  "reflections": "your critique",\n'
                '  "score": <number 0-10>,\n'
                '  "found_solution": <true/false>\n'
                "}"
            )))

#Initial Response
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant. You can use the following tools to help answer questions:\n"
            "- TavilySearchResults: Search the internet for current information\n"
            "When you need to use a tool, write it in your response like this:\n"
            "To search: <search>your search query</search>\n"
            "Wait for the search results before continuing.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

# 移除工具绑定，使用普通chain
initial_answer_chain = prompt_template | llm.with_config(
    run_name="GenerateInitialCandidate"
)


parser = JsonOutputToolsParser(return_id=True)

'''initial_response = initial_answer_chain.invoke(
    {"input": "Write a research report on lithium pollution."}
)
print(initial_response)'''



#Starting Node
import json


# Define the node we will add to the graph
def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial candidate response."""
    system_prompt = """You are an AI assistant that MUST search for information before providing answers. Follow these rules strictly:

1. ALWAYS start by searching for relevant information using [Search] format
2. DO NOT generate responses based on prior knowledge without searching first
3. Use multiple focused searches to gather comprehensive information
4. After getting search results, organize the information into your response

For example, if asked about frameworks, you should search like this:
[Search] popular agent frameworks GitHub repositories
[Search] LangChain features and license
[Search] Rasa framework comparison

Remember: NEVER provide information without searching first!

Available tools:
- TavilySearchResults: Search the internet for current information

Format: [Search] your search query"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["input"]},
        {"role": "system", "content": "Remember: You MUST start with at least one search query before providing any information."}
    ]
    
    res = llm.invoke(messages)
    debug_log(f"Initial response: {res}")
    
    # Extract search queries from the response
    content = res.content
    import re
    
    # 修改正则表达式以更准确地匹配搜索查询
    search_queries = []
    
    # 使用更严格的正则表达式来匹配搜索查询
    search_patterns = re.findall(r'\[Search\]\s*([^\[\n]+)', content, re.IGNORECASE)
    
    for pattern in search_patterns:
        query = pattern.strip()
        if query:
            search_queries.append(query)
    
    debug_log(f"Extracted search queries: {search_queries}")
    
    # 如果没有找到搜索查询，强制要求模型重新生成响应
    if not search_queries:
        retry_prompt = """You MUST start with search queries before providing any information. Please reformulate your response starting with at least one search using the [Search] format. For example:
[Search] your search query

Please try again, starting with relevant searches."""
        
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": retry_prompt}
        ]
        
        res = llm.invoke(retry_messages)
        content = res.content
        
        # 再次尝试提取搜索查询
        search_patterns = re.findall(r'\[Search\]\s*([^\[\n]+)', content, re.IGNORECASE)
        for pattern in search_patterns:
            query = pattern.strip()
            if query:
                search_queries.append(query)
        
        debug_log(f"Retry - Extracted search queries: {search_queries}")
        
        # 如果重试后仍然没有搜索查询，强制执行一个默认搜索
        if not search_queries:
            default_query = state["input"].strip()
            search_queries.append(default_query)
            debug_log(f"Using default search query: {default_query}")
    
    # Execute tool calls
    tool_responses = []
    for query in search_queries:
        search_result = tool_executor.invoke(
            ToolInvocation(
                tool="tavily_search_results_json",
                tool_input=query
            )
        )
        # 清理和解码搜索结果
        if isinstance(search_result, (list, dict)):
            search_result = json.dumps(search_result, ensure_ascii=False)
        search_result = _clean_text(search_result)
        tool_responses.append(search_result)
    
    debug_log(f"Tool responses length: {len(tool_responses)}")
    
    # Format tool responses as messages
    output_messages = [res]
    for i, response in enumerate(tool_responses):
        tool_message = ToolMessage(
            content=json.dumps(response),
            tool_call_id=f"search_{i}"
        )
        output_messages.append(tool_message)
    
    # Get reflection on the response
    reflection = reflection_chain.invoke(
        {"input": state["input"], "candidate": output_messages}
    )
    
    root = Node(output_messages, reflection=reflection)
    return {
        **state,
        "root": root,
    }

#Candidate Generation
# This generates N candidate values
# for a single input to sample actions from the environment
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
    """Generate multiple candidate responses for a given input."""
    # Get number of candidates to generate (default to 3 if not specified)
    n = min(config["configurable"].get("N", 3), 3)  # Limit max to 3 for Ollama
    
    try:
        # Extract and convert messages to format compatible with Ollama
        if isinstance(messages, ChatPromptValue):
            prompt_messages = messages.messages
        else:
            prompt_messages = messages

        # Convert messages to Ollama-compatible format
        formatted_messages = []
        for msg in prompt_messages:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                # Map message types to Ollama roles
                role = {
                    HumanMessage: "user",
                    AIMessage: "assistant",
                    SystemMessage: "system"
                }.get(type(msg), "user")
                
                formatted_messages.append({
                    "role": role,
                    "content": msg.content
                })

        # System prompt to guide response generation
        system_message = {
            "role": "system",
            "content": """You are a helpful AI assistant that generates diverse and specific responses. 
            For each response:
            1. Be specific and actionable
            2. Include relevant details and context
            3. Consider different aspects or approaches
            4. Use search queries when needed using [Search] format
            
            Available tools:
            - TavilySearchResults: Search the internet for current information"""
        }

        # Generate responses
        responses = []
        for num in range(n):
            try:
                # 修改这里的debug_log调用，确保安全访问formatted_messages
                debug_log(f"Generating candidate {num}", 
                         formatted_messages=formatted_messages if num == 0 else "subsequent message")
                result = llm.invoke([system_message, *formatted_messages])
                debug_log(f"Generated result for candidate {num}", result=result)
                if result.content.strip():
                    responses.append(result)
            except Exception as e:
                debug_log(f"Error generating individual response", error=str(e))
                continue

        debug_log("Total responses generated", count=len(responses))
        
        # Ensure we return at least one response
        if not responses:
            default_response = AIMessage(
                content="I apologize, but I need to search for more information to provide a detailed response. "
                       "[Search] " + str(messages)
            )
            responses = [default_response]

        return responses[:n]

    except Exception as e:
        debug_log(f"Error in generate_candidates: {str(e)}")
        # Return a single fallback response in case of errors
        fallback_response = AIMessage(
            content="I apologize, but I need to search for more information to provide a detailed response. "
                   "[Search] " + str(messages)
        )
        return [fallback_response]


# 修改 expansion_chain
expansion_chain = prompt_template | generate_candidates
#res = expansion_chain.invoke({"input": "Write a research report on lithium pollution."})
#print(res)


#Candidate generation node
from collections import defaultdict

def select(root: Node) -> dict:
    """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""

    if not root.children:
        return root

    node = root
    while node.children:
        max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
        node = max_child

    return node


def expand(state: TreeState, config: RunnableConfig) -> dict:
    """Starting from the "best" node in the tree, generate N candidates for the next step."""
    root = state["root"]
    best_candidate: Node = select(root)
    messages = best_candidate.get_trajectory()
    # Generate N candidates from the single child candidate
    new_candidates = expansion_chain.invoke(
        {"input": state["input"], "messages": messages}, config
    )
    parsed = parser.batch(new_candidates)
    flattened = [
        (i, tool_call)
        for i, tool_calls in enumerate(parsed)
        for tool_call in tool_calls
    ]
    tool_responses = tool_executor.batch(
        [
            ToolInvocation(tool=tool_call["type"], tool_input=tool_call["args"])
            for _, tool_call in flattened
        ]
    )
    collected_responses = defaultdict(list)
    for (i, tool_call), resp in zip(flattened, tool_responses):
        collected_responses[i].append(
            ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
        )
    output_messages = []
    for i, candidate in enumerate(new_candidates):
        output_messages.append([candidate] + collected_responses[i])

    # Reflect on each candidate
    # For tasks with external validation, you'd add that here.
    reflections = reflection_chain.batch(
        [{"input": state["input"], "candidate": msges} for msges in output_messages],
        config,
    )
    # Grow tree
    child_nodes = [
        Node(cand, parent=best_candidate, reflection=reflection)
        for cand, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    # We have already extended the tree directly, so we just return the state
    return state



#Create Graph
from typing import Literal

from langgraph.graph import END, StateGraph, START


def should_loop(state: TreeState) -> Literal["expand", "__end__"]:
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 5:
        return END
    return "expand"


builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.add_edge(START, "start")


builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
)

graph = builder.compile()

#Invoke
question = "请搜索获取主流Agent框架，并获取其名称、开源协议、github地址、框架功能、主要特点"
last_step = None
for step in graph.stream({"input": question}):
    print("STEP-------->"+str(step))
    last_step = step
    step_name, step_state = next(iter(step.items()))
    print(step_name)
    print("rolled out: ", step_state["root"].height)
    print("---")

solution_node = last_step["expand"]["root"].get_best_solution()
best_trajectory = solution_node.get_trajectory(include_reflections=False)
debug_log(best_trajectory[-1].content)
