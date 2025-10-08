# dealing with the imports 
# importing langchain library for loading, splitting, storing, preprocessing of documents and creating of tools for the AI Agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from duckduckgo_search import DDGS
# importing langgraph libraries for creating the stateful react agent
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain.agents.tool_node import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
# langsmith
# step1: load the environment variables and initialize the chat_model and the api keys.
load_dotenv()
# initialize the chat_model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)
# get the embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model= "gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)
# get tavily search api key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_search = TavilySearchResults(api_key = TAVILY_API_KEY)
langsmith_api_key =os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING=True
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_WORKSPACE_ID="<agentic_rag_app>"

# step2: load and preprocess the documents using langchain
# load  from the local database names faiss index
FAISS_INDEX_PATH = "faiss_index"
if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
    print(f"Loading existing memory index from {FAISS_INDEX_PATH}...")
    vectorStore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("memory index loaded successfully.")
else:
    print("memory index not found. Building and saving new index...")
    file_path = r"C:\\Users\\USER\\Desktop\\slot_pages.pdf" 
    loader = PyPDFLoader(file_path)
    raw_data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_data)
    vectorStore = FAISS.from_documents(chunks, embedding_model)
    vectorStore.save_local(FAISS_INDEX_PATH)
    print(f"memory index built and saved to {FAISS_INDEX_PATH}.")

# step4: creating the tools using the @tool decorator from langchain_core.tools
# the retriever tool
@tool(response_format = "content")
def retriever_tool(query:str)->str:
    """use this tool to look up the faiss index  and search the relevant information based on user query"""
    retriever_response = vectorStore.similarity_search(query, k = 4)
    formatted_response = "\n\n".join(doc.page_content for doc in retriever_response)
    return formatted_response, retriever_response

# create the websearch tools using the tavilysearch tool and the duckduckgo search tool
@tool(response_format = "content")
def tavilySearch_tool(query:str)->str:
    """use this tool to look up the web and search for real time information if need be"""
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)
    formatted_search_docs = "\n\n----\n\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in search_docs
    ])
    return formatted_search_docs
# creating the duckduckgo search tool as an augmented tool for websearch tool
@tool(response_format = "content")
def duckduckgo_search_tool(query: str)-> str:
    
    """Use this tool to search the web."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
    results_str = str(results)
    return f"{results_str}\n\n(Source: duckduckgo_search_tool)"

# register the tools, by storing the tools in a list of tools
tools = [retriever_tool, tavilySearch_tool, duckduckgo_search_tool]
# create the system prompt using clean prompt Engineering, the very heart of context Engineering
system_prompt = (
    """You are Alex, a smart and helpful AI customer care assistant specializing in smartphones sells.

CORE OPERATIONAL MODES:

1. DIRECT RETRIEVAL MODE
- When the user asks about smartphone prices or related product details, first search your internal FAISS index using the retriever tool.
- Provide accurate, up-to-date pricing and product information retrieved from your indexed database.
- Ensure responses are clear, precise, and relevant to the user's query.

2. TOOL-ASSISTED INFORMATION RETRIEVAL MODE
- If the requested information is not found or insufficient in your FAISS index, resolve to use other external search tools such as Tavily web search or DuckDuckGo search.
- Use these tools to obtain real-time, comprehensive, and authoritative information.
- Prioritize tool selection based on query complexity and specificity.

3. GENERIC TOPIC DISCUSSION MODE
- For general or conversational queries unrelated to smartphone pricing or technical support, engage using your internal knowledge base.
- Provide informative, friendly, and contextually relevant discussions on a wide range of topics.
- Demonstrate empathy and maintain a professional tone.
4. INTELLIGENT ROUTING CRITERIA
TRIGGER TOOL CALL WHEN:
- The FAISS retriever does not return sufficient or relevant information.
- The query requires current market data or news.
- The user requests detailed technical explanations or comparisons.
- Precise, verifiable, or time-sensitive information is needed.

5. RESPONSE GENERATION STRATEGY
- For retriever-based responses: Present the retrieved information clearly and concisely.
- For tool-assisted responses:
  * Indicate when external sources are used.
  * Integrate search results smoothly into your reply.
  * Maintain conversational flow and cite sources when appropriate.
- For generic discussions: Provide engaging, knowledgeable, and helpful responses.

6. INTERACTION PRINCIPLES
- Be transparent about your knowledge limits.
- Avoid speculation or providing inaccurate information.
- Adapt your communication style to the user's needs and context.
- Prioritize clarity, helpfulness, and user satisfaction.

OPERATIONAL WORKFLOW:
1. Assess the user's query to determine if it relates to smartphone pricing or support.
2. Use the FAISS retriever tool to find relevant information internally.
3. If internal data is insufficient, invoke external search tools as needed.
4. For non-product queries, engage using your internal knowledge base.
5. Generate responses that are accurate, clear, and user-friendly.

COMMUNICATION TONE:
- Professional, approachable, and empathetic.
- Clear and concise.
- Supportive and solution-oriented.  
"""
)
# create a tool calling agent that invokes the tool base on user query and gets the tool message as the response, we do this by binding the tools to the chatmodel
# exposing the tools to the chat_model which is the brain of the agent.
response_model = chat_model.bind_tools(tools)
# step5: create the state graph dict 
class AgentState(MessagesState):
    pass
# create the agent node 
def SmartPhoneAgent(state:AgentState):
    """this agent is responsible for direct response or tool call depending on the user query"""
    prompt = SystemMessage(content= system_prompt)
    ai_response = response_model.invoke([prompt] + state["messages"])
    return{"messages": [ai_response]}

# create the graph_builder 
graph_builder = StateGraph(AgentState)
# create and add the nodes: the individual tasks, python functions, or another graphs
graph_builder.add_node("SmartPhoneAgent", SmartPhoneAgent)
# adding the tool node to expose the tools to the agent.
graph_builder.add_node("tools", ToolNode(tools))
# create the edges: the connectors
graph_builder.add_edge(START, "SmartPhoneAgent")
graph_builder.add_conditional_edges("SmartPhoneAgent", tools_condition, {"tools":"tools", END:END})
graph_builder.add_edge("tools", "SmartPhoneAgent")
# create the agent short term memory that exists within the thread_id using the InMemorySaver
agents_memory = InMemorySaver()
# create the config dict
config = {"configurable":{"thread_id": "thread_A"}}
# compile the graph
compiler = graph_builder.compile(checkpointer=agents_memory)

