from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import httpx
import traceback
import re
import streamlit as st

# deepagents imports
from deepagents.agent import DeepAgent
from deepagents.runtime import DeepAgentRuntime


# ============================================================
# LangGraph State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, "chat history"]


# ============================================================
# Azure / Model CONFIG
# ============================================================

AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "sk-wwXoGekBcGsk52Y3lWZv1g")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "https://genailab.tcs.in")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-06-01")

LLM_MODEL = os.environ.get("AZURE_LLM_MODEL", "azure/genailab-maas-gpt-4o")
EMBED_MODEL = os.environ.get("AZURE_EMBED_MODEL", "azure/genailab-maas-text-embedding-3-large")

POLICY_DOCS_FOLDER = "data/policy_docs"

# ============================================================
# Deepcopy-Safe Client (for embeddings / httpx)
# ============================================================

class DeepcopySafeClient(httpx.Client):
    def __deepcopy__(self, memo):
        return DeepcopySafeClient(verify=False, timeout=self.timeout)

http_client = DeepcopySafeClient(verify=False, timeout=60.0)

# ============================================================
# RAG: load policy docs + build vectordb
# ============================================================

def load_policy_docs():
    docs = []
    if not os.path.exists(POLICY_DOCS_FOLDER):
        return docs
    for fname in os.listdir(POLICY_DOCS_FOLDER):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(POLICY_DOCS_FOLDER, fname))
            docs.extend(loader.load())
    return docs

def build_vectordb(docs):
    if not docs:
        return None

    embedder = AzureOpenAIEmbeddings(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        model=EMBED_MODEL,
        http_client=http_client,
    )
    return Chroma.from_documents(docs, embedding=embedder)

docs = load_policy_docs()
vectordb = build_vectordb(docs)

def rag_search(query: str) -> str:
    """Tool: search policy documents."""
    if vectordb is None:
        return "No policy documents available."
    results = vectordb.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in results])

# ============================================================
# LLM: ChatOpenAI configured for Azure
# ============================================================

llm = ChatOpenAI(
    model=LLM_MODEL,                           # Azure deployment name
    base_url=f"{AZURE_ENDPOINT}/openai/v1/",   # Azure OpenAI v1 endpoint
    api_key=AZURE_API_KEY,
    timeout=60.0,
)

# ============================================================
# DeepAgents mirroring your ConversableAgents
# ============================================================

def create_rag_agent() -> DeepAgent:
    return DeepAgent(
        name="rag_agent",
        llm=llm,
        system_prompt="Answer loan policy queries using retrieved context.",
        tools=[rag_search],
    )

def create_intake_agent() -> DeepAgent:
    return DeepAgent(
        name="intake_agent",
        llm=llm,
        system_prompt="Extract all applicant details from the input.",
        tools=[],
    )

def create_docs_agent() -> DeepAgent:
    return DeepAgent(
        name="docs_agent",
        llm=llm,
        system_prompt="Interpret bank statements, IDs, and income documents.",
        tools=[],
    )

def create_credit_agent() -> DeepAgent:
    return DeepAgent(
        name="credit_agent",
        llm=llm,
        system_prompt="Calculate DTI, FOIR, risk score, and repayment ability.",
        tools=[],
    )

def create_decision_agent() -> DeepAgent:
    return DeepAgent(
        name="decision_agent",
        llm=llm,
        system_prompt="Make the final decision: APPROVE, REJECT, or MORE INFO.",
        tools=[],
    )

# Instantiate agents & runtimes
rag_agent = create_rag_agent()
intake_agent = create_intake_agent()
docs_agent = create_docs_agent()
credit_agent = create_credit_agent()
decision_agent = create_decision_agent()

rag_runtime = DeepAgentRuntime(agent=rag_agent)
intake_runtime = DeepAgentRuntime(agent=intake_agent)
docs_runtime = DeepAgentRuntime(agent=docs_agent)
credit_runtime = DeepAgentRuntime(agent=credit_agent)
decision_runtime = DeepAgentRuntime(agent=decision_agent)

# ============================================================
# LangGraph nodes – one per DeepAgent
# ============================================================

def rag_node(state: AgentState):
    """Optional: node to answer policy questions directly."""
    response: AIMessage = rag_runtime.run(messages=state["messages"])
    return {"messages": state["messages"] + [response]}

def intake_node(state: AgentState):
    response: AIMessage = intake_runtime.run(messages=state["messages"])
    return {"messages": state["messages"] + [response]}

def docs_node(state: AgentState):
    response: AIMessage = docs_runtime.run(messages=state["messages"])
    return {"messages": state["messages"] + [response]}

def credit_node(state: AgentState):
    response: AIMessage = credit_runtime.run(messages=state["messages"])
    return {"messages": state["messages"] + [response]}

def decision_node(state: AgentState):
    response: AIMessage = decision_runtime.run(messages=state["messages"])
    return {"messages": state["messages"] + [response]}

# ============================================================
# Build the LangGraph – intake → docs → credit → decision
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("intake", intake_node)
graph.add_node("docs", docs_node)
graph.add_node("credit", credit_node)
graph.add_node("decision", decision_node)

# rag_node is registered but not in the main pipeline;
# you can invoke a separate graph starting at "rag" if you want.
graph.add_node("rag", rag_node)

graph.set_entry_point("intake")
graph.add_edge("intake", "docs")
graph.add_edge("docs", "credit")
graph.add_edge("credit", "decision")
graph.add_edge("decision", END)

# Compile app once and reuse
app = graph.compile()

# ============================================================
# LoanApprovalWorkflow wrapper around LangGraph
# ============================================================

class LoanApprovalWorkflow:
    def __init__(self):
        self.app = app

    def invoke(self, text: str) -> str:
        try:
            initial_state: AgentState = {
                "messages": [HumanMessage(content=text)]
            }
            final_state = self.app.invoke(initial_state)
            last_msg: AIMessage = final_state["messages"][-1]
            return last_msg.content
        except Exception:
            return "Workflow error:\n" + traceback.format_exc()

# (Optional) Example initial_state for debugging (not used by Streamlit)
# initial_state: AgentState = {
#     "messages": [HumanMessage(content="Explain LangGraph and deepagents in 3 short points.")]
# }
# debug_state = app.invoke(initial_state)
# print(debug_state["messages"][-1].content)

# ============================================================
# Output Formatting (your code)
# ============================================================

def parse_metrics(response):
    # Basic regular expressions to extract metrics; improve as needed!
    metrics = {
        "DTI Ratio": None,
        "FOIR": None,
        "Disposable Income": None,
        "Decision": None
    }
    dti_match = re.search(r"DTI.*?(\d+)%", response)
    foir_match = re.search(r"FOIR.*?(\d+)%", response)
    di_match = re.search(r"(inadequate|insufficient) disposable income", response, re.I)
    if dti_match:
        metrics["DTI Ratio"] = f"{dti_match.group(1)}%"
    if foir_match:
        metrics["FOIR"] = f"{foir_match.group(1)}%"
    if di_match:
        metrics["Disposable Income"] = "Insufficient"
    metrics["Decision"] = (
        "Rejected" if "reject" in response.lower()
        else "Approved" if "approve" in response.lower()
        else "Requires Review"
    )
    return metrics

def show_loan_output(response):
    # Extract metrics and decision
    metrics = parse_metrics(response)
    # Decision header
    if metrics["Decision"] == "Approved":
        st.markdown("<h2 style='color:#27ae60'>✅ Loan Approved</h2>", unsafe_allow_html=True)
    elif metrics["Decision"] == "Rejected":
        st.markdown("<h2 style='color:#c0392b'>❌ Loan Rejected</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>⚠️ Requires Review</h2>", unsafe_allow_html=True)

    st.markdown("<h3>Applicant Summary</h3>", unsafe_allow_html=True)
    st.write(response)

    st.markdown("<h3>Key Risk Metrics</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <table>
            <tr>
                <th>Metric</th><th>Value</th><th>Risk Level</th>
            </tr>
            <tr>
                <td>DTI Ratio</td><td>{metrics.get("DTI Ratio", "Not found")}</td>
                <td>{'High 🚨' if metrics.get('DTI Ratio') and int(metrics['DTI Ratio'].replace('%','')) > 40 else 'Acceptable'}</td>
            </tr>
            <tr>
                <td>FOIR</td><td>{metrics.get("FOIR", "Not found")}</td>
                <td>{'Very High 🚨' if metrics.get('FOIR') and int(metrics['FOIR'].replace('%','')) > 50 else 'Acceptable'}</td>
            </tr>
            <tr>
                <td>Disposable Income</td><td>{metrics.get("Disposable Income", "OK")}</td>
                <td>{'Insufficient 🚫' if metrics.get('Disposable Income') == 'Insufficient' else 'OK'}</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
    st.markdown(
        "- Try a smaller loan amount\n"
        "- Consider extending the tenure\n"
        "- Seek alternative financial support\n"
        "- Improve income or reduce liabilities"
    )

# ============================================================
# Streamlit UI integrated with LangGraph + DeepAgents workflow
# ============================================================

def main():
    st.set_page_config(page_title="Loan Approval AI Workflow", page_icon="💰")
    st.title("Loan Approval AI Workflow")

    # Mode select
    run_test = st.toggle("Run LangGraph + DeepAgents Explanation Test Mode", value=False)

    if run_test:
        st.info("Test Mode active: This will bypass the form and run a pipeline explanation request.")
        if st.button("Run Test Now"):
            test_text = "Explain LangGraph and deepagents in 3 short points."
            st.markdown("### 🔄 Running Test Request...")
            wf = LoanApprovalWorkflow()
            response = wf.invoke(test_text)
            st.subheader("Test Response")
            st.write(response)
        return  # Do NOT show loan form while this mode is on

    # Loan Approval Mode (the original UI)
    st.markdown("Enter applicant details below and get AI-powered loan approval analysis.")

    with st.form(key="applicant_form"):
        name = st.text_input("Applicant Name", "Rahul Sharma")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, value=85000)
        existing_emi = st.number_input("Existing EMI (₹)", min_value=0, value=8500)
        requested_loan = st.number_input("Requested Loan Amount (₹)", min_value=0, value=500000)
        tenure_months = st.number_input("Loan Tenure (months)", min_value=1, max_value=360, value=36)
        purpose = st.text_input("Purpose of Loan", "Medical Emergency")

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        input_text = f"""
Applicant Name: {name}
Age: {age}
Monthly Salary: {monthly_salary}
Existing EMI: {existing_emi}
Requested Loan: {requested_loan}
Tenure: {tenure_months} months
Purpose: {purpose}
"""
        st.markdown("### Processing your request...")
        wf = LoanApprovalWorkflow()
        response = wf.invoke(input_text)
        st.markdown("## Loan Approval AI Response")
        show_loan_output(response)