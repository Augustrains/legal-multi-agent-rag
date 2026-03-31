from agno.agent import Agent
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.websearch import WebSearchTools

from logging_config import get_logger

logger = get_logger("legal_app")

#负责根据不同的分析类型构建不同的prompt，并调用相应的agent或team进行分析
ANALYSIS_CONFIGS = {
    "Contract Review": {
        "query": """
        Review the uploaded contract from a document-analysis perspective.
        Extract the key clauses, obligations, rights, deadlines, payment terms, termination conditions, liabilities, and dispute-related provisions.
        Identify ambiguities, missing protections, inconsistent wording, and drafting weaknesses.
        Focus on what the document explicitly says and what it fails to address.
        """,
        "agents": ["Clause Analyst"],
        "description": "Document-level review of clauses, obligations, and drafting issues"
    },
    "Legal Research": {
        "query": """
        Identify the main legal issues raised by the uploaded document and provide the relevant legal principles, precedents, regulatory concerns, or compliance background.
        Distinguish clearly between what is stated in the uploaded document and what comes from legal research or external authority.
        Focus on legal interpretation and authority support rather than contract redrafting.
        """,
        "agents": ["Legal Researcher"],
        "description": "Legal doctrine, precedent, and regulatory research related to the document"
    },
    "Risk Assessment": {
        "query": """
        First identify the document-level issues, problematic clauses, missing protections, or ambiguous drafting in the uploaded document.
        Then assess the legal and practical risks associated with those issues.
        Rank the major risks by priority and explain why they matter.
        Finally, provide practical risk-mitigation recommendations.
        """,
        "agents": ["Clause Analyst", "Risk Strategist"],
        "description": "Issue spotting, risk prioritization, and practical mitigation planning"
    },
    "Compliance Check": {
        "query": """
        Review the uploaded document for compliance-related clauses, omissions, and regulatory risk points.
        Identify the relevant legal or regulatory concerns raised by the document.
        Then explain the practical compliance risks and recommend remediation or follow-up actions.
        Separate document findings, legal/compliance analysis, and recommended next steps.
        """,
        "agents": ["Clause Analyst", "Legal Researcher", "Risk Strategist"],
        "description": "Cross-functional review of document clauses, regulatory concerns, and remediation actions"
    },
    "Custom Query": {
        "query": None,
        "agents": ["Clause Analyst", "Legal Researcher", "Risk Strategist"],
        "description": "Custom multi-agent legal analysis based on the user’s question"
    },
    "Local Query": {
        "query": None,
        "agents": ["Local Legal KB Agent"],
        "description": "Question answering over the local legal knowledge base only"
    }
}


class AgentService:
    def __init__(self, llm, knowledge_base=None, local_knowledge_base=None, local_retriever=None):
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.local_knowledge_base = local_knowledge_base
        self.local_retriever = local_retriever
        self._local_agent = None
        self._clause_analyst = None
        self._legal_researcher = None
        self._risk_strategist = None
        self._legal_team = None

        logger.info(
            "[AgentService] Initialized | "
            f"has_llm={self.llm is not None} | "
            f"has_knowledge_base={self.knowledge_base is not None} | "
            f"has_local_knowledge_base={self.local_knowledge_base is not None} | "
            f"has_local_retriever={self.local_retriever is not None}"
        )

    def get_analysis_config(self, analysis_type: str):
        logger.info(f"[AgentService] get_analysis_config analysis_type={analysis_type}")
        if analysis_type not in ANALYSIS_CONFIGS:
            logger.error(f"[AgentService] Unsupported analysis type: {analysis_type}")
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

        return ANALYSIS_CONFIGS[analysis_type]

    def build_local_agent(self):
        if self._local_agent is None:
            logger.info("[AgentService] Creating Local Legal KB Agent")
            self._local_agent = Agent(
                name="Local Legal KB Agent",
                role="Local legal knowledge base question-answering and citation specialist",
                model=self.llm,
                knowledge=self.local_knowledge_base,
                search_knowledge=True,
                knowledge_retriever=self.local_retriever,
                tools=[],
                instructions=[
                    "Answer questions using ONLY the local legal knowledge base.",
                    "Always search the local legal knowledge base before answering.",
                    "Do not use uploaded user documents or any external web information.",
                    "Your task is factual legal knowledge-base question answering, not contract drafting or business strategy.",
                    "Support every important claim with relevant quoted content from the knowledge base.",
                    "Do NOT display technical metadata such as file_name, chunk, content_hash, legal_topic, or jurisdiction.",
                    "Do NOT fabricate evidence or citations.",
                    "If the answer is not clearly supported by the local knowledge base, say so explicitly.",
                    "Do not reveal internal search steps or chain-of-thought.",
                    "Output format:",
                    "(1) Answer",
                    "(2) Evidence from KB (quote only the relevant content)",
                    "(3) Caveats",
                    "(4) Follow-up questions"
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Local Legal KB Agent")

        return self._local_agent

    def build_clause_analyst(self):
        if self._clause_analyst is None:
            logger.info("[AgentService] Creating Clause Analyst")
            self._clause_analyst = Agent(
                name="Clause Analyst",
                role="Document clause extraction and contract structure specialist",
                model=self.llm,
                knowledge=self.knowledge_base,
                search_knowledge=True,
                instructions=[
                    "Your primary responsibility is document analysis only.",
                    "Analyze ONLY the uploaded document itself.",
                    "Extract key clauses, obligations, rights, deadlines, liabilities, payment terms, termination conditions, dispute-related provisions, and drafting structure from the document.",
                    "Identify ambiguities, missing protections, inconsistent wording, undefined terms, drafting weaknesses, and clause imbalances.",
                    "Always reference specific clauses, sections, or quoted passages from the uploaded document.",
                    "Do NOT perform external legal research.",
                    "Do NOT provide legal conclusions about enforceability unless explicitly asked.",
                    "Do NOT provide risk prioritization.",
                    "Do NOT provide negotiation strategy or practical next-step recommendations unless explicitly requested.",
                    "Focus strictly on clause content, clause structure, drafting quality, and missing or unclear provisions.",
                    "Do not speculate beyond the uploaded document."
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Clause Analyst")

        return self._clause_analyst

    def build_legal_researcher(self):
        if self._legal_researcher is None:
            logger.info("[AgentService] Creating Legal Researcher")
            self._legal_researcher = Agent(
                name="Legal Researcher",
                role="Legal authority, precedent, and compliance research specialist",
                model=self.llm,
                tools=[WebSearchTools()],
                knowledge=self.knowledge_base,
                search_knowledge=True,
                instructions=[
                    "Your primary responsibility is legal research.",
                    "Identify relevant legal principles, legal doctrines, precedents, regulatory issues, and compliance requirements related to the user's question and the uploaded document.",
                    "Use the uploaded document to understand context, but focus on legal authority rather than document clause extraction.",
                    "Clearly distinguish between what is explicitly stated in the uploaded document and what comes from external legal research.",
                    "When applicable, identify jurisdiction-specific uncertainty or missing legal context.",
                    "Do NOT rewrite the contract.",
                    "Do NOT focus on detailed clause-by-clause drafting unless necessary for legal interpretation.",
                    "Do NOT provide final business or negotiation strategy unless explicitly asked.",
                    "Support important legal claims with sources whenever possible."
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Legal Researcher")

        return self._legal_researcher

    def build_risk_strategist(self):
        if self._risk_strategist is None:
            logger.info("[AgentService] Creating Risk Strategist")
            self._risk_strategist = Agent(
                name="Risk Strategist",
                role="Legal risk prioritization and practical action recommendation specialist",
                model=self.llm,
                knowledge=self.knowledge_base,
                search_knowledge=True,
                instructions=[
                    "Your primary responsibility is risk prioritization and action recommendations.",
                    "Use document findings and legal research findings to identify high, medium, and low priority issues.",
                    "Explain why each major issue matters in practice.",
                    "Provide actionable next steps, such as what to revise, clarify, negotiate, escalate, verify, or monitor.",
                    "Focus on decision support, prioritization, and practical next actions.",
                    "Do NOT focus on exhaustive clause extraction.",
                    "Do NOT spend most of the answer on legal case summaries or broad legal background.",
                    "Do NOT restate the full document unless necessary.",
                    "Prioritize practical risk management over generic commentary."
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Risk Strategist")

        return self._risk_strategist

    def build_legal_team(self):
        if self._legal_team is None:
            logger.info("[AgentService] Creating Legal Team")
            self._legal_team = Team(
                name="Legal Team Lead",
                model=self.llm,
                members=[
                    self.build_legal_researcher(),
                    self.build_clause_analyst(),
                    self.build_risk_strategist(),
                ],
                knowledge=self.knowledge_base,
                search_knowledge=True,
                instructions=[
                    "You are the team lead coordinating three distinct specialists.",
                    "Clause Analyst is responsible for extracting and analyzing what the uploaded document explicitly says.",
                    "Legal Researcher is responsible for legal principles, precedents, regulations, and external legal context.",
                    "Risk Strategist is responsible for prioritizing risks and recommending practical next actions.",
                    "Do not let one specialist take over another specialist's primary responsibility unless necessary.",
                    "Use only the specialists relevant to the task focus and keep irrelevant specialists lightweight.",
                    "Synthesize the specialists' findings into one coherent final answer.",
                    "Structure the final answer clearly and separate document findings, legal research, and recommendations when relevant.",
                    "Ensure that important conclusions are supported either by the uploaded document or by identified legal authority.",
                    "Do not reveal internal reasoning or delegation process."
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Legal Team")

        return self._legal_team

    def build_prompt(self, analysis_type, user_query=None) -> str:
        logger.info(
            f"[AgentService] build_prompt analysis_type={analysis_type} "
            f"user_query_preview={(user_query or '')[:200]}"
        )
        config = self.get_analysis_config(analysis_type)

        if analysis_type == "Local Query":
            return user_query or ""

        if analysis_type == "Custom Query":
            combined_query = f"""
                Using the uploaded document as reference, answer the user's request below.

                User Request:
                {user_query}

                Task Coordination Rules:
                1. First determine whether the request is primarily about:
                - document clause analysis,
                - legal research / legal authority,
                - or risk prioritization / practical strategy.
                2. Let the most relevant specialist lead the answer.
                3. Other specialists should contribute only where their expertise is necessary.
                4. Provide specific references from the uploaded document whenever possible.
                5. Clearly separate document findings, legal research, and recommendations when relevant.

                Primary Specialists for this task:
                {', '.join(config['agents'])}
                """
            return combined_query

        combined_query = f"""
                Using the uploaded document as reference, complete the following task.

                Primary Task:
                {config['query']}

                Task Coordination Rules:
                1. The primary specialists for this task are: {', '.join(config['agents'])}.
                2. Let the most relevant specialist(s) lead the analysis.
                3. Keep non-primary specialists lightweight unless their expertise is necessary.
                4. Provide specific references from the uploaded document whenever possible.
                5. Separate document findings, legal analysis, and recommendations where appropriate.
                """
        return combined_query

    def run(self, analysis_type, user_query=None):
        logger.info(
            f"[AgentService] run started analysis_type={analysis_type} "
            f"user_query_preview={(user_query or '')[:200]}"
        )

        combined_query = self.build_prompt(analysis_type, user_query)

        if analysis_type == "Local Query":
            agent = self.build_local_agent()
            result = agent.run(combined_query)
            logger.info("[AgentService] Local Query finished")
            return result

        team = self.build_legal_team()
        result = team.run(combined_query)
        logger.info("[AgentService] Team analysis finished")
        return result
