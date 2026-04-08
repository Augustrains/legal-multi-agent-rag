from agno.agent import Agent
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.websearch import WebSearchTools

from .logging_config import get_logger
from .kb_service import create_local_kb_search_tool

logger = get_logger("legal_app")

#负责根据不同的分析类型构建不同的prompt，并调用相应的agent或team进行分析
ANALYSIS_CONFIGS = {
    "Contract Review": {
        "query": """
        Review the uploaded contract from a document-analysis perspective.
        Extract the key clauses, obligations, rights, deadlines, payment terms, termination conditions, liabilities, and dispute-related provisions.
        Identify ambiguities, missing protections, inconsistent wording, and drafting weaknesses.
        Prioritize the clauses, obligations, and drafting issues most central to the contract's operative terms.
        Do not expand into unrelated legal background or broad strategic recommendations unless clearly necessary to complete the task.
        Focus on what the document explicitly says and what it fails to address.
        """,
        "agents": ["Clause Analyst"],
        "description": "Document-level review of clauses, obligations, and drafting issues"
    },
    "Legal Research": {
        "query": """
        Identify the main legal issues raised by the uploaded document and provide the relevant legal principles, precedents, regulatory concerns, or compliance background.
        Distinguish clearly between what is stated in the uploaded document and what comes from legal research or external authority.
        Provide stable legal background, key definitions, and jurisdiction-relevant support where helpful.
        Focus on legal interpretation and authority support rather than contract redrafting.
        """,
        "agents": ["Legal Researcher", "Local Legal KB Specialist"],
        "description": "Legal doctrine, precedent, and regulatory research related to the document"
    },
    "Risk Assessment": {
        "query": """
        First identify the document-level issues, problematic clauses, missing protections, or ambiguous drafting in the uploaded document.
        Then assess the legal and practical risks associated with those issues.
        Use relevant legal background where helpful to support the analysis of identified risks and mitigation considerations.
        Focus only on risks that arise from clauses, omissions, or ambiguities identifiable in the uploaded document.
        Rank the major risks by priority and explain why they matter.
        Finally, provide practical risk-mitigation recommendations.
        """,
        "agents": ["Clause Analyst", "Local Legal KB Specialist", "Risk Strategist"],
        "description": "Issue spotting, risk prioritization, and practical mitigation planning"
    },
    "Compliance Check": {
        "query": """
        Review the uploaded document for compliance-related clauses, omissions, and regulatory risk points.
        Identify the relevant legal or regulatory concerns raised by the document.
        Clarify the applicable compliance framework, regulatory background, and jurisdiction-relevant obligations where helpful.
        Then explain the practical compliance risks and recommend remediation or follow-up actions.
        Separate document findings, legal/compliance analysis, and recommended next steps.
        """,
        "agents": ["Clause Analyst", "Legal Researcher", "Local Legal KB Specialist", "Risk Strategist"],
        "description": "Cross-functional review of document clauses, regulatory concerns, and remediation actions"
    },
    "Custom Query": {
        "query": None,
        "agents": ["Clause Analyst", "Legal Researcher", "Local Legal KB Specialist", "Risk Strategist"],
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
        self._local_kb_specialist = None
        self._local_agent_without_kb = None
        self._single_legal_agent = None
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

    #纯本地使用agent
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
    
    #用于参与多智能体的本地agent
    def build_local_kb_specialist(self):
        if self._local_kb_specialist is None:
            logger.info("[AgentService] Creating Local Legal KB Specialist")
            self._local_kb_specialist = Agent(
                name="Local Legal KB Specialist",
                role="Local legal knowledge-base support specialist for team-based legal analysis",
                model=self.llm,
                knowledge=self.local_knowledge_base,
                search_knowledge=True,
                knowledge_retriever=self.local_retriever,
                tools=[],
                instructions=[
                    "Your primary responsibility is to support the team with legal background from the local legal knowledge base.",
                    "Use the local legal knowledge base to retrieve relevant legal concepts, compliance frameworks, regulatory background, and jurisdiction-relevant legal support.",
                    "You may use the user's question and the uploaded document only to understand context and identify what legal background should be retrieved.",
                    "Base your substantive claims and evidence only on the local legal knowledge base.",
                    "Clearly distinguish between what comes from the uploaded document and what comes from the local legal knowledge base.",
                    "Do NOT use external web information.",
                    "Do NOT perform general web-based legal research; that is the Legal Researcher's responsibility.",
                    "Do NOT focus on clause-by-clause extraction from the uploaded document; that is the Clause Analyst's responsibility.",
                    "Do NOT prioritize risks or give final action plans unless explicitly asked to provide limited KB-based support; that is primarily the Risk Strategist's responsibility.",
                    "Support every important claim with relevant knowledge-base evidence whenever possible.",
                    "Do NOT display technical metadata such as file_name, chunk, content_hash, legal_topic, or jurisdiction.",
                    "Do NOT fabricate citations, quotations, or source support.",
                    "If the answer is not clearly supported by the local legal knowledge base, say so explicitly.",
                    "Do not reveal internal reasoning or chain-of-thought."
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Local Legal KB Specialist")

        return self._local_kb_specialist
   
   #用于比较本地agent
    def build_local_agent_without_kb(self):
        if self._local_agent_without_kb is None:
            logger.info("[AgentService] Creating Local Legal Agent Without KB")
            self._local_agent_without_kb = Agent(
                name="Local Legal Agent Without KB",
                role="Standalone legal question-answering specialist without retrieval access",
                model=self.llm,
                knowledge=None,
                search_knowledge=False,
                tools=[],
                instructions=[
                    "Answer the user's legal question without using the local legal knowledge base or any uploaded user document.",
                    "Do not use external web information.",
                    "Rely only on the model's internal knowledge and be explicit when you are uncertain.",
                    "Do NOT fabricate quotations, citations, or source excerpts.",
                    "Do NOT display technical metadata or pretend to have retrieved supporting text.",
                    "If the answer is uncertain or jurisdiction-specific, say so clearly.",
                    "Do not reveal internal reasoning or chain-of-thought.",
                    "Output format:",
                    "(1) Answer",
                    "(2) What I can say confidently",
                    "(3) Caveats",
                    "(4) Follow-up questions"
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Local Legal Agent Without KB")

        return self._local_agent_without_kb
    
    #用于和多智能体对比的单智能体
    def build_single_legal_agent(self):
        if self._single_legal_agent is None:
            logger.info("[AgentService] Creating Single Legal Agent")
            local_kb_tool = create_local_kb_search_tool(
                local_knowledge_base=self.local_knowledge_base,
                local_retriever=self.local_retriever,
            )
            self._single_legal_agent = Agent(
                name="Single Legal Agent",
                role="Generalist legal analyst handling document review, legal research, and risk analysis in one agent",
                model=self.llm,
                knowledge=self.knowledge_base,
                search_knowledge=True,
                tools=[WebSearchTools(enable_news=False), local_kb_tool],
                instructions=[
                    "You are a single generalist legal analyst covering the responsibilities that are otherwise split across a clause analyst, legal researcher, and risk strategist.",
                    "When reviewing an uploaded document, identify what the document explicitly says, what legal or regulatory issues it raises, and what practical risks or next steps follow.",
                    "Extract key clauses, obligations, rights, deadlines, payment terms, termination conditions, liabilities, and dispute-related provisions when they are relevant.",
                    "Identify ambiguities, missing protections, inconsistent wording, undefined terms, drafting weaknesses, and clause imbalances.",
                    "Use the local legal knowledge base tool when stable legal background, definitions, compliance framework, or jurisdiction-relevant support would help answer the task.",
                    "When legal research is needed, distinguish clearly between what comes from the uploaded document and what comes from general legal principles or external authority.",
                    "Clearly distinguish between uploaded-document evidence, local legal knowledge-base support, and external web research.",
                    "When risk analysis is needed, prioritize the major issues and explain why they matter in practice.",
                    "Provide practical next steps only when they are relevant to the user's requested task.",
                    "Use uploaded-document evidence whenever possible and support important legal claims with sources whenever possible.",
                    "Do not reveal internal reasoning or chain-of-thought.",
                    "Structure the answer clearly, separating document findings, legal analysis, and recommendations whenever the task calls for those distinctions."
                ],
                debug_mode=False,
                markdown=True
            )
        else:
            logger.info("[AgentService] Reusing cached Single Legal Agent")

        return self._single_legal_agent

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
                    self.build_local_kb_specialist(),
                    self.build_risk_strategist(),
                ],
                knowledge=self.knowledge_base,
                search_knowledge=True,
                instructions=[
                    "You are the team lead coordinating four distinct specialists.",
                    "Clause Analyst is responsible for extracting and analyzing what the uploaded document explicitly says.",
                    "Legal Researcher is responsible for legal principles, precedents, regulations, and external legal context.",
                    "Local Legal KB Specialist is responsible for retrieving stable legal background and compliance support from the local legal knowledge base.",
                    "Risk Strategist is responsible for prioritizing risks and recommending practical next actions.",
                    "Local Legal KB Specialist complements Legal Researcher by using the local legal knowledge base only, while Legal Researcher handles external legal and web research.",
                    "Answer the requested task directly before adding any broader context.",
                    "Do not produce a comprehensive contract review unless the task explicitly requires one.",
                    "Do not let one specialist take over another specialist's primary responsibility unless necessary.",
                    "Use only the specialists relevant to the task focus and keep irrelevant specialists lightweight.",
                    "Synthesize the specialists' findings into one coherent final answer.",
                    "Do not mention delegation steps, internal coordination, or tool usage in the final answer.",
                    "Structure the final answer clearly and separate document findings, local knowledge-base support, legal research, and recommendations when relevant.",
                    "Ensure that important conclusions are supported either by the uploaded document or by identified legal authority.",
                    "Do not state clause numbers, dates, obligations, or legal conclusions unless they are clearly supported by the uploaded document or explicitly identified authority.",
                    "If the uploaded document does not clearly support a point, say that the document does not provide enough support.",
                    "Keep external legal background brief and include it only when it materially helps complete the task.",
                    "Do not add generic negotiation advice, strategic commentary, or broad legal background unless the task explicitly asks for it.",
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
                - local legal knowledge-base support,
                - legal research / legal authority,
                - or risk prioritization / practical strategy.
                2. Let the most relevant specialist lead the answer.
                3. Other specialists should contribute only where their expertise is necessary.
                4. Use Local Legal KB Specialist when stable legal background, compliance framework, definitions, or jurisdiction-relevant support from the local legal knowledge base would help answer the task.
                5. Local Legal KB Specialist should provide support only from the local legal knowledge base.
                6. Legal Researcher should handle external legal and web research.
                7. Provide specific references from the uploaded document whenever possible.
                8. Clearly separate document findings, local knowledge-base support, legal research, and recommendations when relevant.
                9. Answer the user's request directly before adding any broader context.
                10. Do not expand into unrelated parts of the contract unless necessary to answer the request.
                11. If the uploaded document does not clearly support a point, explicitly state that the document does not provide enough support.
                12. Avoid generic legal background, broad contract summaries, and unnecessary recommendations unless they directly help answer the request.

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
                3. Use Local Legal KB Specialist when stable legal background, compliance framework, definitions, or jurisdiction-relevant support from the local legal knowledge base would help answer the task.
                4. Local Legal KB Specialist should provide support only from the local legal knowledge base.
                5. Legal Researcher should handle external legal and web research.
                6. Keep non-primary specialists lightweight unless their expertise is necessary.
                7. Provide specific references from the uploaded document whenever possible.
                8. Separate document findings, local knowledge-base support, legal analysis, and recommendations where appropriate.
                9. Stay tightly scoped to the Primary Task.
                10. Do not expand into unrelated parts of the contract unless necessary to complete the task.
                11. Use uploaded-document findings as the default source of truth.
                12. Add local knowledge-base support or external legal context only when it materially helps the task.
                13. If document support is incomplete, explicitly note the limitation instead of filling gaps with assumptions.
                14. Avoid generic legal background, broad contract summaries, and unnecessary recommendations unless the task specifically asks for them.
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
    




    #进行对比实验测试
    def build_single_prompt(self, analysis_type, user_query=None) -> str:
        logger.info(
            f"[AgentService] build_single_prompt analysis_type={analysis_type} "
            f"user_query_preview={(user_query or '')[:200]}"
        )
        config = self.get_analysis_config(analysis_type)

        if analysis_type == "Local Query":
            return user_query or ""

        if analysis_type == "Custom Query":
            return f"""
    Using the uploaded document as reference, answer the user's request below.

    User Request:
    {user_query}

    Instructions:
    1. Answer the user's request directly and stay focused on the specific question asked.
    2. Use the uploaded document as the primary source when the answer depends on document content.
    3. Use the local legal knowledge base when stable legal background, definitions, or compliance framework support would help.
    4. Use external legal/web research only when it adds necessary legal authority or context.
    5. Clearly distinguish between document findings, local knowledge-base support, and external legal research when relevant.
    6. Do not provide unnecessary background if the user's question is narrow.
    """

    def run_single_agent(self, analysis_type, user_query=None):
        logger.info(
            f"[AgentService] run_single_agent started analysis_type={analysis_type} "
            f"user_query_preview={(user_query or '')[:200]}"
        )

        combined_query = self.build_single_prompt(analysis_type, user_query)

        if analysis_type == "Local Query":
            agent = self.build_local_agent_without_kb()
            result = agent.run(combined_query)
            logger.info("[AgentService] Single-agent Local Query finished")
            return result

        agent = self.build_single_legal_agent()
        result = agent.run(combined_query)
        logger.info("[AgentService] Single-agent analysis finished")
        return result
