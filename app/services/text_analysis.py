from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, function_tool, ItemHelpers, trace
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
import asyncio
import json
from dotenv import load_dotenv
from agents.extensions.visualization import draw_graph
from datetime import datetime

load_dotenv()

# Data Models
class CriteriaScore(BaseModel):
    score: float
    explanation: str
    supporting_quotes: List[str]

class TechnicalAnalysis(BaseModel):
    overall_score: float
    knowledge_depth: CriteriaScore
    problem_solving: CriteriaScore
    best_practices: CriteriaScore
    system_design: CriteriaScore
    testing_approach: CriteriaScore
    # hiring_recommendation: str
    # key_strengths: List[str]  # Added to fix AttributeError
    # areas_for_improvement: List[str]  # Added to fix AttributeError

class CommunicationAnalysis(BaseModel):
    overall_score: float
    clarity: CriteriaScore
    articulation: CriteriaScore
    listening_skills: CriteriaScore
    professionalism: CriteriaScore

class ExperienceAnalysis(BaseModel):
    overall_score: float
    project_complexity: CriteriaScore
    impact: CriteriaScore
    growth: CriteriaScore
    technical_breadth: CriteriaScore

class BehavioralAnalysis(BaseModel):
    overall_score: float
    problem_approach: CriteriaScore
    collaboration: CriteriaScore
    learning_attitude: CriteriaScore
    initiative: CriteriaScore

class FinalAnalysis(BaseModel):
    technical: TechnicalAnalysis
    communication: CommunicationAnalysis
    experience: ExperienceAnalysis
    behavioral: BehavioralAnalysis
    overall_score: float
    key_strengths: List[str]
    # areas_for_improvement: List[str]
    # hiring_recommendation: str
    # confidence_score: float

class TranscriptGuardrailOutput(BaseModel):
    is_valid: bool
    reasoning: str

class AnalysisResult(BaseModel):
    """Complete analysis result in JSON format."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    technical: TechnicalAnalysis
    communication: CommunicationAnalysis
    experience: ExperienceAnalysis
    behavioral: BehavioralAnalysis
    
# Web Search Tool
@function_tool
async def web_search(query: str) -> str:
    """Search the web for technical information or best practices to validate candidate responses."""
    # Implementation would use a web search API
    return "Web search results for: " + query

# Specialized Agents
technical_agent = Agent(
    name="Technical Evaluator",
    handoff_description="Specialist agent for evaluating technical skills and knowledge",
    instructions="""
    You are an expert technical interviewer for frontend engineering roles. Analyze the interview transcript to evaluate:
    1. Technical knowledge depth (score 0-10)
    2. Problem-solving capabilities (score 0-10)
    3. Understanding of best practices (score 0-10)
    4. System design knowledge (score 0-10)
    5. Testing approach (score 0-10)
    
    For each area:
    - Assign a score between 0 and 10
    - Provide detailed explanation
    - Include relevant quotes from the transcript
    - Use web search to validate technical claims when needed
    
    Calculate overall_score as the average of all scores.
    Be objective and thorough in your analysis.
    
    """,
    output_type=TechnicalAnalysis,
    tools=[web_search]
)

communication_agent = Agent(
    name="Communication Evaluator",
    handoff_description="Specialist agent for evaluating communication skills",
    instructions="""
    You are an expert in evaluating communication skills. Analyze the interview transcript to evaluate:
    1. Clarity of explanations (score 0-10)
    2. Articulation of complex concepts (score 0-10)
    3. Active listening skills (score 0-10)
    4. Professional communication (score 0-10)
    
    For each area:
    - Assign a score between 0 and 10
    - Provide detailed explanation
    - Include relevant quotes from the transcript
    
    Calculate overall_score as the average of all scores.
    Focus on how effectively the candidate communicates technical concepts.
    """,
    output_type=CommunicationAnalysis,
    model="gpt-4o-mini"
)

experience_agent = Agent(
    name="Experience Evaluator",
    handoff_description="Specialist agent for evaluating professional experience",
    instructions="""
    You are an expert in evaluating professional experience. Analyze the interview transcript to evaluate:
    1. Project complexity (score 0-10)
    2. Impact and results (score 0-10)
    3. Career growth (score 0-10)
    4. Technical breadth (score 0-10)
    
    Important: Be efficient with web searches:
    - Only use web search for critical validation
    - Batch your web searches when possible
    - Prioritize transcript evidence over web validation
    - If turns are limited, focus on scoring and basic analysis first
    
    For each area:
    1. First pass: Quick score and basic analysis using transcript only
    2. Second pass: Add supporting quotes from transcript
    3. Final pass (if turns available): Validate key claims with web search
    
    Calculate overall_score as the average of all scores.
    Focus on the depth and quality of experience, not just duration.
    
    If running out of turns, ensure you at least provide:
    - Basic scores for all areas
    - Overall score
    - Key supporting evidence from transcript
    """,
    output_type=ExperienceAnalysis,
    tools=[web_search],
    model="gpt-4o-mini"
)

behavioral_agent = Agent(
    name="Behavioral Evaluator",
    handoff_description="Specialist agent for evaluating behavioral competencies",
    instructions="""
    You are an expert in evaluating behavioral competencies. Analyze the interview transcript to evaluate:
    1. Problem-solving approach (score 0-10)
    2. Collaboration skills (score 0-10)
    3. Learning attitude (score 0-10)
    4. Initiative and ownership (score 0-10)
    
    For each area:
    - Assign a score between 0 and 10
    - Provide detailed explanation
    - Include relevant quotes from the transcript
    
    Calculate overall_score as the average of all scores.
    Look for specific examples that demonstrate these competencies.
    """,
    output_type=BehavioralAnalysis,
    model="gpt-4o-mini"
)

# Guardrail Agent
guardrail_agent = Agent(
    name="Transcript Validator",
    instructions="""
    Ensure the text is a proper interview transcript.
    """,
    output_type=TranscriptGuardrailOutput,
    model="gpt-4o-mini"
)

async def transcript_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(TranscriptGuardrailOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_valid
    )

# Main Orchestrator Agent
# orchestrator_agent = Agent(
#     name="Interview Analysis Orchestrator",
#     instructions="""
#     You are the main orchestrator for interview analysis. Your role is to:
#     1. Coordinate analysis across all specialized agents
#     2. Aggregate their findings
#     3. Generate a comprehensive final analysis
#     4. Provide clear hiring recommendations
    
#     For the final analysis:
#     - Calculate overall_score as the weighted average of all specialized scores
#     - Assign confidence_score between 0 and 1
#     - List key strengths and areas for improvement
#     - Make a clear hiring recommendation
    
#     Ensure thorough analysis and maintain objectivity.
#     """,
#     handoffs=[
#         technical_agent, 
#         communication_agent, 
#         experience_agent, 
#         behavioral_agent
#     ],
#     input_guardrails=[InputGuardrail(guardrail_function=transcript_guardrail)],
#     output_type=FinalAnalysis
# )

# async def analyze_interview(transcript: str) -> FinalAnalysis:
#     """
#     Analyze an interview transcript using the multi-agent system.
    
#     Args:
#         transcript (str): The interview transcript to analyze
        
#     Returns:
#         FinalAnalysis: Comprehensive analysis of the interview
#     """
#     # result = await Runner.run(orchestrator_agent, transcript)
#     return result.final_output_as(FinalAnalysis)

async def analyze_interview_parallel(transcript: str) -> AnalysisResult:
    """
    Analyze an interview transcript by running all agents in parallel.
    
    Args:
        transcript (str): The interview transcript to analyze
        
    Returns:
        AnalysisResult: Comprehensive analysis of the interview
    """
    # First validate the transcript with explicit turn limit
    guardrail_result = await Runner.run(
        guardrail_agent, 
        transcript,
        max_turns=2  # Explicit limit for validation
    )
    guardrail_output = guardrail_result.final_output_as(TranscriptGuardrailOutput)
    
    if not guardrail_output.is_valid:
        raise ValueError(f"Invalid transcript: {guardrail_output.reasoning}")
    
    # Create a unique trace for this analysis session
    trace_id = f"interview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run all analysis agents in parallel with their own turn limits
    with trace("Parallel Interview Analysis", group_id=trace_id):
        technical_result, communication_result, experience_result, behavioral_result = await asyncio.gather(
            Runner.run(technical_agent, transcript, max_turns=5),
            Runner.run(communication_agent, transcript, max_turns=5),
            Runner.run(experience_agent, transcript, max_turns=7),
            Runner.run(behavioral_agent, transcript, max_turns=5)
        )
        
        # Extract results
        technical_analysis = technical_result.final_output_as(TechnicalAnalysis)
        communication_analysis = communication_result.final_output_as(CommunicationAnalysis)
        experience_analysis = experience_result.final_output_as(ExperienceAnalysis)
        behavioral_analysis = behavioral_result.final_output_as(BehavioralAnalysis)
        
        # Create the final result
        return AnalysisResult(
            technical=technical_analysis,
            communication=communication_analysis,
            experience=experience_analysis,
            behavioral=behavioral_analysis,
            metadata={
                "version": "1.0",
                "model": "gpt-4",
                "timestamp": datetime.now().isoformat(),
                "transcript_length": len(transcript),
                "trace_id": trace_id
            }
        )

async def main():
    from sample_transcripts import GREAT_INTERVIEW
    
    try:
        # Run parallel analysis
        analysis = await analyze_interview_parallel(GREAT_INTERVIEW["transcript"])
        
        # Convert to JSON and print
        result_json = analysis.model_dump_json(indent=2)
        print(result_json)
        
        # Save to file
        with open("analysis_result.json", "w") as f:
            f.write(result_json)
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
    # draw_graph(analysis)
