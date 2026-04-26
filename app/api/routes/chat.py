from fastapi import APIRouter, HTTPException
from app.schemas.explain import ExplainAnalysisRequest, ExplainAnalysisResponse
from app.services.llm.explanation_service import ExplanationService
from app.schemas.chat import ChatQueryRequest, ChatQueryResponse
from app.services.llm.chat_service import generate_chat_answer

router = APIRouter(prefix="/api/chat", tags=["Chat / LLM"])

explanation_service = ExplanationService()


@router.post("/explain-analysis", response_model=ExplainAnalysisResponse)
def explain_analysis(payload: ExplainAnalysisRequest):
    try:
        if not payload.analysis_context:
            raise ValueError("analysis_context is required.")

        result = explanation_service.explain_analysis(
            analysis_context=payload.analysis_context,
            audience=payload.audience or "management",
            tone=payload.tone or "executive",
        )
        return ExplainAnalysisResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print("LLM explanation error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"LLM explanation failed: {str(e)}"
        )


@router.post("/query", response_model=ChatQueryResponse)
def query_chat(payload: ChatQueryRequest):
    try:
        if not payload.question or not payload.question.strip():
            raise ValueError("question is required.")

        if not payload.analysis_context:
            raise ValueError("analysis_context is required.")

        result = generate_chat_answer(
            question=payload.question,
            analysis_context=payload.analysis_context
        )
        return ChatQueryResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print("Chat query error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Chat query failed: {str(e)}"
        )