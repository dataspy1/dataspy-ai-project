import json
import re
from pathlib import Path
from typing import Any, Dict, List

from app.services.llm.llm_client import LLMClient


class ExplanationService:
    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "analysis_explainer.txt"
        return prompt_path.read_text(encoding="utf-8")

    def explain_analysis(
        self,
        analysis_context: Dict[str, Any],
        audience: str = "management",
        tone: str = "executive",
    ) -> Dict[str, Any]:
        compact_context = self._normalize_context(analysis_context)

        user_prompt = f"""
Audience: {audience}
Tone: {tone}

Important Rules:
- Use ONLY the provided analysis context.
- Do NOT invent trends, risks, performance changes, root causes, or recommendations.
- If evidence is partial, explicitly use cautious wording such as:
  - "based on available insights"
  - "the current analysis suggests"
  - "this is not explicitly available in the current analysis"
- If forecast reliability is weak, reflect that uncertainty.
- Use decisions directly for recommended next steps when available.
- Link insights, forecast, and decisions where the evidence supports that linkage.
- Keep the language business-friendly, clear, concise, and executive-ready.
- Return ONLY valid JSON.

Required JSON format:
{{
  "executive_summary": "string",
  "business_explanation": "string",
  "recommended_next_steps": ["string"],
  "risk_summary": ["string"]
}}

Analysis Context:
{json.dumps(compact_context, indent=2, default=str)}

Generate the JSON response now.
""".strip()

        raw_output = self.llm_client.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )

        parsed = self._safe_parse_json(raw_output)

        parsed["executive_summary"] = self._soften_overclaims(
            str(parsed.get("executive_summary", "")).strip()
        )
        parsed["business_explanation"] = self._soften_overclaims(
            str(parsed.get("business_explanation", "")).strip()
        )

        recommended = parsed.get("recommended_next_steps", [])
        if isinstance(recommended, str):
            recommended = [recommended]
        elif not isinstance(recommended, list):
            recommended = []

        risks = parsed.get("risk_summary", [])
        if isinstance(risks, str):
            risks = [risks]
        elif not isinstance(risks, list):
            risks = []

        parsed["recommended_next_steps"] = [
            self._soften_overclaims(str(step)).strip()
            for step in recommended[:5]
            if str(step).strip()
        ]

        parsed["risk_summary"] = [
            self._soften_overclaims(str(risk)).strip()
            for risk in risks[:5]
            if str(risk).strip()
        ]

        parsed["model_used"] = self.llm_client.model
        return parsed

    def _normalize_context(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx = ctx or {}

        dashboard_summary = ctx.get("dashboard_summary") or {}
        file_summary = ctx.get("file_summary") or {}
        profile = ctx.get("profile") or {}
        dataset_profile = profile.get("dataset_profile") or ctx.get("dataset_profile") or {}
        capabilities = ctx.get("capabilities") or {}
        insights = ctx.get("insights") or {}
        forecast = ctx.get("forecast") or {}
        decisions = ctx.get("decisions") or {}
        missing_data_summary = ctx.get("missing_data_summary") or {}
        narratives = ctx.get("narratives") or {}
        schema_suggestions = ctx.get("schema_suggestions") or {}

        sales_insights = insights.get("sales") or {}
        inventory_insights = insights.get("inventory") or {}
        logistics_insights = insights.get("logistics") or {}

        # Prefer compact top lists from either revenue or quantity
        top_products = (
            sales_insights.get("top_products_by_revenue")
            or sales_insights.get("top_products_by_quantity")
            or sales_insights.get("top_products")
            or []
        )
        top_regions = (
            sales_insights.get("top_regions_by_revenue")
            or sales_insights.get("top_regions_by_quantity")
            or sales_insights.get("top_regions")
            or []
        )

        top_decisions = decisions.get("top_decisions") if isinstance(decisions, dict) else decisions
        if not isinstance(top_decisions, list):
            top_decisions = []

        compact_decisions: List[Dict[str, Any]] = []
        for d in top_decisions[:3]:
            if not isinstance(d, dict):
                continue

            confidence_obj = d.get("confidence") if isinstance(d.get("confidence"), dict) else {}
            impact_obj = d.get("impact_estimate") if isinstance(d.get("impact_estimate"), dict) else {}

            compact_decisions.append({
                "title": d.get("title"),
                "category": d.get("category") or d.get("decision_type"),
                "priority": d.get("priority"),
                "risk_level": d.get("risk_level"),
                "recommendation": d.get("recommendation") or d.get("action"),
                "rationale": d.get("rationale"),
                "trigger_source": d.get("trigger_source"),
                "confidence_label": confidence_obj.get("label"),
                "confidence_score": confidence_obj.get("score"),
                "impact_label": impact_obj.get("label"),
                "impact_score": impact_obj.get("score"),
            })

        compact_capabilities = {}
        for cap_name, cap_data in capabilities.items():
            if not isinstance(cap_data, dict):
                continue
            compact_capabilities[cap_name] = {
                "status": cap_data.get("status", "unknown"),
                "enabled": bool(cap_data.get("enabled", False)) or str(cap_data.get("status", "")).lower() == "enabled",
                "confidence": cap_data.get("confidence"),
                "enable_reason": cap_data.get("enable_reason"),
            }

        forecast_summary = forecast.get("forecast_summary", {})
        if not isinstance(forecast_summary, dict):
            forecast_summary = {}

        null_counts = dataset_profile.get("null_counts") or {}
        total_missing = sum(null_counts.values()) if isinstance(null_counts, dict) else None

        compact_context = {
            "message": ctx.get("message"),
            "dashboard_summary": {
                "executive_summary": dashboard_summary.get("executive_summary"),
                "headline_summary": dashboard_summary.get("headline_summary"),
                "recommended_next_step": dashboard_summary.get("recommended_next_step"),
                "top_risk": dashboard_summary.get("top_risk"),
                "decision_count": dashboard_summary.get("decision_count"),
                "high_priority_count": dashboard_summary.get("high_priority_count"),
                "forecast_mode": dashboard_summary.get("forecast_mode"),
                "forecast_reliability_label": dashboard_summary.get("forecast_reliability_label"),
                "decision_quality_label": dashboard_summary.get("decision_quality_label"),
                "data_quality_label": dashboard_summary.get("data_quality_label"),
            },
            "dataset_profile": {
                "dataset_name": file_summary.get("saved_filename"),
                "total_rows": dataset_profile.get("total_rows") or file_summary.get("rows"),
                "total_columns": dataset_profile.get("total_columns") or len(file_summary.get("columns", [])),
                "duplicate_rows": dataset_profile.get("duplicate_rows"),
                "missing_values": total_missing,
            },
            "schema_suggestions": {
                "date": schema_suggestions.get("date"),
                "product": schema_suggestions.get("product"),
                "region": schema_suggestions.get("region"),
                "revenue": schema_suggestions.get("revenue"),
                "quantity": schema_suggestions.get("quantity"),
                "stock": schema_suggestions.get("stock"),
                "reorder_level": schema_suggestions.get("reorder_level"),
            },
            "missing_data_summary": {
                "has_missing_data": missing_data_summary.get("has_missing_data"),
                "total_missing_values": missing_data_summary.get("total_missing_values"),
                "columns_with_missing_count": len(missing_data_summary.get("columns_with_missing", []) or []),
                "warning_message": missing_data_summary.get("warning_message"),
            },
            "capabilities": compact_capabilities,
            "insights": {
                "analysis_metric": sales_insights.get("analysis_metric"),
                "top_products": top_products[:3] if isinstance(top_products, list) else [],
                "top_regions": top_regions[:3] if isinstance(top_regions, list) else [],
                "sales_summary": sales_insights.get("sales_summary"),
                "primary_trend_summary": sales_insights.get("primary_trend_summary"),
                "revenue_trend_summary": sales_insights.get("revenue_trend_summary"),
                "quantity_trend_summary": sales_insights.get("quantity_trend_summary"),
                "product_concentration_summary": sales_insights.get("product_concentration_summary"),
                "region_concentration_summary": sales_insights.get("region_concentration_summary"),
                "weakest_region_summary": sales_insights.get("weakest_region_summary"),
                "inventory_summary": inventory_insights.get("inventory_summary"),
                "inventory_pressure_level": inventory_insights.get("inventory_pressure_level"),
                "low_stock_items_count": len(inventory_insights.get("low_stock_items", []) or []),
                "logistics_summary": logistics_insights.get("logistics_summary"),
                "delivery_delay_summary": logistics_insights.get("delivery_delay_summary"),
                "supplier_delay_summary": logistics_insights.get("supplier_delay_summary"),
                "logistics_delay_risk": logistics_insights.get("delay_risk") or logistics_insights.get("logistics_delay_risk"),
            },
            "narratives": {
                "dataset_summary": narratives.get("dataset_summary"),
                "capability_summary": narratives.get("capability_summary"),
                "sales_summary": narratives.get("sales_summary"),
                "inventory_summary": narratives.get("inventory_summary"),
                "logistics_summary": narratives.get("logistics_summary"),
                "executive_summary": narratives.get("executive_summary"),
            },
            "forecast": {
                "forecast_mode": forecast.get("forecast_mode"),
                "target_role": forecast.get("target_role") or forecast.get("forecast_target_role"),
                "target_column": forecast.get("target_column") or forecast.get("forecast_target_column"),
                "date_column": forecast.get("date_column") or forecast.get("forecast_date_column"),
                "forecast_horizon": forecast.get("forecast_horizon"),
                "model_used": forecast.get("model_used") or forecast.get("model_type"),
                "model_selection_reason": forecast.get("model_selection_reason"),
                "reliability_label": forecast.get("reliability_label"),
                "reliability_score": forecast.get("reliability_score"),
                "decision_usability_flag": forecast.get("decision_usability_flag"),
                "recommended_next_step": forecast.get("recommended_next_step"),
                "latest_actual_value": forecast.get("latest_actual_value"),
                "average_forecast_value": forecast.get("average_forecast_value") or forecast.get("average_forecast"),
                "summary": forecast.get("summary"),
                "trend_direction": forecast.get("trend_direction") or forecast_summary.get("trend_direction"),
                "growth_percent": forecast.get("growth_percent") or forecast_summary.get("projected_change_percent"),
                "validation_metrics": forecast.get("validation_metrics", {}),
                "warnings": (forecast.get("warnings", []) or [])[:3],
            },
            "decisions": {
                "summary": decisions.get("summary") if isinstance(decisions, dict) else None,
                "decision_quality": decisions.get("decision_quality", {}) if isinstance(decisions, dict) else {},
                "data_quality": decisions.get("data_quality", {}) if isinstance(decisions, dict) else {},
                "top_decisions": compact_decisions,
            },
        }

        return compact_context

    def _safe_parse_json(self, raw_output: str) -> Dict[str, Any]:
        cleaned = (raw_output or "").strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
            return self._normalize_parsed_response(parsed)
        except Exception:
            pass

        try:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                return self._normalize_parsed_response(parsed)
        except Exception:
            pass

        return {
            "executive_summary": "Could not parse model output cleanly.",
            "business_explanation": raw_output,
            "recommended_next_steps": [],
            "risk_summary": [],
        }

    def _normalize_parsed_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return {
                "executive_summary": "",
                "business_explanation": "",
                "recommended_next_steps": [],
                "risk_summary": [],
            }

        return {
            "executive_summary": parsed.get("executive_summary", ""),
            "business_explanation": parsed.get("business_explanation", ""),
            "recommended_next_steps": parsed.get("recommended_next_steps", []),
            "risk_summary": parsed.get("risk_summary", []),
        }

    def _soften_overclaims(self, text: str) -> str:
        replacements = {
            "Revenue is growing": "Based on available insights, revenue appears to be growing",
            "Revenue is declining": "Based on available insights, revenue may be under pressure",
            "This proves": "This suggests",
            "This confirms": "This indicates",
            "will definitely": "may",
            "will certainly": "is likely to",
            "guarantees": "supports",
            "clearly proves": "suggests",
        }

        cleaned = text or ""
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned.strip()