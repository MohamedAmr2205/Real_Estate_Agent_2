import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from groq import Groq

def _load_env():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_env()


class SelfRefine:
    """
    Self-Refine algorithm for cheap sub-task revision.
    Critiques and refines drafts against a clear rubric.

    GROUNDING NOTE:
      critique() optionally runs a grounded Environment check FIRST
      (Person 2's environment.py) before the LLM rubric check.
      This lets the comparison table show what the grounded version
      catches that the ungrounded version misses.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_initial_draft(
        self, task_description: str, context: Dict[str, Any]
    ) -> str:
        prompt = f"""
        You are an expert real estate strategy advisor at Cornerstone Realty.
        Task: {task_description}
        Context: {context}
        Draft a clear and professional counter-offer message or response.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def critique(
        self,
        draft: str,
        rubric: List[str],
        environment=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Two-stage critique:
          1. Grounded check (environment.py) — real DB/RAG validation.
             If this fails, return the failure immediately.
          2. LLM rubric check — tone, confidentiality, etc.
        """
        # Stage 1: Grounded check
        if environment is not None and context is not None:
            env_feedback = environment.evaluate(context)
            if not env_feedback.success:
                failure = "\n".join(env_feedback.details) if env_feedback.details else "Grounded check failed."
                return f"GROUNDED FAILURE:\n{failure}"

        # Stage 2: LLM rubric check
        rubric_str = "\n".join([f"- {r}" for r in rubric])
        prompt = f"""
        Critique the following draft based strictly on these business rules:
        {rubric_str}

        Draft to critique:
        \"\"\"{draft}\"\"\"

        If the draft passes ALL rules, respond with exact word 'PASSED'.
        Otherwise, specify what failed and how to fix it.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def refine(
        self, draft: str, critique: str, task_description: str
    ) -> str:
        prompt = f"""
        Task: {task_description}
        Original Draft:
        \"\"\"{draft}\"\"\"

        Critique to address:
        \"\"\"{critique}\"\"\"

        Rewrite the draft addressing all critique points.
        Output ONLY the refined draft text.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def run(
        self,
        task_description: str,
        context: Dict[str, Any],
        rubric: List[str],
        environment=None,
        max_steps: int = 2,
    ) -> Dict[str, Any]:
        """
        Args:
            task_description: What to draft.
            context:          Offer/property context dict.
            rubric:           Business-rule strings for LLM check.
            environment:      Grounded Environment (optional — pass None
                              for ungrounded run in comparison table).
            max_steps:        Max critique-refine iterations.
        """
        draft = self.generate_initial_draft(task_description, context)
        history = [{"step": 0, "draft": draft}]

        for step in range(1, max_steps + 1):
            critique_res = self.critique(
                draft, rubric,
                environment=environment,
                context=context,
            )
            if critique_res == "PASSED":
                history[-1]["status"] = "PASSED"
                break

            draft = self.refine(draft, critique_res, task_description)
            history.append({
                "step": step,
                "critique": critique_res,
                "grounded_check_used": environment is not None,
                "refined_draft": draft,
            })

        # Save artifact trace
        _save_trace(task_description, history, environment is not None)

        return {
            "final_output": draft,
            "refinement_history": history,
            "total_iterations": len(history),
            "grounded": environment is not None,
        }


def _save_trace(task: str, history: List[Dict], grounded: bool) -> None:
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "algorithm": "self_refine",
        "task": task,
        "timestamp": datetime.utcnow().isoformat(),
        "grounded": grounded,
        "total_iterations": len(history),
        "history": history,
    }
    path = artifacts_dir / f"self_refine_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    REPO_ROOT = Path(__file__).resolve().parents[2]
    MCP_SERVER_DIR = REPO_ROOT / "mcp_server"
    for p in (str(REPO_ROOT), str(MCP_SERVER_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from environment import Environment

    refiner = SelfRefine()
    env = Environment()

    sample_task = "Draft a counter-offer response to a cash offer 8% below asking price."
    sample_context = {
        "property_id": 1,
        "offer_id": 3,
        "proposed_action": "counter",
        "proposed_price": 460000,
        "seller_deadline_weeks": 3,
        "financing_contingency_days": 30,
        "acknowledges_tier1_risk": False,
    }
    sample_rubric = [
        "Must keep a professional and polite tone.",
        "Must NOT reveal the seller's minimum floor price.",
        "Must propose a middle-ground counter offer price.",
    ]

    print("Running Self-Refine — GROUNDED...")
    result = refiner.run(
        sample_task, sample_context, sample_rubric,
        environment=env,
    )
    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])
    print("\n--- REFINEMENT HISTORY ---")
    for step in result["refinement_history"]:
        print(f"Step {step.get('step')}:")
        if "critique" in step:
            print(f"  Critique: {step['critique'][:100]}...")

    print("\n\nRunning Self-Refine — UNGROUNDED (for comparison)...")
    result_ug = refiner.run(
        sample_task, sample_context, sample_rubric,
        environment=None,
    )
    print("\n--- UNGROUNDED FINAL OUTPUT ---")
    print(result_ug["final_output"])