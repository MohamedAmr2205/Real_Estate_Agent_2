import os
import json
from pathlib import Path
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


class Reflexion:
    """
    Reflexion algorithm for agentic task execution.
    Combines evaluation with explicit verbal self-reflection stored in memory.

    GROUNDING NOTE (lab requirement):
      evaluate() uses the REAL grounded Environment from environment.py
      (Person 2) as its PRIMARY check — not the model's own opinion.
      The LLM rubric check runs AFTER the grounded check, and only if
      the grounded check passes. This is the grounded-vs-ungrounded
      contrast the lab explicitly tests.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_initial_attempt(
        self, task_description: str, context: Dict[str, Any]
    ) -> str:
        prompt = f"""
        You are an expert real estate strategy advisor at Cornerstone Realty.
        Task: {task_description}
        Context: {context}
        Draft a clear and professional response or counter-offer message.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def evaluate(
        self,
        draft: str,
        rubric: List[str],
        environment=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Two-stage evaluation:
          1. GROUNDED check — real DB/RAG checks via environment.py (Person 2).
             If this fails, we skip the LLM check and return immediately.
             This is the failure the ungrounded version would have missed.
          2. LLM rubric check — model evaluates professional tone,
             confidentiality, etc. (things the DB can't check).
        """
        grounded_details = []

        # ------------------------------------------------------------------
        # Stage 1: GROUNDED check (Person 2's real environment)
        # ------------------------------------------------------------------
        if environment is not None and context is not None:
            env_feedback = environment.evaluate(context)
            if not env_feedback.success:
                failure_details = "\n".join(env_feedback.details) if env_feedback.details else "Unknown grounded failure."
                return {
                    "passed": False,
                    "grounded": True,
                    "feedback": (
                        "GROUNDED CHECK FAILED — real DB/RAG validation:\n"
                        + failure_details
                        + "\n\n[LLM rubric check skipped — grounded check must pass first]"
                    ),
                }
            grounded_details.append(
                f"Grounded check passed (score={env_feedback.score:.2f})"
            )

        # ------------------------------------------------------------------
        # Stage 2: LLM rubric check (professional tone, confidentiality, etc.)
        # ------------------------------------------------------------------
        rubric_str = "\n".join([f"- {r}" for r in rubric])
        grounded_note = (
            "\n".join(grounded_details) if grounded_details
            else "No grounded environment provided — LLM-only evaluation."
        )

        prompt = f"""
        Evaluate the following draft strictly against these business rules:
        {rubric_str}

        Draft to evaluate:
        \"\"\"{draft}\"\"\"

        Note: The following grounded checks have already passed:
        {grounded_note}

        Provide your response in this exact format:
        STATUS: [PASSED or FAILED]
        FEEDBACK: [Detailed explanation of what failed or why it passed]
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        passed = "STATUS: PASSED" in content or content.startswith("PASSED")
        return {
            "passed": passed,
            "grounded": environment is not None,
            "feedback": content,
        }

    def reflect(
        self, draft: str, feedback: str, task_description: str
    ) -> str:
        prompt = f"""
        Task: {task_description}
        Draft Attempt:
        \"\"\"{draft}\"\"\"

        Evaluation Feedback:
        \"\"\"{feedback}\"\"\"

        Write a concise verbal self-reflection explaining what went wrong
        and specific actions to fix it in the next attempt.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def refine_with_reflection(
        self,
        draft: str,
        memory: List[str],
        task_description: str,
        context: Dict[str, Any],
    ) -> str:
        reflections_str = "\n".join(
            [f"- Lesson {i+1}: {r}" for i, r in enumerate(memory)]
        )
        prompt = f"""
        Task: {task_description}
        Context: {context}
        Previous Draft:
        \"\"\"{draft}\"\"\"

        Past Self-Reflections (Lessons Learned):
        {reflections_str}

        Rewrite the response incorporating all insights from past reflections.
        Output ONLY the updated draft text.
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
        max_steps: int = 3,
    ) -> Dict[str, Any]:
        """
        Args:
            task_description: What to draft/decide.
            context:          Offer/property context dict.
            rubric:           List of business-rule strings for LLM check.
            environment:      Grounded Environment from environment.py (Person 2).
                              Pass None to run ungrounded (for comparison table).
            max_steps:        Max retry trials (episodic buffer is capped here).
        """
        draft = self.generate_initial_attempt(task_description, context)
        # Episodic memory buffer — capped at max_steps reflections
        memory: List[str] = []
        history = [{"step": 0, "draft": draft}]

        for step in range(1, max_steps + 1):
            eval_res = self.evaluate(
                draft, rubric,
                environment=environment,
                context=context,
            )

            if eval_res["passed"]:
                history[-1]["status"] = "PASSED"
                break

            reflection = self.reflect(
                draft, eval_res["feedback"], task_description
            )
            # Cap episodic buffer at max_steps entries
            if len(memory) < max_steps:
                memory.append(reflection)

            draft = self.refine_with_reflection(
                draft, memory, task_description, context
            )
            history.append({
                "step": step,
                "feedback": eval_res["feedback"],
                "grounded_check_used": eval_res.get("grounded", False),
                "reflection": reflection,
                "refined_draft": draft,
            })

        # Save artifact trace
        _save_trace(task_description, history, memory)

        return {
            "final_output": draft,
            "refinement_history": history,
            "reflections_memory": memory,
            "total_iterations": len(history),
            "grounded": environment is not None,
        }


def _save_trace(
    task: str,
    history: List[Dict],
    memory: List[str],
) -> None:
    from datetime import datetime
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "algorithm": "reflexion",
        "task": task,
        "timestamp": datetime.utcnow().isoformat(),
        "total_iterations": len(history),
        "reflections_memory": memory,
        "history": history,
    }
    path = artifacts_dir / f"reflexion_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
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

    reflexion_agent = Reflexion()
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

    print("Running Reflexion — GROUNDED...")
    result = reflexion_agent.run(
        sample_task, sample_context, sample_rubric,
        environment=env,
    )
    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])
    print("\n--- REFLECTION MEMORY ---")
    for idx, ref in enumerate(result["reflections_memory"], 1):
        print(f"Lesson {idx}: {ref}")

    print("\n\nRunning Reflexion — UNGROUNDED (for comparison)...")
    result_ungrounded = reflexion_agent.run(
        sample_task, sample_context, sample_rubric,
        environment=None,
    )
    print("\n--- UNGROUNDED FINAL OUTPUT ---")
    print(result_ungrounded["final_output"])