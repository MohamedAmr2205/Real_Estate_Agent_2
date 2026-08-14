import os
from typing import Dict, Any, List
from groq import Groq


class SelfRefine:
    """
    Self-Refine algorithm for cheap sub-task revision.
    Critiques and refines drafts against a clear rubric.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_initial_draft(self, task_description: str, context: Dict[str, Any]) -> str:
        prompt = f"""
        You are an expert real estate strategy advisor at Cornerstone Realty.
        Task: {task_description}
        Context: {context}
        Draft a clear and professional counter-offer message or response.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def critique(self, draft: str, rubric: List[str]) -> str:
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
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def refine(self, draft: str, critique: str, task_description: str) -> str:
        prompt = f"""
        Task: {task_description}
        Original Draft:
        \"\"\"{draft}\"\"\"

        Critique to address:
        \"\"\"{critique}\"\"\"

        Rewrite the draft addressing all critique points. Output ONLY the refined draft text.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def run(self, task_description: str, context: Dict[str, Any], rubric: List[str], max_steps: int = 2) -> Dict[str, Any]:
        draft = self.generate_initial_draft(task_description, context)
        history = [{"step": 0, "draft": draft}]

        for step in range(1, max_steps + 1):
            critique_res = self.critique(draft, rubric)
            if critique_res == "PASSED":
                break
            draft = self.refine(draft, critique_res, task_description)
            history.append({"step": step, "critique": critique_res, "refined_draft": draft})

        return {
            "final_output": draft,
            "refinement_history": history,
            "total_iterations": len(history)
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Load GROQ_API_KEY from .env file

    # Initialize algorithm instance
    refiner = SelfRefine()

    # Define sample task and context
    sample_task = "Draft a counter-offer response to a cash offer 8% below asking price."
    sample_context = {
        "asking_price": "$500,000",
        "offer_price": "$460,000",
        "seller_deadline": "3 weeks"
    }

    # Define rubric evaluation criteria
    sample_rubric = [
        "Must keep a professional and polite tone.",
        "Must NOT reveal the seller's minimum floor price.",
        "Must propose a middle-ground counter offer price."
    ]

    print("🚀 Running Self-Refine Test...")
    result = refiner.run(sample_task, sample_context, sample_rubric)

    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])

    print("\n--- REFINEMENT HISTORY ---")
    for step in result["refinement_history"]:
        print(f"Step {step.get('step')}:")
        if "critique" in step:
            print(f" Critique: {step['critique']}")