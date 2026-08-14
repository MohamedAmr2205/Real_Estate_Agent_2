import os
from typing import Dict, Any, List
from groq import Groq


class Reflexion:
    """
    Reflexion algorithm for agentic task execution.
    Combines evaluation with explicit verbal self-reflection stored in memory.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_initial_attempt(self, task_description: str, context: Dict[str, Any]) -> str:
        prompt = f"""
        You are an expert real estate strategy advisor at Cornerstone Realty.
        Task: {task_description}
        Context: {context}
        Draft a clear and professional response or counter-offer message.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def evaluate(self, draft: str, rubric: List[str]) -> Dict[str, Any]:
        rubric_str = "\n".join([f"- {r}" for r in rubric])
        prompt = f"""
        Evaluate the following draft strictly against these business rules:
        {rubric_str}

        Draft to evaluate:
        \"\"\"{draft}\"\"\"

        Provide your response in this exact format:
        STATUS: [PASSED or FAILED]
        FEEDBACK: [Detailed explanation of what failed or why it passed]
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        passed = "STATUS: PASSED" in content or content.startswith("PASSED")
        return {"passed": passed, "feedback": content}

    def reflect(self, draft: str, feedback: str, task_description: str) -> str:
        prompt = f"""
        Task: {task_description}
        Draft Attempt:
        \"\"\"{draft}\"\"\"

        Evaluation Feedback:
        \"\"\"{feedback}\"\"\"

        Write a concise verbal self-reflection explaining what went wrong and specific actions to fix it in the next attempt.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def refine_with_reflection(self, draft: str, memory: List[str], task_description: str, context: Dict[str, Any]) -> str:
        reflections_str = "\n".join([f"- Lesson {i+1}: {r}" for i, r in enumerate(memory)])
        prompt = f"""
        Task: {task_description}
        Context: {context}
        Previous Draft:
        \"\"\"{draft}\"\"\"

        Past Self-Reflections (Lessons Learned):
        {reflections_str}

        Rewrite the response incorporating all insights from past reflections. Output ONLY the updated draft text.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def run(self, task_description: str, context: Dict[str, Any], rubric: List[str], max_steps: int = 2) -> Dict[str, Any]:
        draft = self.generate_initial_attempt(task_description, context)
        memory = []
        history = [{"step": 0, "draft": draft}]

        for step in range(1, max_steps + 1):
            eval_res = self.evaluate(draft, rubric)
            if eval_res["passed"]:
                break

            reflection = self.reflect(draft, eval_res["feedback"], task_description)
            memory.append(reflection)

            draft = self.refine_with_reflection(draft, memory, task_description, context)
            history.append({
                "step": step,
                "feedback": eval_res["feedback"],
                "reflection": reflection,
                "refined_draft": draft
            })

        return {
            "final_output": draft,
            "refinement_history": history,
            "reflections_memory": memory,
            "total_iterations": len(history)
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    reflexion_agent = Reflexion()

    sample_task = "Draft a counter-offer response to a cash offer 8% below asking price."
    sample_context = {
        "asking_price": "$500,000",
        "offer_price": "$460,000",
        "seller_deadline": "3 weeks"
    }

    sample_rubric = [
        "Must keep a professional and polite tone.",
        "Must NOT reveal the seller's minimum floor price.",
        "Must propose a middle-ground counter offer price."
    ]

    print("🚀 Running Reflexion Test...")
    result = reflexion_agent.run(sample_task, sample_context, sample_rubric)

    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])

    print("\n--- REFLECTION MEMORY ---")
    for idx, ref in enumerate(result["reflections_memory"], 1):
        print(f"Lesson {idx}: {ref}")