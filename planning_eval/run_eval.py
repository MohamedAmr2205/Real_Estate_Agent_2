import os
import json
from dotenv import load_dotenv
from planning.algorithms.self_refine import SelfRefine
from planning.algorithms.reflexion import Reflexion

# تحميل المفاتيح من ملف .env
load_dotenv(override=True)

# حالات الاختبار المجهزة للتقييم
TEST_CASES = [
    {
        "id": "TC01",
        "task": "Draft a counter-offer response to a cash offer 8% below asking price.",
        "context": {
            "asking_price": "$500,000",
            "offer_price": "$460,000",
            "seller_deadline": "3 weeks"
        },
        "rubric": [
            "Must keep a professional and polite tone.",
            "Must NOT reveal the seller's minimum floor price.",
            "Must propose a middle-ground counter offer price."
        ]
    },
    {
        "id": "TC02",
        "task": "Respond to a client demanding an immediate 15% discount due to minor cosmetic defects.",
        "context": {
            "asking_price": "$1,200,000",
            "offer_price": "$1,020,000",
            "defects": "Minor paint touch-ups required"
        },
        "rubric": [
            "Acknowledge the cosmetic defects professionally.",
            "Decline the 15% reduction while remaining open to negotiation.",
            "Offer a reasonable repair credit or minor price adjustment instead."
        ]
    }
]

def run_evaluation():
    print("🧪 Running Planning Evaluation Benchmark...\n")
    
    self_refine_agent = SelfRefine()
    reflexion_agent = Reflexion()
    
    results_summary = []
    
    for case in TEST_CASES:
        print(f"==========================================")
        print(f" Running Test Case: {case['id']}")
        print(f" Task: {case['task']}")
        print(f"==========================================\n")
        
        # 1. تشغيل Self-Refine
        print("▶ Running Self-Refine Algorithm...")
        sr_res = self_refine_agent.run(case['task'], case['context'], case['rubric'])
        
        # 2. تشغيل Reflexion
        print("▶ Running Reflexion Algorithm...")
        re_res = reflexion_agent.run(case['task'], case['context'], case['rubric'])
        
        results_summary.append({
            "test_id": case['id'],
            "task": case['task'],
            "self_refine": sr_res,
            "reflexion": re_res
        })
        print("✓ Test case done.\n")

    # كتابة التقرير في ملف planning_eval/results.md
    report_path = os.path.join("planning_eval", "results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Planning Algorithms Evaluation Report\n\n")
        f.write("This report compares the performance of **Self-Refine** and **Reflexion** planning algorithms on Cornerstone Realty task requests.\n\n")
        
        for item in results_summary:
            f.write(f"## Test Case: {item['test_id']}\n")
            f.write(f"**Task:** {item['task']}\n\n")
            
            f.write("### 1. Self-Refine Output\n")
            f.write(f"```text\n{item['self_refine'].get('final_output', 'N/A')}\n```\n\n")
            
            f.write("### 2. Reflexion Output\n")
            f.write(f"```text\n{item['reflexion'].get('final_output', 'N/A')}\n```\n\n")
            f.write(f"**Reflexion Iterations:** {item['reflexion'].get('total_iterations', 1)}\n")
            f.write(f"**Lessons Learned Memory:** {len(item['reflexion'].get('reflections_memory', []))} entry(ies)\n\n")
            f.write("---\n\n")
            
    print(f"🎉 Evaluation Complete! Report successfully generated at: {report_path}")

if __name__ == "__main__":
    run_evaluation()