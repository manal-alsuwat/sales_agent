from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)

from agent import (
    build_chain,
    build_vectorstore,
    load_policies,
    full_pipeline,
    retrieve
)

from llm_judge import judge_ticket


# ---------------------------------
# Run the BeamData Agent
# ---------------------------------
def run_agent(user_input, vectorstore, chain):
    """
    Runs the BeamData agent and returns both:
    1. The final answer.
    2. The retrieved RAG context.
    """

    judge_result = judge_ticket(user_input)

    mock_row = {
        "ticket_text": user_input,
        "action": judge_result["action"],
        "risk_level": judge_result["risk_level"],
        "reason": judge_result["reason"]
    }

    # Retrieve the context used by the RAG system
    context = retrieve(vectorstore, user_input)

    # Run the full pipeline
    result = full_pipeline(
        vectorstore=vectorstore,
        chain=chain,
        row=mock_row,
        history_text=""
    )

    return {
        "answer": result["reply"],
        "context": context
    }


# ---------------------------------
# Main Evaluation Pipeline
# ---------------------------------
def main():
    print("\n" + "=" * 60)
    print("      BeamData Agent Evaluation (DeepEval)")
    print("=" * 60)

    # ----------------------------
    # Load BeamData Agent
    # ----------------------------
    print("\nLoading BeamData Agent...")

    policies = load_policies()
    vectorstore = build_vectorstore(policies)
    chain = build_chain()

    # ----------------------------
    # Load evaluation dataset
    # ----------------------------
    eval_df = pd.read_csv("data/evaluation_set.csv")

    # ----------------------------
    # Initialize metrics
    # ----------------------------
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7
    )

    total_relevancy = 0
    total_faithfulness = 0
    total_cases = 0

    # ----------------------------
    # Evaluate each test case
    # ----------------------------
    for index, row in eval_df.iterrows():

        question = row["user_input"]

        print("\n" + "=" * 60)
        print(f"Test Case #{index + 1}")
        print(f"Question: {question}")

        # Run agent
        agent_result = run_agent(
            question,
            vectorstore,
            chain
        )

        answer = agent_result["answer"]
        context = agent_result["context"]

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[context]
        )

        # Run metrics
        relevancy_metric.measure(test_case)
        faithfulness_metric.measure(test_case)

        relevancy_score = relevancy_metric.score
        faithfulness_score = faithfulness_metric.score

        total_relevancy += relevancy_score
        total_faithfulness += faithfulness_score
        total_cases += 1

        # ------------------------
        # Print result
        # ------------------------
        print("\nAgent Answer:")
        print(answer)

        print(f"\nAnswer Relevancy Score : {relevancy_score:.2f}")
        print(f"Faithfulness Score     : {faithfulness_score:.2f}")

        print("\nRelevancy Reason:")
        print(relevancy_metric.reason)

        print("\nFaithfulness Reason:")
        print(faithfulness_metric.reason)

    # ----------------------------
    # Final summary
    # ----------------------------
    average_relevancy = (
        total_relevancy / total_cases
        if total_cases > 0 else 0
    )

    average_faithfulness = (
        total_faithfulness / total_cases
        if total_cases > 0 else 0
    )

    overall_score = (
        average_relevancy + average_faithfulness
    ) / 2

    print("\n" + "=" * 60)
    print("           Evaluation Summary")
    print("=" * 60)
    print(f"Total Test Cases           : {total_cases}")
    print(f"Average Relevancy Score    : {average_relevancy:.2f}")
    print(f"Average Faithfulness Score : {average_faithfulness:.2f}")

    if overall_score >= 0.9:
        level = "Excellent"
    elif overall_score >= 0.8:
        level = "Good"
    elif overall_score >= 0.7:
        level = "Acceptable"
    else:
        level = "Needs Improvement"

    print(f"Overall Result             : {level}")
    print("=" * 60)


# ---------------------------------
# Entry Point
# ---------------------------------
if __name__ == "__main__":
    main()