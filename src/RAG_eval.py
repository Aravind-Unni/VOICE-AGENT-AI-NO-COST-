import os
import json
import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, evaluate
from rag_engine import initialize_support_rag_pipeline

load_dotenv()

# Initialize LangSmith Client
client = Client()

def setup_evaluation_dataset(dataset_name: str):
    if client.has_dataset(dataset_name=dataset_name):
        print(f"📊 Dataset '{dataset_name}' already exists.")
        return client.read_dataset(dataset_name=dataset_name)

    print(f"🛠️ Creating new dataset: '{dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for the Zendesk Support Voice Agent."
    )
    
    examples = [
        {
            "question": "What are the three types of user roles available in Zendesk?",
            "answer": "The three user roles are End-user, Agent, and Administrator."
        },
        {
            "question": "What is the maximum file size limit for a ticket attachment on a Plus or Enterprise Zendesk account?",
            "answer": "The maximum size of an attachment is 20 MB for Plus and Enterprise accounts."
        },
        {
            "question": "How long does the \"recently viewed tickets\" list remain visible if no action is taken on any of your tickets?",
            "answer": "The recently viewed tickets list disappears after 72 hours if there is no action taken on any of your tickets."
        },
        {
            "question": "How long are unrecovered suspended tickets kept before they are permanently deleted?",
            "answer": "All unrecovered suspended tickets are automatically deleted after 14 days."
        },
        {
            "question": "What is the \"agent collision\" feature and what is its purpose?",
            "answer": "Agent collision is an alert displayed in a ticket when you and another agent are simultaneously viewing the same ticket. It helps prevent more than one agent from working on the ticket at the same time."
        },
        {
            "question": "What does setting a ticket's status to \"Pending\" indicate?",
            "answer": "The \"Pending\" status indicates that the assigned agent has a follow-up question for the requester and the ticket is on hold until that information is received."
        },
        {
            "question": "What happens to linked \"Incident\" tickets when the main \"Problem\" ticket is solved?",
            "answer": "When you solve the problem ticket, the status of all the linked incident tickets is automatically set to solved too."
        },
        {
            "question": "Who has the permission to change a user's role in Zendesk?",
            "answer": "Only administrators can change a user's role."
        },
        {
            "question": "Can agents delete closed tickets in bulk using a view?",
            "answer": "No, you cannot delete closed tickets in bulk in a view. However, an administrator can use the API to delete closed tickets in bulk."
        },
        {
            "question": "What is the maximum number of tickets you can bulk update at one time?",
            "answer": "The maximum number of tickets you can update at one time is 99 tickets."
        },
        {
            "question": "How long can a screencast recording be if the administrator has upgraded Screenr for forums screencasting?",
            "answer": "It can be up to 15 minutes long if the admin has upgraded Screenr for forums screencasting."
        },
        {
            "question": "In Markdown formatting, which character is used to create both bold and italic text?",
            "answer": "The asterisk character (*) is used for both italic and bold emphasis."
        },
        {
            "question": "When merging a duplicate user account into a receiving account, what happens to the merging user's tags?",
            "answer": "The merging user's tags are lost during the merge."
        },
        {
            "question": "In Zendesk Voice, how much time does an agent have to answer an incoming phone call before it is placed back into the queue?",
            "answer": "If the agent does not answer the phone within 30 seconds, the call is placed back into the queue to wait for the next available agent."
        },
        {
            "question": "What specific keyword syntax should you use to search for tickets generated from inbound phone calls?",
            "answer": "You should use the keyword via:phone_call_inbound."
        },
        {
            "question": "What is the maximum number of end-users you can add as CCs on a ticket?",
            "answer": "You can add up to 24 end-users as CCs on a ticket."
        },
        {
            "question": "How long does a chat request wait before timing out if it is not accepted by an agent?",
            "answer": "Chat requests time out if they are not accepted within one minute."
        },
        {
            "question": "How many days after an inbound email is received does access to the \"original email\" (including full HTML and source header) expire?",
            "answer": "Access to the original email expires 30 days after the email is received in your Zendesk."
        },
        {
            "question": "Can you recover tickets after they have been deleted using the bulk delete feature?",
            "answer": "No, once deleted, tickets cannot be recovered; they are permanently deleted."
        },
        {
            "question": "What is the maximum length allowed for voicemail messages in Zendesk Voice?",
            "answer": "Voicemail messages can be up to 3 minutes long."
        }
    ]
    
    # FIXED: Iterate through dictionaries correctly
    for example in examples:
        client.create_example(
            inputs={"question": example["question"]},
            outputs={"expected_answer": example["answer"]},
            dataset_id=dataset.id
        )
        
    return dataset

def main():
    print("🚀 Starting LangSmith Evaluation Process...")
    
    try:
        # Load the newly updated Support RAG
        rag_app = initialize_support_rag_pipeline()
    except FileNotFoundError as e:
        print(f"\n⚠️ Critical Error: {e}")
        return

    eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    dataset_name = "Zendesk_Support_Test_Suite_v1"
    setup_evaluation_dataset(dataset_name)

    print("\n⚖️ Running Evaluation... (This will process all 20 questions sequentially)")
    
    # 1. Target Task Function (Generates the answer)
    def predict_rag_answer(inputs: dict) -> dict:
        question = inputs["question"]
        state = {
            "original_question": question,
            "chat_history": [],
            "session_id": "eval_session" 
        }
        result = rag_app.invoke(state)
        return {"actual_answer": result["generation"]}

    # 2. Custom Evaluator (Grades the answer)
    def custom_accuracy_evaluator(run, example) -> dict:
        question = example.inputs["question"]
        expected = example.outputs["expected_answer"]
        actual = run.outputs["actual_answer"]
        
        eval_prompt = PromptTemplate.from_template(
            """You are an expert Zendesk QA Auditor. 
            Your task is to evaluate a Support AI Agent's response by comparing it to a verified Ground Truth Reference Answer.

            ---
            USER QUESTION: 
            {question}

            GROUND TRUTH REFERENCE ANSWER: 
            {reference}

            SUPPORT AI AGENT ANSWER: 
            {prediction}
            ---

            EVALUATION CRITERIA:
            1. Factual Accuracy: Does the Agent's answer contain the same core information and business rules as the Ground Truth?
            2. Formatting: The Agent is a Voice AI, so it may paraphrase or speak conversationally. Do NOT penalize for missing exact terminology if the meaning is identical.
            3. Hallucinations: If the Agent introduces features, rules, or steps not present in the Ground Truth, it must fail.
            4. Contradictions: If the Agent's answer contradicts the core logic of the Ground Truth, it must fail.

            Return your evaluation as a valid JSON object with EXACTLY two keys:
            - "reasoning": A brief 1-2 sentence explanation of why the answer passed or failed.
            - "score": integer 1 (if correct) or integer 0 (if incorrect).

            JSON Output:"""
        )
        
        chain = eval_prompt | eval_llm | StrOutputParser()
        
        # FIXED: Variable names match prompt template {reference} and {prediction}
        result_str = chain.invoke({
            "question": question, 
            "reference": expected, 
            "prediction": actual
        })
        
        # FIXED: Proper JSON parsing to extract "score" and "reasoning"
        try:
            # Strip markdown formatting if the LLM wraps it in ```json ... ```
            clean_json_str = re.sub(r'```json|```', '', result_str).strip()
            parsed_result = json.loads(clean_json_str)
            score = float(parsed_result.get("score", 0))
            reasoning = parsed_result.get("reasoning", "No reasoning provided.")
        except Exception as e:
            print(f"⚠️ Failed to parse LLM evaluation JSON: {e}")
            score = 0.0
            reasoning = f"Evaluation parse failed. Raw LLM output: {result_str}"
            
        status_icon = "✅ Correct (1.0)" if score == 1.0 else "❌ Incorrect (0.0)"
        print(f"↳ Evaluated: '{question[:35]}...' -> {status_icon}")
        
        # We also pass the reasoning text to LangSmith as 'comment' so you can read it in the dashboard
        return {"key": "accuracy", "score": score, "comment": reasoning}

    # 3. Run the LangSmith Experiment
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[custom_accuracy_evaluator],
        experiment_prefix="zendesk-voice-rag-eval",
        metadata={"version": "1.0"},
        max_concurrency=1  # <--- Prevents hitting Groq API Rate Limits
    )
        
    print("\n✅ Evaluation Complete! Check the 'Datasets & Testing' tab in your LangSmith dashboard.")

if __name__ == "__main__":
    main()