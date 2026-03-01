"""
Text-based Question Answering System
Using Pre-trained BERT models
No training required - works out of the box
"""

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Step 1: Load Question Answering Pipeline
# ==========================================
print("Loading Question Answering Model...")
print("Model: distilbert-base-cased-distilled-squad\n")

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

print("Model loaded successfully!\n")
print("="*60)

# ==========================================
# Step 2: Define Context (Paragraph)
# ==========================================
context = """
Artificial intelligence is widely used in healthcare.
It helps doctors detect diseases early and improve diagnosis accuracy.
AI is also used in finance for fraud detection and risk management.
In education, AI powers personalized learning systems and automated grading.
Machine learning, a subset of AI, enables computers to learn from data.
Deep learning uses neural networks to solve complex problems.
"""

print("CONTEXT (Paragraph):")
print(context)
print("="*60)

# ==========================================
# Step 3: Ask Questions
# ==========================================
questions = [
    "Where is artificial intelligence widely used?",
    "What does AI help doctors do?",
    "What is machine learning?",
    "What does deep learning use?",
    "What is AI used for in finance?"
]

print("\nQUESTION-ANSWER PAIRS:\n")

for i, question in enumerate(questions, 1):
    # Get answer
    result = qa_pipeline(
        question=question,
        context=context
    )
    
    print(f"{i}. Question: {question}")
    print(f"   Answer: {result['answer']}")
    print(f"   Confidence: {result['score']:.4f}")
    print(f"   Start Position: {result['start']}")
    print(f"   End Position: {result['end']}")
    print("-"*60)

# ==========================================
# Step 4: Interactive Mode
# ==========================================
def interactive_qa():
    """
    Interactive Question Answering
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nDefault Context:")
    print(context)
    print("\nType 'quit' to exit")
    print("Type 'change' to use new context\n")
    
    current_context = context
    
    while True:
        user_input = input("\nEnter your question: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting... Thank you!")
            break
        
        if user_input.lower() == 'change':
            print("\nEnter new context (paragraph):")
            current_context = input().strip()
            print("Context updated!")
            continue
        
        if not user_input:
            print("Please enter a valid question.")
            continue
        
        try:
            result = qa_pipeline(
                question=user_input,
                context=current_context
            )
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['score']:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")

# ==========================================
# Step 5: Advanced Features
# ==========================================
def qa_with_details(question, context):
    """
    Question Answering with detailed output
    """
    result = qa_pipeline(question=question, context=context)
    
    answer = result['answer']
    score = result['score']
    start = result['start']
    end = result['end']
    
    # Extract surrounding context
    context_window = 50
    start_context = max(0, start - context_window)
    end_context = min(len(context), end + context_window)
    
    surrounding_text = context[start_context:end_context]
    
    print("\n" + "="*60)
    print("DETAILED ANSWER REPORT")
    print("="*60)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print(f"Confidence Score: {score:.4f}")
    print(f"Confidence Percentage: {score*100:.2f}%")
    print(f"\nAnswer Position: characters {start} to {end}")
    print(f"\nSurrounding Context:")
    print(f"...{surrounding_text}...")
    print("="*60)
    
    return result

# Test advanced function
print("\n" + "="*60)
print("ADVANCED QA WITH DETAILS")
print("="*60)

qa_with_details(
    "What is machine learning?",
    context
)

# ==========================================
# Step 6: Multiple Models Comparison
# ==========================================
def compare_models(question, context):
    """
    Compare answers from different models
    """
    models = [
        "distilbert-base-cased-distilled-squad",
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    ]
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\nQuestion: {question}\n")
    
    for model_name in models:
        try:
            print(f"Loading {model_name}...")
            qa = pipeline("question-answering", model=model_name)
            result = qa(question=question, context=context)
            
            print(f"\nModel: {model_name}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['score']:.4f}")
            print("-"*60)
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            print("-"*60)

# Uncomment to test model comparison (takes time)
# compare_models("Where is AI used?", context)

# ==========================================
# Step 7: Batch Question Answering
# ==========================================
def batch_qa(questions_list, context):
    """
    Process multiple questions at once
    """
    results = []
    
    print("\n" + "="*60)
    print("BATCH QUESTION ANSWERING")
    print("="*60)
    
    for i, q in enumerate(questions_list, 1):
        result = qa_pipeline(question=q, context=context)
        results.append({
            'question': q,
            'answer': result['answer'],
            'score': result['score']
        })
        
        print(f"\n{i}. Q: {q}")
        print(f"   A: {result['answer']} (Confidence: {result['score']:.2f})")
    
    return results

# Test batch processing
batch_questions = [
    "What is AI used for in healthcare?",
    "What enables computers to learn?",
    "What solves complex problems?"
]

batch_results = batch_qa(batch_questions, context)

# ==========================================
# Step 8: Save Results to File
# ==========================================
def save_results_to_file(results, filename="qa_results.txt"):
    """
    Save QA results to text file
    """
    with open(filename, 'w') as f:
        f.write("QUESTION ANSWERING RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"{i}. Question: {r['question']}\n")
            f.write(f"   Answer: {r['answer']}\n")
            f.write(f"   Confidence: {r['score']:.4f}\n")
            f.write("-"*60 + "\n")
    
    print(f"\nResults saved to {filename}")

save_results_to_file(batch_results)

# ==========================================
# Step 9: Error Handling
# ==========================================
def safe_qa(question, context):
    """
    Question Answering with error handling
    """
    try:
        if not context or len(context.strip()) == 0:
            return {"error": "Context cannot be empty"}
        
        if not question or len(question.strip()) == 0:
            return {"error": "Question cannot be empty"}
        
        result = qa_pipeline(question=question, context=context)
        
        if result['score'] < 0.1:
            return {
                "answer": result['answer'],
                "score": result['score'],
                "warning": "Low confidence - answer may not be reliable"
            }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

# Test error handling
print("\n" + "="*60)
print("ERROR HANDLING TEST")
print("="*60)

test_result = safe_qa("What is quantum computing?", context)
print(f"\nQuestion: What is quantum computing?")
if 'error' in test_result:
    print(f"Error: {test_result['error']}")
elif 'warning' in test_result:
    print(f"Answer: {test_result['answer']}")
    print(f"Warning: {test_result['warning']}")
else:
    print(f"Answer: {test_result['answer']}")

# ==========================================
# Run Interactive Mode (Optional)
# ==========================================
print("\n" + "="*60)
print("Would you like to try interactive mode? (yes/no)")
choice = input().strip().lower()

if choice == 'yes':
    interactive_qa()
else:
    print("\nProgram finished. Thank you!")