Assignment Overview: GenAI & ML Mini Project
This is a comprehensive 4-part project (100 points total) combining traditional ML, deep learning, and generative AI techniques. Here's the structure:

Part A: Machine Learning (25%)
Insurance Claim Prediction — Build a classification model to predict insurance claims.

Tasks:

Dataset Understanding — Load insurance data, display records, identify columns/types/missing values
Exploratory Data Analysis — Visualize Age, BMI, Charges, Insurance Claims
Data Preprocessing — Encode categoricals, split train/test, apply scaling, prevent data leakage
Build ML Model — Train, generate predictions & probabilities, evaluate
Evaluation — Report accuracy
Questions to answer: Which variables are numerical/categorical? What's the target? Is it balanced?

Part B: Deep Learning (25%)
Neural Network Alternative — Improve the ML solution using TensorFlow/Keras.

Tasks:
7. Design Neural Network — Document: hidden layers, neurons, activation functions, optimizer, epochs
8. Train Model — Train on prepared data
9. Analyze Overfitting — Determine if underfitting/properly fitted/overfitting
10. Evaluate — Report accuracy

Questions: Why this architecture? What does increasing training accuracy + declining validation accuracy indicate?

Part C: LangChain + RAG (30%) ⭐
M&A Knowledge Assistant — Build a Q&A system from an M&A Playbook PDF ONLY.

Constraints: Use ONLY the provided M&A PDF—no internet search, Wikipedia, or external docs.

Tasks:
12. Load & Process PDF — Extract content, chunk with experimentation, document chunk size decisions
13. Generate Embeddings — Create embeddings for chunks using appropriate model
14. Vector Store — Store embeddings in FAISS
15. Build RAG Pipeline — Accept question → retrieve chunks → pass to LLM → return answer + sources
16. Prompt Engineering — Instruct LLM to: use only retrieved context, avoid hallucination, state when info unavailable

Questions: Why chunk documents? Effects of too-small/too-large chunks? How do embeddings enable semantic search?

Part D: Scenario-Based Assessment (20%)
No model building. Answer conceptual questions on when to use ML/DL/GenAI/RAG.

6 Scenarios across 3 domains:

Finance: Credit default prediction (ML) + Financial document assistant (RAG)
Telecom: Customer churn (ML) + Network operations copilot (RAG)
Healthcare: Patient readmission (ML) + Clinical knowledge assistant (RAG)
Focus areas: Problem formulation, technology selection, architecture, model selection, evaluation, business implications.

Key Insights:
Part A+B test progression: traditional ML → deep learning
Part C is practical GenAI — building a constrained RAG system
Part D tests decision-making: knowing which tool to use and why
All parts interconnected through the insurance claim dataset
Do you want help starting any specific part?