# Responsible AI and Course Study Guide — AIML-500

## What Does Ethical AI Mean in This Course?

In AIML-500, ethical AI means building and using AI systems in ways that are
fair, transparent, privacy-respecting, and accountable to the people they
affect. Three ideas anchor the course's treatment of ethics:

First, accuracy is not the same as ethics. A model can be highly accurate
and still cause harm — for example, a hiring model that is accurate on
historical data may simply be accurately reproducing past discrimination.
Technical performance never automatically makes a system right to deploy.
Always ask not only "does it work?" but "who does it affect, and how?"

Second, human oversight is required. AI systems in this course are treated
as decision support, not decision makers. A human must be able to review,
override, and take responsibility for consequential decisions — loan
approvals, medical flags, grading. This is often called keeping a
"human in the loop."

Third, privacy matters at every stage. Training data, user queries, and
model outputs can all leak personal information. Responsible practice means
collecting the minimum data needed, securing it, being transparent about its
use, and never repurposing personal data without consent.

## Fairness and Bias

AI bias is systematic unfairness in a model's outputs, usually inherited
from its training data. Models learn whatever patterns exist in the data —
including human prejudices baked into historical records. A face-recognition
system trained mostly on one demographic performs worse on others; a resume
screener trained on past hires can learn to prefer the kinds of people who
were hired before.

Key lesson: bias is not usually caused by malicious code. It arrives
silently through data. That is why the design-thinking emphasis on empathy —
understanding all the people a system affects, not just its builders —
comes first in this course's problem-solving framework.

Mitigations include: auditing datasets for representation, measuring
performance per subgroup (not just overall accuracy), diverse teams
reviewing systems, and ongoing monitoring after deployment.

## Transparency and Explainability

Transparency means being open about when AI is used and what data it relies
on. Explainability means being able to say why a model made a particular
prediction. Simple models like decision trees are easy to explain; deep
neural networks are much harder, which creates real tension between model
power and accountability. In high-stakes settings, a slightly less accurate
but explainable model is often the more responsible choice.

## Responsible Use of Generative AI in Your Studies

Generative AI tools can accelerate learning when used honestly: explaining
concepts, generating practice questions, critiquing your drafts. Used
dishonestly, they undermine the entire point of a course.

Academic integrity policy, in short: submitting AI-generated work as your
own is plagiarism. Assignments, reflections, discussion posts, and projects
must be your own thinking. Using an AI tutor to understand a concept and
then writing your own answer is learning; copying its output into a
submission is not. When a course permits AI assistance, disclose it.

A good self-test: if the tool disappeared tomorrow, could you still explain
and reproduce the work you submitted? If not, the tool did the learning
instead of you.

## Study Tips for AIML-500

- Space it out: three 30-minute sessions beat one 90-minute cram. Memory
  consolidates between sessions.
- Test yourself: retrieval practice (answering questions from memory) builds
  far stronger recall than re-reading notes. Use the quiz bank below.
- Explain it aloud: if you can teach a concept in plain English to a friend
  — or a rubber duck — you actually understand it.
- Connect analogies to definitions: first grasp the intuition (open-book
  exam), then attach the technical term (retrieval-augmented generation).
- Do the labs: concepts stick when you've touched real data and real errors.
- Review mistakes: a wrong quiz answer is worth more than a right one,
  because it shows you exactly where to spend your next study session.

## Practice Quiz Bank

Use these for self-testing. Try to answer before reading the answer.

### Question 1
What is the difference between supervised and unsupervised learning?

Answer: Supervised learning uses labeled data — every training example has
a correct answer — to learn a mapping from inputs to outputs. Unsupervised
learning uses unlabeled data and finds structure on its own, such as
clustering similar items. If an answer key exists, it's supervised.

### Question 2
A model gets 98% accuracy on its training data but only 60% on new data.
What is this called, and what could help?

Answer: This is overfitting — the model memorized the training data instead
of learning the general pattern. Remedies include more training data, a
simpler model, regularization, and early stopping.

### Question 3
Why is accuracy a poor metric for a fraud-detection model?

Answer: Because fraud data is heavily imbalanced. If 1% of transactions are
fraud, predicting "not fraud" every time scores 99% accuracy while catching
nothing. Precision, recall, and F1 reveal performance on the rare class
that accuracy hides.

### Question 4
In a RAG system, what is the purpose of chunking documents?

Answer: Chunking splits documents into focused passages so retrieval can
find the specific text relevant to a question, and so the retrieved context
fits in the LLM's prompt. Splitting on headings keeps chunks coherent, and
overlap avoids cutting ideas in half.

### Question 5
What is a hallucination, and how does RAG reduce it?

Answer: A hallucination is a confident but false statement from an LLM.
RAG reduces hallucination by retrieving real passages from a trusted
knowledge base and instructing the model to answer only from them — and to
say "I don't know" when nothing relevant is retrieved.

### Question 6
What does "human in the loop" mean, and why does it matter?

Answer: It means a human reviews and can override an AI system's outputs
before consequential decisions are made. It matters because models make
mistakes and inherit bias, and accountability for decisions must rest with
people, not software.

### Question 7
What is an embedding?

Answer: A vector of numbers representing the meaning of text, produced so
that similar meanings land close together. Closeness is typically measured
with cosine similarity, which is how RAG retrieval finds relevant chunks.

### Question 8
Give an example of how a model can be accurate but still unethical.

Answer: A hiring model trained on historical decisions can be highly
accurate at predicting past outcomes while reproducing past discrimination
against certain groups. Accuracy measures agreement with the data; ethics
asks whether the data — and the system's impact — are fair.
