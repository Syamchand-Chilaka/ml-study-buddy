# Generative AI and RAG — AIML-500 Study Guide

## What is Generative AI?

Generative AI refers to models that create new content — text, images, code,
audio — rather than only labeling existing content. A classifier looks at an
email and says "spam"; a generative model can write the email. The generative
models behind chatbots are called large language models (LLMs).

## How Large Language Models Work

An LLM is a neural network trained on enormous amounts of text to do one
deceptively simple thing: predict the next token (a token is a word or piece
of a word). Given "The capital of France is", it predicts "Paris". Repeat
that prediction over and over and you get fluent paragraphs.

Analogy: an LLM is like the world's most well-read autocomplete. It has
absorbed patterns from so much text that completing sentences convincingly
requires something that looks a lot like knowledge and reasoning.

Important consequences of this design:

- LLMs have a knowledge cutoff: they only know what was in their training
  data, and nothing after it.
- They don't look things up by default: answers come from learned patterns,
  not from a database.
- They are probabilistic: the same question can get different answers.

## Prompting

A prompt is the text you give an LLM to steer its output. Good prompts give
context (who the model should be), a clear task, and constraints (format,
length, tone). A system prompt is a special standing instruction that frames
the entire conversation — like a job description the model reads before
talking to anyone.

## Hallucination

Hallucination is when an LLM confidently states something false. Because the
model is a next-token predictor and not a lookup engine, it would rather
produce a plausible-sounding answer than say "I don't know." It can invent
citations, dates, statistics, and names. Hallucination is the single biggest
obstacle to trusting LLMs in serious applications — and the main problem
that retrieval-augmented generation was designed to reduce.

## Embeddings: Turning Meaning into Numbers

An embedding is a list of numbers (a vector) that represents the meaning of
a piece of text. An embedding model converts text into these vectors such
that texts with similar meanings end up close together in vector space.
"How do I reduce overfitting?" and "ways to stop my model memorizing data"
produce nearby vectors even though they share almost no words.

Analogy: embeddings are like map coordinates for meaning. Just as two cafes
with nearby GPS coordinates are physically close, two sentences with nearby
embeddings are close in meaning.

Similarity between two embeddings is usually measured with cosine similarity,
which scores how much two vectors point in the same direction: 1.0 means
identical meaning, around 0 means unrelated.

## What is RAG (Retrieval-Augmented Generation)?

RAG is a technique that connects an LLM to a specific knowledge base, so its
answers are grounded in real documents instead of just its training memory.

Analogy: a plain LLM is a student answering from memory in a closed-book
exam. RAG turns it into an open-book exam: before answering, the student
first looks up the relevant pages, then writes an answer based on what the
pages actually say.

## The Five Steps of a RAG Pipeline

A RAG system works in five steps. The first three happen once, when the
knowledge base is prepared (indexing); the last two happen every time a
user asks a question.

1. Ingest: load the source documents (for example, a folder of course
   guides) into the system.

2. Chunk: split each document into smaller passages, called chunks, often
   a few hundred tokens each. Chunks are needed because LLM prompts have
   limited room, and because retrieval works better on focused passages
   than on whole documents. Splitting on natural boundaries like headings
   keeps each chunk about one coherent topic, and a small overlap between
   neighboring chunks avoids cutting ideas in half.

3. Embed: convert every chunk into an embedding vector using an embedding
   model, and store the vectors in an index (a vector database, or for
   small collections, simply an array in memory).

4. Retrieve: when a question arrives, embed the question with the same
   model, then find the chunks whose vectors are most similar to the
   question's vector (for example, the top 4 by cosine similarity). These
   chunks are the "open book pages" for this question.

5. Generate: build a prompt containing the retrieved chunks plus the user's
   question, and ask the LLM to answer using only that provided context.
   The model writes the final answer grounded in the retrieved text.

## Why RAG Helps

- Fewer hallucinations: the model answers from provided text, and can be
  instructed to refuse when the text doesn't contain the answer.
- Fresh and private knowledge: the knowledge base can contain documents the
  model never saw in training — yesterday's policy update, your course
  notes — without retraining anything.
- Traceability: because you know which chunks were retrieved, you can show
  sources for every answer, and a reader can verify them.
- Cost: updating a folder of documents is vastly cheaper than retraining
  or fine-tuning a model.

## RAG vs Fine-tuning

Fine-tuning means further training a model on your own examples, changing
its weights. It is good for teaching style, format, or specialized behavior.
RAG is good for giving access to specific, changeable knowledge. Rule of
thumb: fine-tuning teaches the model how to speak; RAG gives it something
to read. They can be combined.

## Limitations of RAG

RAG is only as good as its retrieval. If the right chunk isn't retrieved —
because the question is phrased oddly, the chunking split an idea apart, or
the knowledge base simply doesn't contain the answer — the model can't
ground its response. Good RAG systems therefore set a similarity threshold
and honestly say "I don't know" when nothing relevant is found, rather than
letting the model guess.

## Key Terms Recap

- Token: a word or word-piece; the unit LLMs read and write.
- Context window: the maximum amount of text an LLM can consider at once.
- Embedding: a vector representing text meaning.
- Cosine similarity: a measure of how close two embeddings are.
- Chunk: a passage of a document, the unit of retrieval.
- Grounding: tying a model's answer to retrieved source text.
- Hallucination: a confident but false model output.
