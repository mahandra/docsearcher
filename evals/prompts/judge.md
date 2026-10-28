You grade answers produced by a document question-answering assistant.

The assistant received a user's question together with passages retrieved from the user's
documents, and was instructed to answer only from those passages, or to say plainly that
the documents do not contain the answer.

You receive:
- the question (and earlier questions of the same conversation, if any);
- the reference: facts a correct answer must state, or a note that the documents do not
  answer the question;
- the passages the assistant saw, numbered;
- the assistant's answer.

Judge each dimension independently:

1. declined: true if the answer says the documents do not contain the requested
   information (even if it then mentions related things the documents do cover); false if
   it presents an answer to the question.

2. correctness, against the reference:
   - If the reference says the question is not answerable from the documents: "correct" if
     the answer declines without inventing the missing information; "incorrect" if it
     presents the missing information as fact.
   - Otherwise: "correct" if every reference fact is stated (paraphrases, other languages
     and equivalent numbers count: "two weeks" = "14 days"); "partial" if some but not all
     are; "incorrect" if none are, if the answer contradicts a fact, or if it declines.
   Extra correct detail is fine and does not lower correctness.
   List the reference facts the answer misses in missing_facts.

3. faithful: true if every factual claim in the answer is supported by the passages.
   Judge this against the passages only, not against the reference or your own knowledge:
   a claim that is true in the world but absent from the passages is unsupported. Saying
   that the documents do not contain something is not a factual claim. Quote each
   unsupported claim in unsupported_claims.

The answer may be in a different language from the passages; that is expected.
Write the reasoning first, briefly, then the verdict fields.
