# Translation Python to Java

This repository contains Python/Jupyter notebooks and Java translations of the same material. Follow these rules when working on Java translations:

- Do not change any Python or Jupyter notebooks; treat them as read-only source material.
- For Java, use only jbang as the build/run tool. Do not introduce Maven, Gradle, or other build systems.
- Limit Java dependencies to `info.picocli:picocli` and `com.google.genai:google-genai` unless explicitly agreed otherwise.
- Use `genai-ws/exercises/java/PromptEngineering.java` as the template for new translations, mirroring its picocli setup, Javadoc/Markdown structure, and file layout.
- For RAG translations of the RAG notebook: mirror the Python pipeline with Java equivalents. Use a recursive character splitter (e.g., LangChain4j’s `DocumentSplitters.recursive` with chunk size/overlap), embed via Gemini (`models/text-embedding-004`), persist/query in a vector store such as Chroma (or in-memory if local only), then augment the prompt and call Gemini chat for generation. Keep chunking/embedding/retrieval/generation as clear, testable building blocks.
- keep any `TODO` comments from the jupyter notebooks exaclty as they are for the java code, use `///` and markdown for the java comments
- add this unicode character at the beginning of your answer: `🍏` if you followed these rules above.
