# Contributing

This repository contains Python/Jupyter notebooks and Java translations of the same material. Follow these rules when working on Java translations:

- Do not change any Python or Jupyter notebooks; treat them as read-only source material.
- For Java, use only jbang as the build/run tool. Do not introduce Maven, Gradle, or other build systems.
- Limit Java dependencies to `info.picocli:picocli` and `com.google.genai:google-genai` unless explicitly agreed otherwise.
- Use `genai-ws/exercises/java/PromptEngineering.java` as the template for new translations, mirroring its picocli setup, Javadoc/Markdown structure, and file layout.
