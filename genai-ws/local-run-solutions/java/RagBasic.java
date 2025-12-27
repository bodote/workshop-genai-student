///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS info.picocli:picocli:4.6.3
//DEPS com.google.genai:google-genai:1.31.0

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.ContentEmbedding;
import com.google.genai.types.EmbedContentConfig;
import com.google.genai.types.EmbedContentResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;

import static java.lang.System.out;

/// # GenAI Workshop
/// ## Lesson 3: Basic RAG
///
/// This lesson is intended to show you the basics of a Retrieval Augmented Generation (RAG) system.
///
/// During this lesson you will learn how to ...
/// - implement the different building blocks of RAG
/// - create a ingestion pipeline from the building blocks
/// - create a retrieval pipeline from the building blocks
/// - use a RAG system to generate responses to user inputs
@Command(name = "02_rag_basic", mixinStandardHelpOptions = true, version = "v0.1", description = "basic rag")
public class RagBasic implements Callable<Integer> {

    @Option(names = {"-e", "--exercise"}, defaultValue = "1",
            description = "Exercise number to run (1-2). Defaults to 1.")
    private int exerciseNumber;

    @Option(names = {"-q", "--question"}, defaultValue =
            "Lucy noticed a number on the ceiling when taking breakfast. Which number was written into the ceiling?",
            description = "Question to run through the RAG pipeline.")
    private String userQuestion;

    @Option(names = {"-v", "--verbose"}, defaultValue = "false",
            description = "Print retrieved context and augmented prompt.")
    private boolean verbose;

    private static final String DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite";
    private static final String GENERATION_MODEL = "gemini-2.5-flash-lite";
    private static final String EMBEDDING_MODEL = "models/text-embedding-004";

    private static final float DEFAULT_CONFIG_TEMPERATURE = 0.9f;
    private static final float DEFAULT_CONFIG_TOP_K = 3f;
    private static final int DEFAULT_CONFIG_MAX_OUTPUT_TOKENS = 200;
    private static final String DEFAULT_SYSTEM_PROMPT = "Your are a friendly assistant";

    private static final int DEFAULT_K = 3;
    private static final int DEFAULT_CHUNK_SIZE = 2000;
    private static final int DEFAULT_CHUNK_OVERLAP = 100;
    private static final Path BOOK_PATH = resolveBookPath();

    private final InMemoryVectorStore vectorStore = new InMemoryVectorStore();

    public static void main(String... args) {
        int exitCode = new CommandLine(new RagBasic()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        if (System.getenv("GOOGLE_API_KEY") == null || System.getenv("GOOGLE_API_KEY").isEmpty()) {
            out.println("GOOGLE_API_KEY is required to run this exercise.");
            return 1;
        }
        Client client = Client.builder().apiKey(System.getenv("GOOGLE_API_KEY")).build();
        if (exerciseNumber < 1 || exerciseNumber > 2) {
            out.println("Unknown exercise number: " + exerciseNumber + ". Please choose 1 or 2.");
            return 1;
        }

        switch (exerciseNumber) {
            case 1:
                runWithoutContext(client);
                break;
            case 2:
                runRagPipeline(client);
                break;
            default:
                return 1;
        }
        return 0;
    }

    /// ## Generate response without internal knowledge
    private void runWithoutContext(Client client) {
        printSeparator("Generate response without internal knowledge");
        String firstPrompt = "Lucy noticed a number on the ceiling when taking breakfast. Which number was written into the ceiling?";
        GenerateContentResponse firstResponse =
                client.models.generateContent(DEFAULT_GEMINI_MODEL, firstPrompt, defaultGenerationConfig());
        out.println(optionalText(firstResponse));
        out.println();

        String secondPrompt = """
                In one Sherlock Holmes book, Lucy noticed a number on the ceiling when taking breakfast.
                Which number was written into the ceiling?""";
        GenerateContentResponse secondResponse =
                client.models.generateContent(DEFAULT_GEMINI_MODEL, secondPrompt, defaultGenerationConfig());
        out.println(optionalText(secondResponse));
        out.println();
    }

    /// ### Exercise 01: Create RAG pipeline
    private void runRagPipeline(Client client) {
        printSeparator("Exercise 01: Create RAG pipeline");
        doIngestion(client, List.of(BOOK_PATH));
        if (verbose) {
            peekKnowledgebase();
        }
        doRag(client, userQuestion, verbose);
    }

    private GenerateContentConfig defaultGenerationConfig() {
        return GenerateContentConfig.builder()
                .maxOutputTokens(DEFAULT_CONFIG_MAX_OUTPUT_TOKENS)
                .temperature(DEFAULT_CONFIG_TEMPERATURE)
                .topK(DEFAULT_CONFIG_TOP_K)
                .systemInstruction(Content.fromParts(Part.fromText(DEFAULT_SYSTEM_PROMPT)))
                .build();
    }

    private String generateGeminiCompletion(Client client, String prompt) {
        GenerateContentResponse response =
                client.models.generateContent(GENERATION_MODEL, prompt, defaultGenerationConfig());
        return optionalText(response);
    }

    private void doIngestion(Client client, List<Path> filePaths) {
        for (Path path : filePaths) {
            String fileContent = loadFileContent(path);
            List<String> chunks = doChunk(fileContent);
            List<List<Float>> embeddings = doBatchEmbed(client, chunks, 100);
            persistEmbeddings(chunks, embeddings);
        }
    }

    private void doRag(Client client, String userInput, boolean verboseOutput) {
        out.println("Question:\n" + userInput);
        List<Float> userInputEmbedding = doEmbed(client, userInput);
        List<String> context = doTopKFetching(userInputEmbedding, DEFAULT_K);
        if (verboseOutput) {
            out.println("Retrieved context:");
            for (String chunk : context) {
                out.println("- " + chunk);
            }
            out.println();
        }
        String augmentedPrompt = augment(userInput, context);
        if (verboseOutput) {
            out.println("Augmented prompt:\n" + augmentedPrompt);
        }
        String response = generateGeminiCompletion(client, augmentedPrompt);
        out.println("Response:\n" + response);
    }

    private String loadFileContent(Path filePath) {
        try {
            return Files.readString(filePath);
        } catch (Exception e) {
            throw new IllegalStateException("Unable to read file " + filePath.toAbsolutePath(), e);
        }
    }

    private List<String> doChunk(String text) {
        List<String> separators = List.of("\n\n", "\n", ". ", " ", "");
        List<String> segments = recursiveSplit(text, separators, DEFAULT_CHUNK_SIZE);
        return assembleChunks(segments, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP);
    }

    private List<Float> doEmbed(Client client, String chunk) {
        EmbedContentResponse response =
                client.models.embedContent(EMBEDDING_MODEL, chunk, EmbedContentConfig.builder().build());
        return extractEmbedding(response, 0);
    }

    private List<List<Float>> doBatchEmbed(Client client, List<String> chunks, int batchSize) {
        List<List<Float>> allEmbeddings = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i += batchSize) {
            List<String> batch = chunks.subList(i, Math.min(i + batchSize, chunks.size()));
            EmbedContentResponse response =
                    client.models.embedContent(EMBEDDING_MODEL, batch, EmbedContentConfig.builder().build());
            List<ContentEmbedding> embeddings =
                    response.embeddings().orElseThrow(() -> new IllegalStateException("No embeddings returned."));
            for (int index = 0; index < embeddings.size(); index++) {
                allEmbeddings.add(extractEmbedding(response, index));
            }
        }
        return allEmbeddings;
    }

    private List<Float> extractEmbedding(EmbedContentResponse response, int index) {
        List<ContentEmbedding> embeddings =
                response.embeddings().orElseThrow(() -> new IllegalStateException("No embeddings returned."));
        if (index >= embeddings.size()) {
            throw new IllegalStateException("Embedding index out of range.");
        }
        return embeddings.get(index)
                .values()
                .orElseThrow(() -> new IllegalStateException("Missing embedding values."));
    }

    private void persistEmbeddings(List<String> chunks, List<List<Float>> embeddings) {
        vectorStore.addAll(chunks, embeddings);
    }

    private String augment(String userInput, List<String> context) {
        String preparedContext = String.join("\n", context);
        return """
                Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
                provided context just say, "answer is not available in the context", don't provide the wrong answer

                Context:
                %s

                Question:
                %s

                Answer:
                """.formatted(preparedContext, userInput);
    }

    private List<String> doTopKFetching(List<Float> userInputEmbedding, int topK) {
        return vectorStore.query(userInputEmbedding, topK);
    }

    private void peekKnowledgebase() {
        out.println("Knowledgebase peek:");
        List<String> sample = vectorStore.peek(10);
        for (String chunk : sample) {
            out.println("- " + chunk);
        }
        out.println();
    }

    private String optionalText(GenerateContentResponse response) {
        return response.text() != null ? response.text() : "";
    }

    private static Path resolveBookPath() {
        List<Path> candidates = List.of(
                Paths.get("../../data/processed/gutenberg/study_in_scarlett.txt"),
                Paths.get("../data/processed/gutenberg/study_in_scarlett.txt"),
                Paths.get("data/processed/gutenberg/study_in_scarlett.txt"),
                Paths.get("genai-ws/data/processed/gutenberg/study_in_scarlett.txt"));
        for (Path candidate : candidates) {
            Path absolute = candidate.toAbsolutePath().normalize();
            if (Files.exists(absolute)) {
                return absolute;
            }
        }
        throw new IllegalStateException("Could not locate study_in_scarlett.txt in expected locations.");
    }

    private void printSeparator(String title) {
        out.println("=== " + title + " ===");
    }

    private List<String> recursiveSplit(String text, List<String> separators, int chunkSize) {
        String trimmed = text.trim();
        if (trimmed.isEmpty()) {
            return List.of();
        }
        if (trimmed.length() <= chunkSize) {
            return List.of(trimmed);
        }
        if (separators.isEmpty()) {
            return hardSplit(trimmed, chunkSize);
        }
        String separator = separators.get(0);
        if (separator.isEmpty()) {
            return hardSplit(trimmed, chunkSize);
        }
        if (!trimmed.contains(separator)) {
            return recursiveSplit(trimmed, separators.subList(1, separators.size()), chunkSize);
        }

        String[] rawParts = trimmed.split(java.util.regex.Pattern.quote(separator), -1);
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < rawParts.length; i++) {
            String part = rawParts[i];
            if (i < rawParts.length - 1) {
                part = part + separator;
            }
            if (part.isBlank()) {
                continue;
            }
            if (part.length() <= chunkSize) {
                parts.add(part);
            } else {
                parts.addAll(recursiveSplit(part, separators.subList(1, separators.size()), chunkSize));
            }
        }
        return parts;
    }

    private List<String> hardSplit(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        for (int i = 0; i < text.length(); i += chunkSize) {
            int end = Math.min(text.length(), i + chunkSize);
            String slice = text.substring(i, end).trim();
            if (!slice.isEmpty()) {
                chunks.add(slice);
            }
        }
        return chunks;
    }

    private List<String> assembleChunks(List<String> segments, int chunkSize, int overlap) {
        List<String> chunks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (String segment : segments) {
            if (segment.isBlank()) {
                continue;
            }
            if (current.length() + segment.length() <= chunkSize) {
                current.append(segment);
                continue;
            }
            String chunk = current.toString().trim();
            if (!chunk.isEmpty()) {
                chunks.add(chunk);
            }
            current.setLength(0);
            if (!chunks.isEmpty() && overlap > 0) {
                String tail = chunks.get(chunks.size() - 1);
                int start = Math.max(0, tail.length() - overlap);
                current.append(tail.substring(start));
            }
            current.append(segment);
        }
        String chunk = current.toString().trim();
        if (!chunk.isEmpty()) {
            chunks.add(chunk);
        }
        return chunks;
    }

    private static final class InMemoryVectorStore {
        private final List<VectorEntry> entries = new ArrayList<>();

        void addAll(List<String> chunks, List<List<Float>> embeddings) {
            if (chunks.size() != embeddings.size()) {
                throw new IllegalArgumentException("Chunks and embeddings sizes must match.");
            }
            for (int i = 0; i < chunks.size(); i++) {
                entries.add(new VectorEntry(chunks.get(i), embeddings.get(i)));
            }
        }

        List<String> query(List<Float> queryEmbedding, int topK) {
            if (entries.isEmpty()) {
                return List.of();
            }
            return entries.stream()
                    .sorted(Comparator.comparingDouble(entry -> -cosineSimilarity(queryEmbedding, entry.embedding())))
                    .limit(topK)
                    .map(VectorEntry::text)
                    .toList();
        }

        List<String> peek(int count) {
            return entries.stream()
                    .limit(count)
                    .map(VectorEntry::text)
                    .toList();
        }

        private double cosineSimilarity(List<Float> a, List<Float> b) {
            if (a.size() != b.size()) {
                throw new IllegalArgumentException("Embedding dimensions do not match.");
            }
            double dot = 0.0;
            double normA = 0.0;
            double normB = 0.0;
            for (int i = 0; i < a.size(); i++) {
                double av = a.get(i);
                double bv = b.get(i);
                dot += av * bv;
                normA += av * av;
                normB += bv * bv;
            }
            if (normA == 0.0 || normB == 0.0) {
                return 0.0;
            }
            return dot / (Math.sqrt(normA) * Math.sqrt(normB));
        }
    }

    private record VectorEntry(String text, List<Float> embedding) {
    }
}
