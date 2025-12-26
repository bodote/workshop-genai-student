///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS info.picocli:picocli:4.6.3
//DEPS com.google.genai:google-genai:1.31.0

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import com.google.genai.Client;
import com.google.genai.types.ContentEmbedding;
import com.google.genai.types.EmbedContentConfig;
import com.google.genai.types.EmbedContentResponse;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;

import static java.lang.System.out;

/// # GenAI Workshop
/// ## Lesson 5: RAG Evaluation & Optimization
///
/// This lesson is intended to show you how different RAG configurations affect the output quality of a Retrieval Augmented Generation system.
///
/// During this lesson you will learn how to ...
/// - evaluate the quality of a RAG system using simple quality metrics
/// - manipulate the *chunk size* and *top_k* to improve the system
@Command(name = "04_rag_evals", mixinStandardHelpOptions = true, version = "v0.1", description = "rag evaluation and optimization")
public class RagEvals implements Callable<Integer> {

    @Option(names = {"-e", "--exercise"}, defaultValue = "1",
            description = "Exercise number to run (1-3). Defaults to 1.")
    private int exerciseNumber;

    @Option(names = {"--top-k"}, defaultValue = "3",
            description = "Top-k value for retrieval evaluation.")
    private int topK;

    @Option(names = {"--chunk-size"}, defaultValue = "500",
            description = "Chunk size used for exercise 3.")
    private int chunkSize;

    private static final String EMBEDDING_MODEL = "models/text-embedding-004";

    private static final int DEFAULT_TOP_K = 3;
    private static final int DEFAULT_CHUNK_OVERLAP = 100;
    private static final int DEFAULT_CHUNK_SIZE = 2000;
    private static final int CHARACTERS_PER_PAGE = 1800;

    private static final Path PROCESSED_DATA_PATH = resolveProcessedDataPath();
    private static final Path EVALUATION_DATA_PATH = resolveEvaluationDataPath();

    private final InMemoryVectorStore vectorStore = new InMemoryVectorStore();

    public static void main(String... args) {
        int exitCode = new CommandLine(new RagEvals()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        if (System.getenv("GOOGLE_API_KEY") == null || System.getenv("GOOGLE_API_KEY").isEmpty()) {
            out.println("GOOGLE_API_KEY is required to run this exercise.");
            return 1;
        }
        if (exerciseNumber < 1 || exerciseNumber > 3) {
            out.println("Unknown exercise number: " + exerciseNumber + ". Please choose between 1 and 3.");
            return 1;
        }

        Client client = Client.builder().apiKey(System.getenv("GOOGLE_API_KEY")).build();
        switch (exerciseNumber) {
            case 1:
                runExercise01(client);
                break;
            case 2:
                runExercise02(client);
                break;
            case 3:
                runExercise03(client);
                break;
            default:
                return 1;
        }
        return 0;
    }

    private void runExercise01(Client client) {
        printSeparator("Exercise 01: Evaluate RAG");
        Set<String> knowledgebaseContent = new LinkedHashSet<>();
        knowledgebaseContent.add("study_in_scarlett.txt");
        doIngestion(client, knowledgebaseContent, DEFAULT_CHUNK_SIZE, false, false);

        List<FetchedChunk> fetchedChunks = doRetrieval(client,
                "Which number did Lucie noticed on the ceiling?",
                DEFAULT_TOP_K);

        out.println("Fetched " + fetchedChunks.size() + " chunks from the knowledgebase");
        for (FetchedChunk chunk : fetchedChunks) {
            out.println("Chunk: " + chunk.chunk());
            out.println("Metadata: " + chunk.metadata());
            out.println();
        }

        out.println("Metadata of the chunk:\t" + fetchedChunks.get(0).metadata());
        out.println("Origin of the chunk:\t" + getOrigin(fetchedChunks.get(0)));

        EvaluationMetrics metrics = evaluateRetrieval(client, DEFAULT_TOP_K);
        printMetrics(metrics);
    }

    /// ### Exercise 01: Evaluate RAG (again)
    private void runExercise02(Client client) {
        printSeparator("Exercise 01: Evaluate RAG (again)");
        Set<String> knowledgebaseContent = new LinkedHashSet<>();
        knowledgebaseContent.add("study_in_scarlett.txt");
        knowledgebaseContent.add("adventures_of_sherlock_holmes.txt");
        doIngestion(client, knowledgebaseContent, DEFAULT_CHUNK_SIZE, true, false);

        /// TODO: Evaluate the retrieval performance
        EvaluationMetrics metrics = evaluateRetrieval(client, topK);

        /// TODO: Print the results
        printMetrics(metrics);
    }

    /// ### Exercise 02: Optimize RAG
    private void runExercise03(Client client) {
        printSeparator("Exercise 02: Optimize RAG");
        Set<String> knowledgebaseContent = new LinkedHashSet<>();
        knowledgebaseContent.add("study_in_scarlett.txt");
        knowledgebaseContent.add("adventures_of_sherlock_holmes.txt");

        /// TODO: Find the optimal configuration
        /// Be careful: Do not set the chunk_size below 250, otherwise the ingestion will take too long
        /// TODO: Set the chunk size
        int selectedChunkSize = chunkSize;
        /// TODO: Set the top_k
        int selectedTopK = topK;

        doIngestion(client, knowledgebaseContent, selectedChunkSize, true, false);

        EvaluationMetrics metrics = evaluateRetrieval(client, selectedTopK);
        printMetrics(metrics);
    }

    private void printMetrics(EvaluationMetrics metrics) {
        out.printf("Average Precision: %.2f%n", metrics.precision());
        out.printf("Average Reciprocal Rank: %.2f%n", metrics.reciprocalRank());
        out.printf("Average Hit Rate: %.2f%n", metrics.hitRate());
    }

    private void doIngestion(Client client,
                             Set<String> fileNames,
                             int chunkSize,
                             boolean clearKnowledgebase,
                             boolean verbose) {
        if (chunkSize < 250) {
            throw new IllegalArgumentException("chunk_size must higher than 250");
        }
        if (clearKnowledgebase) {
            vectorStore.clear();
        }

        for (String fileName : fileNames) {
            String fileContent = loadFileContent(fileName);
            ChunkResult chunkResult = doChunk(fileContent, chunkSize, DEFAULT_CHUNK_OVERLAP);
            if (verbose) {
                out.println("Loaded " + chunkResult.chunks().size() + " chunks from " + fileName);
            }
            List<List<Float>> embeddings = doBatchEmbed(client, chunkResult.chunks(), 100);
            List<Metadata> metadatas = new ArrayList<>();
            for (int page : chunkResult.pageNumbers()) {
                metadatas.add(new Metadata(page, fileName));
            }
            persistEmbeddings(chunkResult.chunks(), embeddings, metadatas);
        }

        if (verbose) {
            out.println("Added " + vectorStore.size() + " chunks to the knowledgebase");
        }
    }

    private List<FetchedChunk> doRetrieval(Client client, String userInput, int topK) {
        List<Float> userInputEmbedding = doEmbed(client, userInput);
        return doTopKFetching(userInputEmbedding, topK);
    }

    private List<FetchedChunk> doTopKFetching(List<Float> userInputEmbedding, int topK) {
        return vectorStore.query(userInputEmbedding, topK);
    }

    private String loadFileContent(String fileName) {
        Path filePath = PROCESSED_DATA_PATH.resolve(fileName);
        try {
            return Files.readString(filePath);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read file " + filePath.toAbsolutePath(), e);
        }
    }

    private ChunkResult doChunk(String text, int chunkSize, int chunkOverlap) {
        List<String> separators = List.of("\n\n", "\n", ". ", " ", "");
        List<String> segments = recursiveSplit(text, separators, chunkSize);
        List<String> chunks = assembleChunks(segments, chunkSize, chunkOverlap);
        List<Integer> pageNumbers = new ArrayList<>();
        for (String chunk : chunks) {
            pageNumbers.add(getPageNumber(chunk, text, CHARACTERS_PER_PAGE));
        }
        return new ChunkResult(chunks, pageNumbers);
    }

    private int getPageNumber(String text, String bookText, int charactersPerPage) {
        int startIndex = bookText.indexOf(text);
        if (startIndex == -1) {
            return -1;
        }
        return (startIndex / charactersPerPage) + 1;
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

    private void persistEmbeddings(List<String> chunks, List<List<Float>> embeddings, List<Metadata> metadatas) {
        vectorStore.addAll(chunks, embeddings, metadatas);
    }

    private String getOrigin(FetchedChunk chunk) {
        return chunk.metadata().book() + ":" + chunk.metadata().page();
    }

    private int calculateHitRate(List<String> groundTruth, List<String> retrieved) {
        Set<String> gtSet = Set.copyOf(groundTruth);
        Set<String> retrievedSet = Set.copyOf(retrieved);
        for (String item : gtSet) {
            if (retrievedSet.contains(item)) {
                return 1;
            }
        }
        return 0;
    }

    private double calculateReciprocalRank(List<String> groundTruth, List<String> retrieved) {
        Set<String> gtSet = Set.copyOf(groundTruth);
        for (int rank = 0; rank < retrieved.size(); rank++) {
            if (gtSet.contains(retrieved.get(rank))) {
                return 1.0 / (rank + 1);
            }
        }
        return 0.0;
    }

    private double calculatePrecision(List<String> groundTruth, List<String> retrieved) {
        int topK = retrieved.size();
        if (topK == 0) {
            return 0.0;
        }
        Set<String> relevant = new LinkedHashSet<>(groundTruth);
        relevant.retainAll(retrieved);
        return (double) relevant.size() / topK;
    }

    private EvaluationMetrics evaluateRetrieval(Client client, int topK) {
        List<EvaluationRow> rows = loadEvaluationDataset();
        List<Double> precisions = new ArrayList<>();
        List<Double> reciprocalRanks = new ArrayList<>();
        List<Integer> hitRates = new ArrayList<>();

        for (EvaluationRow row : rows) {
            List<FetchedChunk> retrievedChunks = doRetrieval(client, row.question(), topK);
            List<String> retrievedOrigins = new ArrayList<>();
            for (FetchedChunk chunk : retrievedChunks) {
                retrievedOrigins.add(getOrigin(chunk));
            }

            precisions.add(calculatePrecision(List.of(row.groundTruthOrigin()), retrievedOrigins));
            reciprocalRanks.add(calculateReciprocalRank(List.of(row.groundTruthOrigin()), retrievedOrigins));
            hitRates.add(calculateHitRate(List.of(row.groundTruthOrigin()), retrievedOrigins));
        }

        double avgPrecision = average(precisions);
        double avgReciprocalRank = average(reciprocalRanks);
        double avgHitRate = hitRates.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        return new EvaluationMetrics(avgPrecision, avgReciprocalRank, avgHitRate);
    }

    private List<EvaluationRow> loadEvaluationDataset() {
        Path datasetPath = EVALUATION_DATA_PATH.resolve("evaluation_dataset.csv");
        List<EvaluationRow> rows = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(datasetPath)) {
            String line = reader.readLine();
            if (line == null) {
                return rows;
            }
            while ((line = reader.readLine()) != null) {
                List<String> columns = parseCsvLine(line);
                if (columns.size() < 2) {
                    continue;
                }
                String question = columns.get(0);
                String groundTruthOrigin = columns.get(1);
                rows.add(new EvaluationRow(question, groundTruthOrigin));
            }
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read evaluation dataset at " + datasetPath, e);
        }
        return rows;
    }

    private List<String> parseCsvLine(String line) {
        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == ',' && !inQuotes) {
                values.add(current.toString());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        values.add(current.toString());
        return values;
    }

    private double average(List<Double> values) {
        if (values.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.size();
    }

    private void printSeparator(String title) {
        out.println("=== " + title + " ===");
    }

    private static Path resolveProcessedDataPath() {
        List<Path> candidates = List.of(
                Paths.get("../data/processed/gutenberg"),
                Paths.get("../../data/processed/gutenberg"),
                Paths.get("data/processed/gutenberg"),
                Paths.get("genai-ws/data/processed/gutenberg"));
        for (Path candidate : candidates) {
            Path absolute = candidate.toAbsolutePath().normalize();
            if (Files.isDirectory(absolute)) {
                return absolute;
            }
        }
        throw new IllegalStateException("Could not locate processed Gutenberg data.");
    }

    private static Path resolveEvaluationDataPath() {
        List<Path> candidates = List.of(
                Paths.get("../data/evaluation"),
                Paths.get("../../data/evaluation"),
                Paths.get("data/evaluation"),
                Paths.get("genai-ws/data/evaluation"));
        for (Path candidate : candidates) {
            Path absolute = candidate.toAbsolutePath().normalize();
            if (Files.isDirectory(absolute)) {
                return absolute;
            }
        }
        throw new IllegalStateException("Could not locate evaluation dataset folder.");
    }

    private List<String> recursiveSplit(String text, List<String> separators, int chunkSize) {
        if (text.isEmpty()) {
            return List.of();
        }
        if (text.length() <= chunkSize) {
            return List.of(text);
        }
        if (separators.isEmpty()) {
            return hardSplit(text, chunkSize);
        }
        String separator = separators.get(0);
        if (separator.isEmpty()) {
            return hardSplit(text, chunkSize);
        }
        if (!text.contains(separator)) {
            return recursiveSplit(text, separators.subList(1, separators.size()), chunkSize);
        }

        String[] rawParts = text.split(java.util.regex.Pattern.quote(separator), -1);
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < rawParts.length; i++) {
            String part = rawParts[i];
            if (i < rawParts.length - 1) {
                part = part + separator;
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
            chunks.add(text.substring(i, end));
        }
        return chunks;
    }

    private List<String> assembleChunks(List<String> segments, int chunkSize, int overlap) {
        List<String> chunks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (String segment : segments) {
            if (segment.isEmpty()) {
                continue;
            }
            if (current.length() + segment.length() <= chunkSize) {
                current.append(segment);
                continue;
            }
            if (current.length() > 0) {
                chunks.add(current.toString());
            }
            current.setLength(0);
            if (!chunks.isEmpty() && overlap > 0) {
                String tail = chunks.get(chunks.size() - 1);
                int start = Math.max(0, tail.length() - overlap);
                current.append(tail.substring(start));
            }
            current.append(segment);
        }
        if (current.length() > 0) {
            chunks.add(current.toString());
        }
        return chunks;
    }

    private static final class InMemoryVectorStore {
        private final List<VectorEntry> entries = new ArrayList<>();

        void addAll(List<String> chunks, List<List<Float>> embeddings, List<Metadata> metadatas) {
            if (chunks.size() != embeddings.size() || chunks.size() != metadatas.size()) {
                throw new IllegalArgumentException("Chunks, embeddings, and metadata sizes must match.");
            }
            for (int i = 0; i < chunks.size(); i++) {
                entries.add(new VectorEntry(chunks.get(i), embeddings.get(i), metadatas.get(i)));
            }
        }

        List<FetchedChunk> query(List<Float> queryEmbedding, int topK) {
            if (entries.isEmpty()) {
                return List.of();
            }
            return entries.stream()
                    .sorted(Comparator.comparingDouble(entry -> -cosineSimilarity(queryEmbedding, entry.embedding())))
                    .limit(topK)
                    .map(entry -> new FetchedChunk(entry.text(), entry.metadata()))
                    .toList();
        }

        void clear() {
            entries.clear();
        }

        int size() {
            return entries.size();
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

    private record VectorEntry(String text, List<Float> embedding, Metadata metadata) {
    }

    private record Metadata(int page, String book) {
    }

    private record FetchedChunk(String chunk, Metadata metadata) {
    }

    private record ChunkResult(List<String> chunks, List<Integer> pageNumbers) {
    }

    private record EvaluationRow(String question, String groundTruthOrigin) {
    }

    private record EvaluationMetrics(double precision, double reciprocalRank, double hitRate) {
    }
}
