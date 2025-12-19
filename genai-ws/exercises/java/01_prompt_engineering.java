///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS info.picocli:picocli:4.6.3
//DEPS com.google.genai:google-genai:1.31.0   

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Parameters;
import com.google.genai.*;
import com.google.genai.types.Content;
import com.google.genai.types.Candidate;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;

import static java.lang.System.out;

@Command(name = "book_order", mixinStandardHelpOptions = true, version = "v0.1", description = "order a book")
class BookOrder implements Callable<Integer> {

    @Parameters(index = "0", description = "The greeting to print", defaultValue = "User!")
    private String greeting;

    private static final String MODEL = "gemini-2.5-flash-lite";
    private static final float TEMPERATURE = 0.9f;
    private static final float TOP_K = 1f;
    private static final int MAX_OUTPUT_TOKENS = 200;
    private static final Path BOOK_DB = resolveBookDatabasePath();

    private final List<Content> history = new ArrayList<>();

    public static void main(String... args) {
        int exitCode = new CommandLine(new BookOrder()).execute(args);
        System.exit(exitCode);
    }
/**
 * see https://ai.google.dev/gemini-api/docs#java
 */
    @Override
    public Integer call() throws Exception {
        System.out.println("Hello " + greeting);
        if (System.getenv("GOOGLE_API_KEY") == null || System.getenv("GOOGLE_API_KEY").isEmpty()) {
            System.out.println("you need the api key");
            return 1;
        }
        out.println("Api-Key all good");

        Client client = Client.builder().apiKey(System.getenv("GOOGLE_API_KEY")).build();

        clearHistory();
        runExercise01(client);
        runExercise02(client);
        runExercise03(client);
        runExercise04(client);

        return 0; 
    }

    private void runExercise01(Client client) throws Exception {
        printSeparator("Exercise 01: Flipped interaction");
        String userPrompt = "I'm looking for this one book about a detective.";
        String systemPrompt = """
                You are an employee in a bookstore. Ask clarifying questions first if the customer \
                did not provide enough details to find the book they want. Keep the tone friendly \
                and concise.
                """;
        GenerateContentResponse response =
                generateBookstoreBotCompletion(client, systemPrompt, userPrompt, false, false);
        printCompletionResult(response, false);
        out.println();
    }

    private void runExercise02(Client client) throws Exception {
        printSeparator("Exercise 02: Basic interaction");
        clearHistory();
        String userPrompt =
                "I'm looking for this book, where Sherlock Holmes and Watson meet the first time.";
        String systemPrompt = """
                You are an employee in a bookstore. When a customer describes a book roughly, use \
                the provided book list to identify the best match, share the title, publication \
                year, and a short summary, then ask if you should order the book.
                """;
        GenerateContentResponse response =
                generateBookstoreBotCompletion(client, systemPrompt, userPrompt, false, false);
        printCompletionResult(response, false);
        out.println();
    }

    private void runExercise03(Client client) throws Exception {
        printSeparator("Exercise 03: Extend the process");
        clearHistory();
        String systemPrompt = """
                You are an employee in a bookstore. If a customer only has a rough idea, search \
                the provided books list for the closest match, tell the customer the name of the \
                book, and ask whether you should order it. After the customer confirms, reply with \
                only the ISBN in the format {"isbn": "<isbn-number>"} so an API can place the order.
                """;

        String firstCustomerMessage =
                "I'm looking for this book, where Sherlock Holmes and Watson meet the first time.";
        out.println("Customer:\n" + firstCustomerMessage + "\n");
        GenerateContentResponse firstBotAnswer =
                generateBookstoreBotCompletion(client, systemPrompt, firstCustomerMessage, false, true);
        out.println("Bookstore bot:\n" + optionalText(firstBotAnswer) + "\n");

        String customerAnswer = "Yes, I like to order the book.";
        out.println("Customer:\n" + customerAnswer + "\n");
        GenerateContentResponse secondBotAnswer =
                generateBookstoreBotCompletion(client, systemPrompt, customerAnswer, false, true);
        out.println("Bookstore bot:\n" + optionalText(secondBotAnswer) + "\n");

        // Simulate calling the order API with the known Sherlock Holmes ISBN from the dataset.
        orderBook("978-3-51593-12345-6");
        out.println();
    }

    private void runExercise04(Client client) throws Exception {
        printSeparator("Exercise 04: Give choices");
        String systemPrompt = """
                You are an employee in a bookstore. When a customer gives a rough description, \
                pick the most appropriate book from the provided list. Share the title and ask \
                whether to order a physical copy or provide an ebook link. If the user wants an \
                ebook, share the URL from the data. If they want paper, respond with \
                {"isbn": "<isbn-number>"} so it can be ordered.
                """;

        clearHistory();
        String initialPrompt =
                "I'm looking for this book, where Sherlock Holmes and Watson meet the first time.";
        out.println("Customer:\n" + initialPrompt + "\n");
        GenerateContentResponse firstBotAnswer =
                generateBookstoreBotCompletion(client, systemPrompt, initialPrompt, false, true);
        out.println("Bookstore bot:\n" + optionalText(firstBotAnswer) + "\n");

        String ebookRequest =
                "I love ebooks. Why should I order a book, if I get an ebook for free. Please provide the link!";
        out.println("Customer:\n" + ebookRequest + "\n");
        GenerateContentResponse ebookAnswer =
                generateBookstoreBotCompletion(client, systemPrompt, ebookRequest, false, true);
        out.println("Bookstore bot:\n" + optionalText(ebookAnswer) + "\n");

        // Verify paper flow still works.
        clearHistory();
        out.println("Customer:\n" + initialPrompt + "\n");
        GenerateContentResponse restartAnswer =
                generateBookstoreBotCompletion(client, systemPrompt, initialPrompt, false, true);
        out.println("Bookstore bot:\n" + optionalText(restartAnswer) + "\n");

        String orderRequest =
                "Please order the book. I love spending money for stuff I can get for free!";
        out.println("Customer:\n" + orderRequest + "\n");
        GenerateContentResponse orderAnswer =
                generateBookstoreBotCompletion(client, systemPrompt, orderRequest, false, true);
        out.println("Bookstore bot:\n" + optionalText(orderAnswer) + "\n");
    }

    private GenerateContentResponse generateBookstoreBotCompletion(
            Client client,
            String systemPrompt,
            String userPrompt,
            boolean verbose,
            boolean expectJsonMimeType)
            throws Exception {

        // Add user message to conversation history.
        history.add(Content.fromParts(Part.fromText(userPrompt)));

        GenerateContentConfig.Builder configBuilder = GenerateContentConfig.builder()
                .temperature(TEMPERATURE)
                .topK(TOP_K)
                .maxOutputTokens(MAX_OUTPUT_TOKENS)
                .systemInstruction(Content.fromParts(Part.fromText(systemPrompt)));

        if (expectJsonMimeType) {
            configBuilder.responseMimeType("application/json");
        }

        GenerateContentResponse response =
                client.models.generateContent(MODEL, history, configBuilder.build());

        // Add model reply to history to keep the dialogue flowing.
        String responseText = optionalText(response);
        history.add(Content.builder().role("model").parts(Part.fromText(responseText)).build());

        if (verbose) {
            out.println("User Prompt: " + userPrompt);
            out.println("Assistant Response: " + responseText);
        }
        return response;
    }

    private void clearHistory() {
        history.clear();
        // Seed the conversation with the book database so the model can reference it.
        history.add(
                Content.builder()
                        .role("user")
                        .parts(Part.fromBytes(readBookDatabase(), "text/plain"))
                        .build());
    }

    private byte[] readBookDatabase() {
        try {
            return Files.readAllBytes(BOOK_DB);
        } catch (Exception e) {
            throw new IllegalStateException("Unable to read book database at " + BOOK_DB, e);
        }
    }

    private void printCompletionResult(GenerateContentResponse completionResult, boolean full) {
        if (full) {
            out.println(completionResult);
        } else {
            out.println(optionalText(completionResult));
        }
    }

    private String optionalText(GenerateContentResponse response) {
        return response.text() != null ? response.text() : extractTextFromCandidate(response);
    }

    private String extractTextFromCandidate(GenerateContentResponse response) {
        Optional<Candidate> candidate =
                response.candidates().flatMap(list -> list.stream().findFirst());
        return candidate.flatMap(c -> c.content().map(Content::text)).orElse("");
    }

    private void orderBook(String isbn) throws InterruptedException {
        out.println("Ordering book with ISBN: " + isbn);
        for (int i = 0; i < 5; i++) {
            out.print(".");
            out.flush();
            Thread.sleep(Duration.ofMillis(500));
        }
        out.println();
        if ("978-3-51593-12345-6".equals(isbn)) {
            out.println("Success: You ordered 'A Study in Scarlet'!");
            out.println("You completed this exercise successfully!");
        } else {
            out.println("Error: Unknown ISBN number.");
        }
    }

    private void printSeparator(String title) {
        out.println("=== " + title + " ===");
    }

    private static Path resolveBookDatabasePath() {
        List<Path> candidates = List.of(
                Paths.get("../../data/processed/gutenberg/books.db"),
                Paths.get("../data/processed/gutenberg/books.db"),
                Paths.get("data/processed/gutenberg/books.db"),
                Paths.get("genai-ws/data/processed/gutenberg/books.db"));
        for (Path candidate : candidates) {
            Path absolute = candidate.toAbsolutePath().normalize();
            if (Files.exists(absolute)) {
                return absolute;
            }
        }
        throw new IllegalStateException("Could not locate books.db. Checked common relative locations from working directory.");
    }
}
