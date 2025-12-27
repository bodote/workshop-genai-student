///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS info.picocli:picocli:4.6.3
//DEPS com.google.genai:google-genai:1.31.0   

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Parameters;
import com.google.genai.*;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;

import java.util.concurrent.Callable;
import static java.lang.System.out;

@Command(name = "hello", mixinStandardHelpOptions = true, version = "hello 0.1", description = "hello made with jbang")
class hello implements Callable<Integer> {

    @Parameters(index = "0", description = "The greeting to print", defaultValue = "User!")
    private String greeting;

    public static void main(String... args) {
        int exitCode = new CommandLine(new hello()).execute(args);
        System.exit(exitCode);
    }
/**
 * see https://ai.google.dev/gemini-api/docs#java
 */
    @Override
    public Integer call() throws Exception {
        System.out.println("Hello " + greeting);
        if (System.getenv("GOOGLE_API_KEY").isEmpty()) {
            System.out.println("you need the api key");
        } else {
            out.println("Api-Key all good");
        }
        Client client = Client.builder().apiKey(System.getenv("GOOGLE_API_KEY")).build();

        // set GenAI model to use
        var MODEL = "gemini-2.5-flash-lite";
        var SYSTEM_PROMPT = "You are a friendly assistant with a preference for the United Kingdom.";
        var USER_PROMPT = "What is the most beautiful city in the world? make the answer short an brief";

        var TEMPERATURE = 0.9f;
        var MAX_OUTPUT_TOKENS = 200;
        var TOP_K = 2f;

        // Create the generation configuration. We can change this for every requested
        // completion. See https://ai.google.dev/gemini-api/docs#java

        GenerateContentConfig config = GenerateContentConfig.builder()
                .temperature(TEMPERATURE)
                .maxOutputTokens(MAX_OUTPUT_TOKENS)
                .topK(TOP_K)
                .systemInstruction(Content.fromParts(Part.fromText(SYSTEM_PROMPT)))
                .build();

       // TODO: ### Exercise 01: Generate response by calling the generate_content method from client.models. 
       // Use the model, user prompt and config as parameters.

        GenerateContentResponse resp = null;
        out.println("output: %s".formatted(resp));

        // TODO ### Exercise 02: Analyse complete response
        // Print the whole response object and familiarize with the attributes of the response. 
        // You can have a look at the [documentation]
        // (https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse) for better understanding.

        return 0; 
    }
}
