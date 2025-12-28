///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
/**
 * with jbang , java classes with a main method do not need any build script like maven pom.xml instead
 * you just add the dependencies like this: 
 */
//DEPS info.picocli:picocli:4.6.3
//DEPS com.google.genai:google-genai:1.31.0   
//TODO run with `jbang SetupJbang.java`, see https://www.jbang.dev/documentation/jbang/latest/first-script.html
void main(String... args) {
    IO.println("Hello World");
}
