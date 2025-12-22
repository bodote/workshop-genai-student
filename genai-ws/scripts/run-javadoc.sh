#!/usr/bin/env bash

# Generate Javadoc for a JBang-based Java file, resolving dependencies via `jbang info classpath`.
# Usage: ./scripts/run-javadoc.sh [source-file] [output-dir]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_FILE="${1:-$REPO_ROOT/exercises/java/PromptEngineering.java}"
OUTPUT_DIR="${2:-$REPO_ROOT/tmp/javadoc}"
JAVA_SOURCE_VERSION="${JAVA_SOURCE_VERSION:-25}"

if ! command -v jbang >/dev/null 2>&1; then
  echo "jbang is required to resolve dependencies. Install it from https://www.jbang.dev/." >&2
  exit 1
fi

if ! command -v javadoc >/dev/null 2>&1; then
  echo "javadoc (from a JDK) must be available on your PATH." >&2
  exit 1
fi

if [ ! -f "$SOURCE_FILE" ]; then
  echo "Source file not found: $SOURCE_FILE" >&2
  exit 1
fi

CLASSPATH="$(jbang info classpath "$SOURCE_FILE")"
rm -rf "$OUTPUT_DIR" 
mkdir -p "$OUTPUT_DIR"

echo "Generating Javadoc for $SOURCE_FILE"
JAVADOC_OPTS=(--enable-preview --source "$JAVA_SOURCE_VERSION")
javadoc -d "$OUTPUT_DIR" -classpath "$CLASSPATH" "${JAVADOC_OPTS[@]}" "$SOURCE_FILE"
echo "Javadoc written to $OUTPUT_DIR"
