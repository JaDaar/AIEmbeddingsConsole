# AIEmbeddingsConsole

Local semantic search over text files, written in C# (.NET 10), using [Ollama](https://ollama.com) to generate embeddings. Your data stays on your machine.

## Prerequisites

1. .NET 10 SDK
2. Ollama: `winget install --id Ollama.Ollama -e --source winget`
3. The embedding model: `ollama pull nomic-embed-text`

Ollama runs in the background (system tray) and serves an API at `http://localhost:11434`.

## Usage

```powershell
dotnet run -- index sample-data                    # embed all .txt files in a folder
dotnet run -- search "why does bread rise"         # one-off search
dotnet run -- search                               # interactive search
dotnet run -- info                                 # index statistics
dotnet run -- --help                               # all options
```

Re-running `index` only re-embeds files whose contents changed. Files deleted from the folder are removed from the index.

## How it works

| File | Purpose |
|------|---------|
| `Program.cs` | Command-line entry point (`index`, `search`, `info`) |
| `TextChunker.cs` | Splits text into ~1000-char chunks on paragraph and sentence boundaries, with overlap |
| `OllamaEmbeddingClient.cs` | Calls Ollama's `POST /api/embed` with batches of chunks |
| `VectorStore.cs` | Stores chunks and vectors in `embeddings-index.json`; cosine-similarity search |

1. **Index:** each file → chunks → 768-dimension vectors from `nomic-embed-text` → saved to JSON.
2. **Search:** the query → a vector → compared to every chunk with cosine similarity → top N results.

`nomic-embed-text` expects the task prefixes `search_document:` and `search_query:`. The app adds them automatically. If you switch models (`--model mxbai-embed-large`, for example), use `--rebuild`, because vectors from different models can't be compared.
