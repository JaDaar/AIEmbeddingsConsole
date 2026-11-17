# AIEmbeddingsConsole

Local semantic search over text files, written in C# (.NET 10), using [Ollama](https://ollama.com) to generate embeddings. Your data stays on your machine.

## Prerequisites

1. .NET 10 SDK
2. Ollama: `winget install --id Ollama.Ollama -e --source winget`
3. The embedding model: `ollama pull nomic-embed-text`

Ollama runs in the background (system tray) and serves an API at `http://localhost:11434`.

## Usage

```powershell
dotnet run -- add sample-data                      # embed all .txt/.md files in a folder (recursive)
dotnet run -- add C:\Notes D:\Docs\plan.md         # add more folders or single files at any time
dotnet run -- search "why does bread rise"         # one-off search
dotnet run -- search                               # interactive search
dotnet run -- search "bread" --min-score 0.3       # change the score cutoff (default 0.5)
dotnet run -- list                                 # indexed files and statistics
dotnet run -- remove C:\Notes                      # remove a file, or everything under a folder
dotnet run -- refresh                              # re-embed changed files, drop deleted ones
dotnet run -- --help                               # all options
```

Adding a path again only re-embeds files whose contents changed. Folders are scanned for `.txt`, `.md` and `.markdown` files (use `--pattern` for others); a file named directly is indexed whatever its extension, unless it looks binary.

In interactive search you can also manage the index without leaving the prompt:

```
search> :add C:\Notes
search> :remove C:\Notes\old.md
search> :refresh
search> :list
```

## How it works

| File | Purpose |
|------|---------|
| `Program.cs` | Command-line entry point (`add`, `search`, `list`, `remove`, `refresh`) |
| `TextChunker.cs` | Splits text into ~1000-char chunks on paragraph and sentence boundaries, with overlap |
| `OllamaEmbeddingClient.cs` | Calls Ollama's `POST /api/embed` with batches of chunks |
| `VectorStore.cs` | Stores files, chunks and vectors in a SQLite database (`embeddings.db`); cosine-similarity search |

1. **Index:** each file → chunks → 768-dimension vectors from `nomic-embed-text` → saved to SQLite, one transaction per file.
2. **Search:** the query → a vector → compared to every chunk with cosine similarity → top N results above the score cutoff.

An `embeddings-index.json` from earlier versions is imported automatically the first time `embeddings.db` is created.

`nomic-embed-text` expects the task prefixes `search_document:` and `search_query:`. The app adds them automatically. If you switch models (`--model mxbai-embed-large`, for example), use `--rebuild`, because vectors from different models can't be compared.
