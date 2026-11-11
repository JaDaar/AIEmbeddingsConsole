using System.Diagnostics;
using System.Security.Cryptography;
using AIEmbeddingsConsole;

const string DefaultModel = "nomic-embed-text";
const string DefaultIndexPath = "embeddings-index.json";
const int EmbedBatchSize = 16;

var (positional, options) = ParseArgs(args);

if (positional.Count == 0 || options.ContainsKey("help"))
{
    PrintUsage();
    return positional.Count == 0 && !options.ContainsKey("help") ? 1 : 0;
}

string command = positional[0].ToLowerInvariant();
string indexPath = options.GetValueOrDefault("index") ?? DefaultIndexPath;
string model = options.GetValueOrDefault("model") ?? DefaultModel;
var ollamaHost = ResolveOllamaHost(Environment.GetEnvironmentVariable("OLLAMA_HOST"));

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

try
{
    return command switch
    {
        "index" => await RunIndexAsync(cts.Token),
        "search" => await RunSearchAsync(cts.Token),
        "info" => RunInfo(),
        _ => Fail($"Unknown command '{command}'. Run with --help for usage."),
    };
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("Cancelled.");
    return 130;
}
catch (Exception ex) when (ex is InvalidOperationException or IOException or ArgumentException)
{
    return Fail(ex.Message);
}

// ---------------------------------------------------------------------------
// index <file-or-folder>
// ---------------------------------------------------------------------------
async Task<int> RunIndexAsync(CancellationToken ct)
{
    if (positional.Count < 2) return Fail("Usage: index <file-or-folder> [options]");

    string target = Path.GetFullPath(positional[1]);
    string pattern = options.GetValueOrDefault("pattern") ?? "*.txt";
    int chunkSize = GetInt("chunk-size", 1000);
    int overlap = GetInt("overlap", 150);

    List<string> files;
    if (File.Exists(target)) files = [target];
    else if (Directory.Exists(target))
        files = Directory.EnumerateFiles(target, pattern, SearchOption.AllDirectories).Order().ToList();
    else return Fail($"Path not found: {target}");

    var index = options.ContainsKey("rebuild") ? new VectorIndex() : VectorStore.Load(indexPath);
    if (index.Chunks.Count > 0 && !string.Equals(index.Model, model, StringComparison.OrdinalIgnoreCase))
    {
        return Fail($"Index '{indexPath}' was built with model '{index.Model}'. " +
                    $"Use --model {index.Model}, or --rebuild to re-embed everything with '{model}'.");
    }
    index.Model = model;

    using var client = new OllamaEmbeddingClient(ollamaHost, model);
    await client.EnsureReadyAsync(ct);

    // Drop entries for files that were deleted from the folder being indexed.
    if (Directory.Exists(target))
    {
        string prefix = target.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        var removed = index.SourceHashes.Keys
            .Where(s => s.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) && !File.Exists(s))
            .ToList();
        foreach (string source in removed) RemoveSource(index, source);
        if (removed.Count > 0) Console.WriteLine($"Removed {removed.Count} deleted file(s) from the index.");
    }

    Console.WriteLine($"Indexing {files.Count} file(s) with '{model}' via {client.BaseAddress}");
    var stopwatch = Stopwatch.StartNew();
    int indexedFiles = 0, skippedFiles = 0, newChunks = 0;

    foreach (string file in files)
    {
        ct.ThrowIfCancellationRequested();
        string display = Path.GetRelativePath(Environment.CurrentDirectory, file);
        string text = await File.ReadAllTextAsync(file, ct);
        string hash = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(text)));

        if (index.SourceHashes.TryGetValue(file, out string? existing) && existing == hash)
        {
            skippedFiles++;
            continue;
        }

        RemoveSource(index, file);
        var chunks = TextChunker.Chunk(text, chunkSize, overlap);
        Console.Write($"  {display}: {chunks.Count} chunk(s) ");

        for (int start = 0; start < chunks.Count; start += EmbedBatchSize)
        {
            var batch = chunks.Skip(start).Take(EmbedBatchSize).ToList();
            var vectors = await client.EmbedAsync(batch.Select(c => DocumentPrefix(model) + c).ToList(), ct);

            for (int i = 0; i < batch.Count; i++)
            {
                index.Chunks.Add(new DocumentChunk
                {
                    Source = file,
                    ChunkIndex = start + i,
                    Text = batch[i],
                    Vector = vectors[i],
                });
            }
            index.Dimensions = vectors[0].Length;
            Console.Write('.');
        }

        index.SourceHashes[file] = hash;
        VectorStore.Save(index, indexPath); // save per file so progress survives Ctrl+C
        Console.WriteLine(" done");
        indexedFiles++;
        newChunks += chunks.Count;
    }

    VectorStore.Save(index, indexPath);
    Console.WriteLine();
    Console.WriteLine($"Embedded {indexedFiles} file(s) ({newChunks} chunks), skipped {skippedFiles} unchanged, " +
                      $"in {stopwatch.Elapsed.TotalSeconds:F1}s.");
    Console.WriteLine($"Index: {Path.GetFullPath(indexPath)} ({index.Chunks.Count} chunks, {index.Dimensions} dims)");
    return 0;
}

// ---------------------------------------------------------------------------
// search [query]   (no query = interactive mode)
// ---------------------------------------------------------------------------
async Task<int> RunSearchAsync(CancellationToken ct)
{
    var index = VectorStore.Load(indexPath);
    if (index.Chunks.Count == 0)
        return Fail($"Index '{indexPath}' is empty or missing. Run 'index <folder>' first.");

    int top = GetInt("top", 5);
    using var client = new OllamaEmbeddingClient(ollamaHost, index.Model);
    await client.EnsureReadyAsync(ct);

    string? query = positional.Count > 1 ? string.Join(' ', positional.Skip(1)) : null;
    if (query is not null)
    {
        await SearchOnceAsync(client, index, query, top, ct);
        return 0;
    }

    Console.WriteLine($"Loaded {index.Chunks.Count} chunks from {index.SourceHashes.Count} file(s). " +
                      "Type a question (blank line or 'exit' to quit).");
    while (true)
    {
        Console.Write("\nsearch> ");
        string? line = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(line) || line.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
            return 0;
        await SearchOnceAsync(client, index, line.Trim(), top, ct);
    }
}

static async Task SearchOnceAsync(OllamaEmbeddingClient client, VectorIndex index, string query, int top, CancellationToken ct)
{
    var queryVector = await client.EmbedAsync(QueryPrefix(client.Model) + query, ct);
    var results = VectorStore.Search(index, queryVector, top);

    for (int i = 0; i < results.Count; i++)
    {
        var (chunk, score) = (results[i].Chunk, results[i].Score);
        string source = Path.GetRelativePath(Environment.CurrentDirectory, chunk.Source);
        string snippet = chunk.Text.Length > 300 ? chunk.Text[..300] + "..." : chunk.Text;

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"\n#{i + 1}  score {score:F3}  {source} [chunk {chunk.ChunkIndex}]");
        Console.ResetColor();
        Console.WriteLine(snippet);
    }
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------
int RunInfo()
{
    if (!File.Exists(indexPath)) return Fail($"Index '{indexPath}' not found.");
    var index = VectorStore.Load(indexPath);
    Console.WriteLine($"Index:      {Path.GetFullPath(indexPath)}");
    Console.WriteLine($"Model:      {index.Model}");
    Console.WriteLine($"Dimensions: {index.Dimensions}");
    Console.WriteLine($"Files:      {index.SourceHashes.Count}");
    Console.WriteLine($"Chunks:     {index.Chunks.Count}");
    Console.WriteLine($"Updated:    {index.UpdatedUtc.ToLocalTime():g}");
    foreach (var group in index.Chunks.GroupBy(c => c.Source).OrderBy(g => g.Key))
        Console.WriteLine($"  {Path.GetRelativePath(Environment.CurrentDirectory, group.Key)} ({group.Count()} chunks)");
    return 0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// nomic-embed-text was trained with task prefixes; using them noticeably improves retrieval.
static string DocumentPrefix(string model) =>
    model.StartsWith("nomic-embed-text", StringComparison.OrdinalIgnoreCase) ? "search_document: " : "";

static string QueryPrefix(string model) =>
    model.StartsWith("nomic-embed-text", StringComparison.OrdinalIgnoreCase) ? "search_query: " : "";

static void RemoveSource(VectorIndex index, string source)
{
    index.Chunks.RemoveAll(c => string.Equals(c.Source, source, StringComparison.OrdinalIgnoreCase));
    index.SourceHashes.Remove(source);
}

int GetInt(string name, int defaultValue)
{
    if (!options.TryGetValue(name, out string? raw) || raw is null) return defaultValue;
    return int.TryParse(raw, out int value) && value >= 0
        ? value
        : throw new ArgumentException($"--{name} must be a non-negative integer (got '{raw}').");
}

static Uri ResolveOllamaHost(string? env)
{
    // OLLAMA_HOST may be "", "0.0.0.0", "127.0.0.1:11434", or a full URL.
    string host = string.IsNullOrWhiteSpace(env) ? "localhost:11434" : env.Trim();
    if (!host.Contains("://")) host = "http://" + host;

    string authority = host[(host.IndexOf("://") + 3)..].Split('/')[0];
    var builder = new UriBuilder(host) { Path = "/" };
    if (!authority.Contains(':')) builder.Port = 11434;
    if (builder.Host == "0.0.0.0") builder.Host = "localhost";
    return builder.Uri;
}

static (List<string> Positional, Dictionary<string, string?> Options) ParseArgs(string[] args)
{
    var positional = new List<string>();
    var opts = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
    for (int i = 0; i < args.Length; i++)
    {
        if (args[i] is "-h" or "/?") { opts["help"] = null; continue; }
        if (args[i].StartsWith("--"))
        {
            string name = args[i][2..];
            bool isFlag = name is "rebuild" or "help";
            opts[name] = !isFlag && i + 1 < args.Length ? args[++i] : null;
        }
        else positional.Add(args[i]);
    }
    return (positional, opts);
}

static int Fail(string message)
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.Error.WriteLine($"Error: {message}");
    Console.ResetColor();
    return 1;
}

static void PrintUsage()
{
    Console.WriteLine("""
        AIEmbeddingsConsole - local semantic search over text files using Ollama

        Usage:
          AIEmbeddingsConsole index <file-or-folder> [options]   Embed .txt files into the index
          AIEmbeddingsConsole search [query] [options]           Search (interactive if no query)
          AIEmbeddingsConsole info [options]                     Show index statistics

        Options:
          --index <path>       Index file (default: embeddings-index.json)
          --model <name>       Ollama embedding model (default: nomic-embed-text)
          --pattern <glob>     File pattern when indexing a folder (default: *.txt)
          --chunk-size <n>     Max characters per chunk (default: 1000)
          --overlap <n>        Characters of overlap between chunks (default: 150)
          --rebuild            Discard the existing index and re-embed everything
          --top <n>            Number of search results (default: 5)

        Environment:
          OLLAMA_HOST          Ollama server address (default: http://localhost:11434)
        """);
}
