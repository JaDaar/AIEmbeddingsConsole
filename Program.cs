using System.Diagnostics;
using System.Security.Cryptography;
using AIEmbeddingsConsole;

const string DefaultModel = "nomic-embed-text";
const string DefaultDbPath = "embeddings.db";
const string LegacyIndexPath = "embeddings-index.json";
const int EmbedBatchSize = 16;
string[] TextExtensions = [".txt", ".md", ".markdown"];

var (positional, options) = ParseArgs(args);

if (positional.Count == 0 || options.ContainsKey("help"))
{
    PrintUsage();
    return positional.Count == 0 && !options.ContainsKey("help") ? 1 : 0;
}

string command = positional[0].ToLowerInvariant();
string dbPath = options.GetValueOrDefault("db") ?? DefaultDbPath;
string? modelOption = options.GetValueOrDefault("model");
var ollamaHost = ResolveOllamaHost(Environment.GetEnvironmentVariable("OLLAMA_HOST"));
int chunkSize = 1000, overlap = 150;

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

try
{
    chunkSize = GetInt("chunk-size", 1000);
    overlap = GetInt("overlap", 150);
    return command switch
    {
        "add" or "index" => await RunAddAsync(cts.Token),
        "search" => await RunSearchAsync(cts.Token),
        "list" or "info" => RunList(),
        "remove" => RunRemove(),
        "refresh" => await RunRefreshAsync(cts.Token),
        _ => Fail($"Unknown command '{command}'. Run with --help for usage."),
    };
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("Cancelled.");
    return 130;
}
catch (Exception ex) when (ex is InvalidOperationException or IOException or ArgumentException
                               or UnauthorizedAccessException or Microsoft.Data.Sqlite.SqliteException)
{
    return Fail(ex.Message);
}

// ---------------------------------------------------------------------------
// add <file-or-folder>...
// ---------------------------------------------------------------------------
async Task<int> RunAddAsync(CancellationToken ct)
{
    if (positional.Count < 2) return Fail("Usage: add <file-or-folder> [more paths...] [options]");

    using var store = OpenStore();
    if (options.ContainsKey("rebuild"))
    {
        store.Clear();
        Console.WriteLine("Cleared the existing index (--rebuild).");
    }

    using var client = await CreateClientAsync(store, forIndexing: true, ct);
    var stopwatch = Stopwatch.StartNew();
    var stats = await AddPathsAsync(store, client, positional.Skip(1), ct);

    Console.WriteLine();
    Console.WriteLine($"Embedded {stats.Embedded} file(s) ({stats.Chunks} chunks), skipped {stats.Unchanged} unchanged" +
                      (stats.Skipped > 0 ? $", ignored {stats.Skipped} non-text" : "") +
                      $", in {stopwatch.Elapsed.TotalSeconds:F1}s.");
    PrintSummary(store);
    return stats.Missing > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// refresh   (re-embed changed files, drop deleted ones)
// ---------------------------------------------------------------------------
async Task<int> RunRefreshAsync(CancellationToken ct)
{
    using var store = OpenStore();
    if (store.SourceCount == 0) return Fail($"Index '{store.Path}' is empty. Run 'add <file-or-folder>' first.");

    using var client = await CreateClientAsync(store, forIndexing: true, ct);
    var stats = await RefreshAsync(store, client, ct);
    Console.WriteLine();
    Console.WriteLine($"Re-embedded {stats.Embedded} changed file(s), removed {stats.Removed} missing, " +
                      $"{stats.Unchanged} unchanged.");
    PrintSummary(store);
    return 0;
}

// ---------------------------------------------------------------------------
// remove <file-or-folder>...
// ---------------------------------------------------------------------------
int RunRemove()
{
    if (positional.Count < 2) return Fail("Usage: remove <file-or-folder> [more paths...]");
    using var store = OpenStore();
    int removed = positional.Skip(1).Sum(p => RemovePath(store, p));
    Console.WriteLine($"Removed {removed} file(s) from the index.");
    PrintSummary(store);
    return removed > 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------
int RunList()
{
    if (!File.Exists(dbPath) && !File.Exists(LegacyIndexPath)) return Fail($"Index '{Path.GetFullPath(dbPath)}' not found. Run 'add <file-or-folder>' first.");
    using var store = OpenStore();
    Console.WriteLine($"Index:      {store.Path}");
    Console.WriteLine($"Model:      {store.Model}");
    Console.WriteLine($"Dimensions: {store.Dimensions}");
    Console.WriteLine($"Files:      {store.SourceCount}");
    Console.WriteLine($"Chunks:     {store.ChunkCount}");
    if (store.UpdatedUtc is { } updated) Console.WriteLine($"Updated:    {updated.ToLocalTime():g}");
    PrintSources(store);
    return 0;
}

// ---------------------------------------------------------------------------
// search [query]   (no query = interactive mode)
// ---------------------------------------------------------------------------
async Task<int> RunSearchAsync(CancellationToken ct)
{
    using var store = OpenStore();
    if (store.ChunkCount == 0)
        return Fail($"Index '{store.Path}' is empty. Run 'add <file-or-folder>' first.");

    int top = GetInt("top", 5);
    float minScore = GetFloat("min-score", 0.5f);
    using var client = await CreateClientAsync(store, forIndexing: false, ct);
    var chunks = store.LoadChunks();

    string? query = positional.Count > 1 ? string.Join(' ', positional.Skip(1)) : null;
    if (query is not null)
    {
        await SearchOnceAsync(client, chunks, query, top, minScore, ct);
        return 0;
    }

    Console.WriteLine($"Loaded {chunks.Count} chunks from {store.SourceCount} file(s). " +
                      "Type a question, ':help' for commands, or a blank line to quit.");
    while (true)
    {
        Console.Write("\nsearch> ");
        string? line = Console.ReadLine()?.Trim().TrimStart('﻿'); // piped input may start with a BOM
        if (string.IsNullOrEmpty(line) || line.Equals("exit", StringComparison.OrdinalIgnoreCase))
            return 0;

        if (!line.StartsWith(':'))
        {
            await SearchOnceAsync(client, chunks, line, top, minScore, ct);
            continue;
        }

        // Interactive commands: the rest of the line is one path, so spaces are fine.
        string verb = line.Split(' ', 2)[0].ToLowerInvariant();
        string arg = line.Length > verb.Length ? line[verb.Length..].Trim().Trim('"') : "";
        try
        {
            switch (verb)
            {
                case ":add" when arg.Length > 0:
                    var stats = await AddPathsAsync(store, client, [arg], ct);
                    Console.WriteLine($"Embedded {stats.Embedded} file(s), skipped {stats.Unchanged} unchanged.");
                    break;
                case ":remove" when arg.Length > 0:
                    Console.WriteLine($"Removed {RemovePath(store, arg)} file(s).");
                    break;
                case ":refresh":
                    var r = await RefreshAsync(store, client, ct);
                    Console.WriteLine($"Re-embedded {r.Embedded} changed, removed {r.Removed} missing.");
                    break;
                case ":list":
                    PrintSources(store);
                    continue;
                default:
                    Console.WriteLine("""
                        :add <file-or-folder>     index more files
                        :remove <file-or-folder>  remove files from the index
                        :refresh                  re-embed changed files, drop deleted ones
                        :list                     show indexed files
                        """);
                    continue;
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
        chunks = store.LoadChunks();
        Console.WriteLine($"Index now has {chunks.Count} chunks from {store.SourceCount} file(s).");
    }
}

static async Task SearchOnceAsync(OllamaEmbeddingClient client, List<DocumentChunk> chunks, string query,
                                  int top, float minScore, CancellationToken ct)
{
    if (chunks.Count == 0)
    {
        Console.WriteLine("The index is empty. Add files with ':add <file-or-folder>'.");
        return;
    }

    var queryVector = await client.EmbedAsync(QueryPrefix(client.Model) + query, ct);
    var results = VectorStore.Search(chunks, queryVector, top)
        .Where(r => r.Score >= minScore)
        .ToList();

    if (results.Count == 0)
    {
        Console.WriteLine($"No results with a score of at least {minScore:F3}.");
        return;
    }

    for (int i = 0; i < results.Count; i++)
    {
        var (chunk, score) = (results[i].Chunk, results[i].Score);
        string snippet = chunk.Text.Length > 300 ? chunk.Text[..300] + "..." : chunk.Text;

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"\n#{i + 1}  score {score:F3}  {Display(chunk.Source)} [chunk {chunk.ChunkIndex}]");
        Console.ResetColor();
        Console.WriteLine(snippet);
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------
async Task<IndexStats> AddPathsAsync(VectorStore store, OllamaEmbeddingClient client,
                                     IEnumerable<string> paths, CancellationToken ct)
{
    var stats = new IndexStats();
    string? pattern = options.GetValueOrDefault("pattern");

    foreach (string raw in paths)
    {
        string target = Path.GetFullPath(raw);
        List<string> files;

        if (File.Exists(target))
        {
            files = [target]; // a file named explicitly is accepted whatever its extension
        }
        else if (Directory.Exists(target))
        {
            files = (pattern is null
                    ? Directory.EnumerateFiles(target, "*", SearchOption.AllDirectories)
                        .Where(f => TextExtensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
                    : Directory.EnumerateFiles(target, pattern, SearchOption.AllDirectories))
                .Order(StringComparer.OrdinalIgnoreCase)
                .ToList();

            // Drop entries for files that were deleted from this folder.
            string prefix = FolderPrefix(target);
            foreach (var source in store.ListSources().Where(s => s.Path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
                                                                   && !File.Exists(s.Path)))
            {
                store.RemoveSource(source.Path);
                Console.WriteLine($"  {Display(source.Path)}: removed (file deleted)");
                stats.Removed++;
            }
        }
        else
        {
            Console.Error.WriteLine($"  Path not found: {target}");
            stats.Missing++;
            continue;
        }

        Console.WriteLine($"Indexing {files.Count} file(s) from {Display(target)} with '{client.Model}'");
        foreach (string file in files)
            await IndexFileAsync(store, client, file, stats, ct);
    }
    return stats;
}

async Task<IndexStats> RefreshAsync(VectorStore store, OllamaEmbeddingClient client, CancellationToken ct)
{
    var stats = new IndexStats();
    foreach (var source in store.ListSources())
    {
        if (!File.Exists(source.Path))
        {
            store.RemoveSource(source.Path);
            Console.WriteLine($"  {Display(source.Path)}: removed (file deleted)");
            stats.Removed++;
        }
        else
        {
            await IndexFileAsync(store, client, source.Path, stats, ct);
        }
    }
    return stats;
}

async Task IndexFileAsync(VectorStore store, OllamaEmbeddingClient client, string file,
                          IndexStats stats, CancellationToken ct)
{
    ct.ThrowIfCancellationRequested();
    string display = Display(file);

    if (await LooksBinaryAsync(file, ct))
    {
        Console.WriteLine($"  {display}: skipped (not a text file)");
        stats.Skipped++;
        return;
    }

    string text = await File.ReadAllTextAsync(file, ct);
    string hash = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(text)));
    if (store.GetSourceHash(file) == hash)
    {
        stats.Unchanged++;
        return;
    }

    var chunks = TextChunker.Chunk(text, chunkSize, overlap);
    Console.Write($"  {display}: {chunks.Count} chunk(s) ");

    var embedded = new List<(string Text, float[] Vector)>(chunks.Count);
    for (int start = 0; start < chunks.Count; start += EmbedBatchSize)
    {
        var batch = chunks.Skip(start).Take(EmbedBatchSize).ToList();
        var vectors = await client.EmbedAsync(batch.Select(c => DocumentPrefix(client.Model) + c).ToList(), ct);
        for (int i = 0; i < batch.Count; i++) embedded.Add((batch[i], vectors[i]));
        Console.Write('.');
    }

    // Each file is written in its own transaction, so progress survives Ctrl+C.
    store.ReplaceSource(file, hash, embedded);
    if (embedded.Count > 0)
    {
        store.Model = client.Model;
        store.Dimensions = embedded[0].Vector.Length;
    }
    Console.WriteLine(" done");
    stats.Embedded++;
    stats.Chunks += chunks.Count;
}

int RemovePath(VectorStore store, string raw)
{
    string target = Path.GetFullPath(raw);
    if (store.RemoveSource(target))
    {
        Console.WriteLine($"  {Display(target)}: removed");
        return 1;
    }

    // Not an indexed file: treat it as a folder and remove everything under it.
    // This works even if the folder no longer exists on disk.
    string prefix = FolderPrefix(target);
    int count = 0;
    foreach (var source in store.ListSources().Where(s => s.Path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
    {
        store.RemoveSource(source.Path);
        Console.WriteLine($"  {Display(source.Path)}: removed");
        count++;
    }
    if (count == 0) Console.WriteLine($"  Nothing indexed at {target}");
    return count;
}

// A file is treated as binary if its first 8 KB contain a NUL byte.
static async Task<bool> LooksBinaryAsync(string file, CancellationToken ct)
{
    var buffer = new byte[8192];
    await using var stream = File.OpenRead(file);
    int read = await stream.ReadAtLeastAsync(buffer, buffer.Length, throwOnEndOfStream: false, ct);
    return Array.IndexOf(buffer, (byte)0, 0, read) >= 0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

VectorStore OpenStore()
{
    bool isNew = !File.Exists(dbPath);
    var store = new VectorStore(dbPath);

    // One-time import of the JSON index used by earlier versions.
    if (isNew && !options.ContainsKey("rebuild") && File.Exists(LegacyIndexPath))
    {
        int imported = store.ImportLegacyJson(LegacyIndexPath);
        if (imported > 0)
            Console.WriteLine($"Imported {imported} file(s) from {LegacyIndexPath} into {store.Path}. " +
                              $"You can delete {LegacyIndexPath}.");
    }
    return store;
}

async Task<OllamaEmbeddingClient> CreateClientAsync(VectorStore store, bool forIndexing, CancellationToken ct)
{
    // Use the index's model unless --model says otherwise; vectors from different models can't be mixed.
    string model = modelOption ?? (store.Model.Length > 0 ? store.Model : DefaultModel);
    if (store.ChunkCount > 0 && !string.Equals(store.Model, model, StringComparison.OrdinalIgnoreCase))
    {
        throw new InvalidOperationException(
            $"Index '{store.Path}' was built with model '{store.Model}'. " +
            (forIndexing ? $"Use --rebuild to re-embed everything with '{model}'." : $"Use --model {store.Model}."));
    }

    var client = new OllamaEmbeddingClient(ollamaHost, model);
    await client.EnsureReadyAsync(ct);
    return client;
}

void PrintSummary(VectorStore store) =>
    Console.WriteLine($"Index: {store.Path} ({store.SourceCount} files, {store.ChunkCount} chunks)");

void PrintSources(VectorStore store)
{
    foreach (var source in store.ListSources())
        Console.WriteLine($"  {Display(source.Path)} ({source.ChunkCount} chunks, {source.IndexedUtc.ToLocalTime():g})");
}

// Paths under the current folder are shown relative; anything else in full.
static string Display(string path)
{
    string relative = Path.GetRelativePath(Environment.CurrentDirectory, path);
    return relative.StartsWith("..") || Path.IsPathRooted(relative) ? path : relative;
}

static string FolderPrefix(string folder) =>
    folder.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;

// nomic-embed-text was trained with task prefixes; using them noticeably improves retrieval.
static string DocumentPrefix(string model) =>
    model.StartsWith("nomic-embed-text", StringComparison.OrdinalIgnoreCase) ? "search_document: " : "";

static string QueryPrefix(string model) =>
    model.StartsWith("nomic-embed-text", StringComparison.OrdinalIgnoreCase) ? "search_query: " : "";

int GetInt(string name, int defaultValue)
{
    if (!options.TryGetValue(name, out string? raw) || raw is null) return defaultValue;
    return int.TryParse(raw, out int value) && value >= 0
        ? value
        : throw new ArgumentException($"--{name} must be a non-negative integer (got '{raw}').");
}

float GetFloat(string name, float defaultValue)
{
    if (!options.TryGetValue(name, out string? raw) || raw is null) return defaultValue;
    return float.TryParse(raw, System.Globalization.NumberStyles.Float,
                          System.Globalization.CultureInfo.InvariantCulture, out float value)
        ? value
        : throw new ArgumentException($"--{name} must be a number (got '{raw}').");
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
          AIEmbeddingsConsole add <file-or-folder>... [options]     Embed files into the index ('index' also works)
          AIEmbeddingsConsole search [query] [options]              Search (interactive if no query)
          AIEmbeddingsConsole list [options]                        Show indexed files ('info' also works)
          AIEmbeddingsConsole remove <file-or-folder>... [options]  Remove files from the index
          AIEmbeddingsConsole refresh [options]                     Re-embed changed files, drop deleted ones

        Folders are searched recursively for .txt, .md and .markdown files.
        A file named explicitly is indexed whatever its extension, unless it looks binary.

        Interactive search commands:
          :add <path>  :remove <path>  :refresh  :list  :help

        Options:
          --db <path>          SQLite index file (default: embeddings.db)
          --model <name>       Ollama embedding model (default: the index's model, else nomic-embed-text)
          --pattern <glob>     Only index folder files matching this pattern (e.g. *.log)
          --chunk-size <n>     Max characters per chunk (default: 1000)
          --overlap <n>        Characters of overlap between chunks (default: 150)
          --rebuild            Discard the existing index and re-embed everything
          --top <n>            Number of search results (default: 5)
          --min-score <x>      Hide results scoring below this (default: 0.5)

        Environment:
          OLLAMA_HOST          Ollama server address (default: http://localhost:11434)
        """);
}

sealed class IndexStats
{
    public int Embedded, Unchanged, Skipped, Removed, Missing, Chunks;
}
