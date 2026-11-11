using System.Text.Json;

namespace AIEmbeddingsConsole;

public sealed class DocumentChunk
{
    public required string Source { get; init; }
    public required int ChunkIndex { get; init; }
    public required string Text { get; init; }
    public required float[] Vector { get; init; }
}

public sealed class VectorIndex
{
    public string Model { get; set; } = "";
    public int Dimensions { get; set; }
    public DateTime UpdatedUtc { get; set; }

    /// <summary>Source file path -> SHA-256 of its contents, used to skip unchanged files.</summary>
    public Dictionary<string, string> SourceHashes { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public List<DocumentChunk> Chunks { get; set; } = [];
}

public sealed record SearchResult(DocumentChunk Chunk, float Score);

/// <summary>
/// A simple in-memory vector store persisted to a JSON file.
/// Brute-force cosine similarity is fast enough for tens of thousands of chunks.
/// </summary>
public static class VectorStore
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = false };

    public static VectorIndex Load(string path)
    {
        if (!File.Exists(path)) return new VectorIndex();
        using var stream = File.OpenRead(path);
        var index = JsonSerializer.Deserialize<VectorIndex>(stream, JsonOptions) ?? new VectorIndex();
        index.SourceHashes = new Dictionary<string, string>(index.SourceHashes, StringComparer.OrdinalIgnoreCase);
        return index;
    }

    public static void Save(VectorIndex index, string path)
    {
        index.UpdatedUtc = DateTime.UtcNow;
        string? dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (dir is not null) Directory.CreateDirectory(dir);

        // Write to a temp file first so a crash never leaves a half-written index.
        string temp = path + ".tmp";
        using (var stream = File.Create(temp))
        {
            JsonSerializer.Serialize(stream, index, JsonOptions);
        }
        File.Move(temp, path, overwrite: true);
    }

    public static List<SearchResult> Search(VectorIndex index, float[] query, int top)
    {
        return index.Chunks
            .Select(c => new SearchResult(c, CosineSimilarity(query, c.Vector)))
            .OrderByDescending(r => r.Score)
            .Take(top)
            .ToList();
    }

    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector length mismatch ({a.Length} vs {b.Length}).");

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return normA == 0 || normB == 0 ? 0 : (float)(dot / (Math.Sqrt(normA) * Math.Sqrt(normB)));
    }
}
