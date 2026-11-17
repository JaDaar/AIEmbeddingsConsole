using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace AIEmbeddingsConsole;

public sealed class DocumentChunk
{
    public required string Source { get; init; }
    public required int ChunkIndex { get; init; }
    public required string Text { get; init; }
    public required float[] Vector { get; init; }
}

public sealed record SourceInfo(string Path, string Hash, DateTime IndexedUtc, int ChunkCount);

public sealed record SearchResult(DocumentChunk Chunk, float Score);

/// <summary>
/// Stores indexed files, their chunks and embedding vectors in a SQLite database.
/// Search loads the vectors into memory and uses brute-force cosine similarity,
/// which is fast enough for tens of thousands of chunks.
/// </summary>
public sealed class VectorStore : IDisposable
{
    private readonly SqliteConnection _db;

    public string Path { get; }

    public VectorStore(string path)
    {
        Path = System.IO.Path.GetFullPath(path);
        string? dir = System.IO.Path.GetDirectoryName(Path);
        if (dir is not null) Directory.CreateDirectory(dir);

        _db = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = Path }.ToString());
        _db.Open();
        Execute("""
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                path        TEXT PRIMARY KEY COLLATE NOCASE,
                hash        TEXT NOT NULL,
                indexed_utc TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                source_path TEXT NOT NULL COLLATE NOCASE REFERENCES sources(path) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                vector      BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS ix_chunks_source ON chunks(source_path);
            """);
    }

    // ----- metadata -------------------------------------------------------------

    /// <summary>The embedding model the stored vectors came from ("" when empty).</summary>
    public string Model
    {
        get => GetMeta("model") ?? "";
        set => SetMeta("model", value);
    }

    public int Dimensions
    {
        get => int.TryParse(GetMeta("dimensions"), out int d) ? d : 0;
        set => SetMeta("dimensions", value.ToString());
    }

    public DateTime? UpdatedUtc =>
        DateTime.TryParse(GetMeta("updated_utc"), null, System.Globalization.DateTimeStyles.RoundtripKind, out var t)
            ? t : null;

    public int SourceCount => Convert.ToInt32(Scalar("SELECT COUNT(*) FROM sources"));
    public int ChunkCount => Convert.ToInt32(Scalar("SELECT COUNT(*) FROM chunks"));

    // ----- sources --------------------------------------------------------------

    public string? GetSourceHash(string path) =>
        Scalar("SELECT hash FROM sources WHERE path = $p", ("$p", path)) as string;

    public List<SourceInfo> ListSources()
    {
        using var cmd = Command("SELECT path, hash, indexed_utc, chunk_count FROM sources ORDER BY path");
        using var reader = cmd.ExecuteReader();
        var list = new List<SourceInfo>();
        while (reader.Read())
        {
            list.Add(new SourceInfo(
                reader.GetString(0),
                reader.GetString(1),
                DateTime.Parse(reader.GetString(2), null, System.Globalization.DateTimeStyles.RoundtripKind),
                reader.GetInt32(3)));
        }
        return list;
    }

    /// <summary>Replaces all chunks for a file in a single transaction.</summary>
    public void ReplaceSource(string path, string hash, IReadOnlyList<(string Text, float[] Vector)> chunks)
    {
        using var tx = _db.BeginTransaction();
        RemoveSourceCore(path, tx);

        using (var cmd = Command(
            "INSERT INTO sources (path, hash, indexed_utc, chunk_count) VALUES ($p, $h, $t, $c)",
            ("$p", path), ("$h", hash), ("$t", DateTime.UtcNow.ToString("O")), ("$c", chunks.Count)))
        {
            cmd.Transaction = tx;
            cmd.ExecuteNonQuery();
        }

        using (var insert = Command(
            "INSERT INTO chunks (source_path, chunk_index, text, vector) VALUES ($p, $i, $t, $v)"))
        {
            insert.Transaction = tx;
            var pPath = insert.Parameters.AddWithValue("$p", path);
            var pIndex = insert.Parameters.Add("$i", SqliteType.Integer);
            var pText = insert.Parameters.Add("$t", SqliteType.Text);
            var pVector = insert.Parameters.Add("$v", SqliteType.Blob);
            for (int i = 0; i < chunks.Count; i++)
            {
                pIndex.Value = i;
                pText.Value = chunks[i].Text;
                pVector.Value = ToBytes(chunks[i].Vector);
                insert.ExecuteNonQuery();
            }
        }

        Touch(tx);
        tx.Commit();
    }

    /// <summary>Removes one file from the index. Returns false if it wasn't indexed.</summary>
    public bool RemoveSource(string path)
    {
        using var tx = _db.BeginTransaction();
        bool removed = RemoveSourceCore(path, tx);
        if (removed) Touch(tx);
        tx.Commit();
        return removed;
    }

    /// <summary>Deletes everything (used by --rebuild).</summary>
    public void Clear()
    {
        Execute("DELETE FROM chunks; DELETE FROM sources; DELETE FROM meta;");
    }

    // ----- search ---------------------------------------------------------------

    public List<DocumentChunk> LoadChunks()
    {
        using var cmd = Command("SELECT source_path, chunk_index, text, vector FROM chunks");
        using var reader = cmd.ExecuteReader();
        var list = new List<DocumentChunk>();
        while (reader.Read())
        {
            list.Add(new DocumentChunk
            {
                Source = reader.GetString(0),
                ChunkIndex = reader.GetInt32(1),
                Text = reader.GetString(2),
                Vector = FromBytes((byte[])reader[3]),
            });
        }
        return list;
    }

    public static List<SearchResult> Search(IEnumerable<DocumentChunk> chunks, float[] query, int top)
    {
        return chunks
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

    // ----- migration from the old JSON index ------------------------------------

    /// <summary>
    /// Imports an index written by earlier versions (embeddings-index.json) so nothing
    /// has to be re-embedded. Returns the number of files imported.
    /// </summary>
    public int ImportLegacyJson(string jsonPath)
    {
        using var stream = File.OpenRead(jsonPath);
        var legacy = JsonSerializer.Deserialize<LegacyIndex>(stream);
        if (legacy is null || legacy.Chunks.Count == 0) return 0;

        foreach (var group in legacy.Chunks.GroupBy(c => c.Source, StringComparer.OrdinalIgnoreCase))
        {
            string hash = legacy.SourceHashes.FirstOrDefault(
                kv => string.Equals(kv.Key, group.Key, StringComparison.OrdinalIgnoreCase)).Value ?? "";
            var chunks = group.OrderBy(c => c.ChunkIndex).Select(c => (c.Text, c.Vector)).ToList();
            ReplaceSource(group.Key, hash, chunks);
        }
        Model = legacy.Model;
        Dimensions = legacy.Dimensions;
        return SourceCount;
    }

    private sealed class LegacyIndex
    {
        public string Model { get; set; } = "";
        public int Dimensions { get; set; }
        public Dictionary<string, string> SourceHashes { get; set; } = [];
        public List<DocumentChunk> Chunks { get; set; } = [];
    }

    // ----- helpers --------------------------------------------------------------

    private bool RemoveSourceCore(string path, SqliteTransaction tx)
    {
        // chunks are removed by ON DELETE CASCADE
        using var cmd = Command("DELETE FROM sources WHERE path = $p", ("$p", path));
        cmd.Transaction = tx;
        return cmd.ExecuteNonQuery() > 0;
    }

    private void Touch(SqliteTransaction tx)
    {
        using var cmd = Command(
            "INSERT INTO meta (key, value) VALUES ('updated_utc', $v) " +
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("$v", DateTime.UtcNow.ToString("O")));
        cmd.Transaction = tx;
        cmd.ExecuteNonQuery();
    }

    private string? GetMeta(string key) =>
        Scalar("SELECT value FROM meta WHERE key = $k", ("$k", key)) as string;

    private void SetMeta(string key, string value)
    {
        using var cmd = Command(
            "INSERT INTO meta (key, value) VALUES ($k, $v) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("$k", key), ("$v", value));
        cmd.ExecuteNonQuery();
    }

    private static byte[] ToBytes(float[] vector) => MemoryMarshal.AsBytes(vector.AsSpan()).ToArray();

    private static float[] FromBytes(byte[] bytes) => MemoryMarshal.Cast<byte, float>(bytes).ToArray();

    private SqliteCommand Command(string sql, params (string Name, object Value)[] parameters)
    {
        var cmd = _db.CreateCommand();
        cmd.CommandText = sql;
        foreach (var (name, value) in parameters) cmd.Parameters.AddWithValue(name, value);
        return cmd;
    }

    private object? Scalar(string sql, params (string Name, object Value)[] parameters)
    {
        using var cmd = Command(sql, parameters);
        return cmd.ExecuteScalar();
    }

    private void Execute(string sql)
    {
        using var cmd = Command(sql);
        cmd.ExecuteNonQuery();
    }

    public void Dispose() => _db.Dispose();
}
