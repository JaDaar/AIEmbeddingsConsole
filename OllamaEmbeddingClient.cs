using System.Net.Http.Json;
using System.Text.Json;

namespace AIEmbeddingsConsole;

/// <summary>
/// Minimal client for Ollama's local REST API (http://localhost:11434 by default).
/// Uses POST /api/embed, which accepts a batch of inputs and returns one vector per input.
/// </summary>
public sealed class OllamaEmbeddingClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    private readonly HttpClient _http;

    public string Model { get; }
    public Uri BaseAddress => _http.BaseAddress!;

    public OllamaEmbeddingClient(Uri baseAddress, string model)
    {
        _http = new HttpClient
        {
            BaseAddress = baseAddress,
            // The first call loads the model into memory, which can take a while.
            Timeout = TimeSpan.FromMinutes(5),
        };
        Model = model;
    }

    /// <summary>
    /// Checks that the Ollama server is reachable and the embedding model has been pulled.
    /// Throws with a friendly message describing how to fix the problem.
    /// </summary>
    public async Task EnsureReadyAsync(CancellationToken ct = default)
    {
        TagsResponse? tags;
        try
        {
            tags = await _http.GetFromJsonAsync<TagsResponse>("api/tags", JsonOptions, ct);
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException(
                $"Could not reach Ollama at {BaseAddress}. Is it running? " +
                "Start it from the Start menu or run 'ollama serve'.", ex);
        }

        var installed = tags?.Models?.Select(m => m.Name).ToList() ?? [];
        bool found = installed.Any(name =>
            string.Equals(name, Model, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(name, Model + ":latest", StringComparison.OrdinalIgnoreCase));

        if (!found)
        {
            throw new InvalidOperationException(
                $"Model '{Model}' is not installed in Ollama. Run: ollama pull {Model}" +
                (installed.Count > 0 ? $"{Environment.NewLine}Installed models: {string.Join(", ", installed)}" : ""));
        }
    }

    public async Task<float[]> EmbedAsync(string input, CancellationToken ct = default)
        => (await EmbedAsync([input], ct))[0];

    public async Task<float[][]> EmbedAsync(IReadOnlyList<string> inputs, CancellationToken ct = default)
    {
        if (inputs.Count == 0) return [];

        var request = new EmbedRequest(Model, inputs, Truncate: true);
        using var response = await _http.PostAsJsonAsync("api/embed", request, JsonOptions, ct);

        if (!response.IsSuccessStatusCode)
        {
            string body = await response.Content.ReadAsStringAsync(ct);
            throw new InvalidOperationException($"Ollama returned {(int)response.StatusCode}: {body}");
        }

        var result = await response.Content.ReadFromJsonAsync<EmbedResponse>(JsonOptions, ct)
            ?? throw new InvalidOperationException("Ollama returned an empty response.");

        if (result.Embeddings is null || result.Embeddings.Length != inputs.Count)
        {
            throw new InvalidOperationException(
                $"Expected {inputs.Count} embeddings but received {result.Embeddings?.Length ?? 0}.");
        }

        return result.Embeddings;
    }

    public void Dispose() => _http.Dispose();

    private sealed record EmbedRequest(string Model, IReadOnlyList<string> Input, bool Truncate);
    private sealed record EmbedResponse(string? Model, float[][]? Embeddings);
    private sealed record TagsResponse(List<ModelInfo>? Models);
    private sealed record ModelInfo(string Name);
}
