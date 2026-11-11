using System.Text;
using System.Text.RegularExpressions;

namespace AIEmbeddingsConsole;

/// <summary>
/// Splits text into chunks of roughly <c>maxChars</c> characters, preferring paragraph
/// and sentence boundaries, with a small overlap so context isn't lost at chunk edges.
/// </summary>
public static partial class TextChunker
{
    [GeneratedRegex(@"\n\s*\n")]
    private static partial Regex ParagraphBreak();

    [GeneratedRegex(@"(?<=[.!?])\s+")]
    private static partial Regex SentenceBreak();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();

    public static List<string> Chunk(string text, int maxChars = 1000, int overlapChars = 150)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxChars, 50);
        ArgumentOutOfRangeException.ThrowIfNegative(overlapChars);
        overlapChars = Math.Min(overlapChars, maxChars / 2);

        var units = SplitIntoUnits(text.Replace("\r\n", "\n"), maxChars);
        var chunks = new List<string>();
        var current = new List<string>();
        int currentLength = 0;

        foreach (string unit in units)
        {
            if (current.Count > 0 && currentLength + 1 + unit.Length > maxChars)
            {
                chunks.Add(string.Join(" ", current));

                // Carry trailing units forward as overlap.
                var overlap = new List<string>();
                int overlapLength = 0;
                for (int i = current.Count - 1; i >= 0; i--)
                {
                    if (overlapLength + current[i].Length > overlapChars) break;
                    overlap.Insert(0, current[i]);
                    overlapLength += current[i].Length + 1;
                }

                // Don't let the overlap push the next chunk over the limit.
                while (overlap.Count > 0 && overlapLength + unit.Length > maxChars)
                {
                    overlapLength -= overlap[0].Length + 1;
                    overlap.RemoveAt(0);
                }

                current = overlap;
                currentLength = Math.Max(0, overlapLength - 1);
            }

            currentLength += (current.Count > 0 ? 1 : 0) + unit.Length;
            current.Add(unit);
        }

        if (current.Count > 0)
        {
            chunks.Add(string.Join(" ", current));
        }

        return chunks;
    }

    /// <summary>Breaks text into paragraphs, then sentences, then words, so no unit exceeds maxChars.</summary>
    private static IEnumerable<string> SplitIntoUnits(string text, int maxChars)
    {
        foreach (string rawParagraph in ParagraphBreak().Split(text))
        {
            string paragraph = Whitespace().Replace(rawParagraph, " ").Trim();
            if (paragraph.Length == 0) continue;

            if (paragraph.Length <= maxChars)
            {
                yield return paragraph;
                continue;
            }

            foreach (string sentence in SentenceBreak().Split(paragraph))
            {
                if (sentence.Length <= maxChars)
                {
                    yield return sentence;
                    continue;
                }

                // Extremely long sentence: fall back to packing words.
                var sb = new StringBuilder();
                foreach (string word in sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    if (sb.Length > 0 && sb.Length + 1 + word.Length > maxChars)
                    {
                        yield return sb.ToString();
                        sb.Clear();
                    }
                    if (sb.Length > 0) sb.Append(' ');
                    sb.Append(word.Length > maxChars ? word[..maxChars] : word);
                }
                if (sb.Length > 0) yield return sb.ToString();
            }
        }
    }
}
