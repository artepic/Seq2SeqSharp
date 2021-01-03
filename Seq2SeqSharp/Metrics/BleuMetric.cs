using System;
using System.Collections.Generic;

namespace Seq2SeqSharp.Metrics
{
    public enum RefHypIdx
    {
        RefIdx = 0,
        HypIdx = 1
    }

    public class BleuMetric : IMetric
    {
        private double[] m_counts;
        private readonly int m_matchIndex;

        private bool m_caseInsensitive { get; }
        private int m_ngramOrder { get; }
        public string Name => "BLEU";

        public BleuMetric(int ngramOrder = 4, bool caseInsensitive = true)
        {
            this.m_ngramOrder = ngramOrder;
            this.m_caseInsensitive = caseInsensitive;
            this.m_matchIndex = (int)RefHypIdx.HypIdx + this.m_ngramOrder;

            this.ClearStatus();
        }

        public void ClearStatus()
        {
            this.m_counts = new double[1 + 2 * this.m_ngramOrder];
        }

        public void Evaluate(List<List<string>> refTokens, List<string> hypTokens)
        {
            if (this.m_caseInsensitive)
            {
                for (var i = 0; i < refTokens.Count; i++)
                {
                    refTokens[i] = this.ToLowerCase(refTokens[i]);
                }
                hypTokens = this.ToLowerCase(hypTokens);
            }
            var refCounts = new List<Dictionary<string, int>>();
            for (var n = 0; n < this.m_ngramOrder; n++)
            {
                refCounts.Add(new Dictionary<string, int>());
            }
            foreach (var r in refTokens)
            {
                var counts = this.GetNgramCounts(r);
                for (var n = 0; n < this.m_ngramOrder; n++)
                {
                    foreach (var e in counts[n])
                    {
                        var ngram = e.Key;
                        var count = e.Value;
                        if (!refCounts[n].ContainsKey(ngram) || count > refCounts[n][ngram])
                        {
                            refCounts[n][ngram] = count;
                        }
                    }
                }
            }

            var hypCounts = this.GetNgramCounts(hypTokens);

            this.m_counts[(int)RefHypIdx.RefIdx] += GetClosestRefLength(refTokens, hypTokens);
            for (var j = 0; j < this.m_ngramOrder; j++)
            {
                var overlap = 0;
                foreach (var e in hypCounts[j])
                {
                    var ngram = e.Key;
                    var hypCount = e.Value;
                    if (refCounts[j].TryGetValue(ngram, out var refCount))
                    {
                        overlap += Math.Min(hypCount, refCount);
                    }
                }

                this.m_counts[(int)RefHypIdx.HypIdx + j] += Math.Max(0, hypTokens.Count - j);
                this.m_counts[this.m_matchIndex + j] += overlap;
            }
        }

        public string GetScoreStr()
        {
            return this.GetPrimaryScore().ToString("F");
        }

        public double GetPrimaryScore()
        {
            var precision = this.Precision();
            var bp = this.BrevityPenalty();

            return 100.0 * precision * bp;
        }

        internal static double GetClosestRefLength(List<List<string>> refTokens, List<string> hypTokens)
        {
            var closestIndex = -1;
            var closestDistance = int.MaxValue;
            for (var i = 0; i < refTokens.Count; i++)
            {
                var distance = Math.Abs(refTokens[i].Count - hypTokens.Count);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestIndex = i;
                }
            }

            return refTokens[closestIndex].Count;
        }

        private double BrevityPenalty()
        {
            var refLen = this.m_counts[(int)RefHypIdx.RefIdx];
            var hypLen = this.m_counts[(int)RefHypIdx.HypIdx];
            if (hypLen == 0.0 || hypLen >= refLen)
            {
                return 1.0;
            }

            return Math.Exp(1.0 - refLen / hypLen);
        }

        private List<Dictionary<string, int>> GetNgramCounts(List<string> tokens)
        {
            var allCounts = new List<Dictionary<string, int>>();
            for (var n = 0; n < this.m_ngramOrder; n++)
            {
                var counts = new Dictionary<string, int>();
                for (var i = 0; i < tokens.Count - n; i++)
                {
                    var ngram = string.Join(" ", tokens.ToArray(), i, n + 1);
                    if (!counts.ContainsKey(ngram))
                    {
                        counts[ngram] = 1;
                    }
                    else
                    {
                        counts[ngram]++;
                    }
                }
                allCounts.Add(counts);
            }
            return allCounts;
        }

        private double Precision()
        {
            var prec = 1.0;
            for (var i = 0; i < this.m_ngramOrder; i++)
            {
                var x = this.m_counts[this.m_matchIndex + i] / (this.m_counts[(int)RefHypIdx.HypIdx + i] + 0.001);
                prec *= Math.Pow(x, 1.0 / this.m_ngramOrder);
            }
            return prec;
        }

        private List<string> ToLowerCase(List<string> tokens)
        {
            var output = new List<string>();
            for (var i = 0; i < tokens.Count; i++)
            {
                output.Add(tokens[i].ToLower());
            }
            return output;
        }
    }
}
