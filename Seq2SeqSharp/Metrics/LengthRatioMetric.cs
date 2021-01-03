using System.Collections.Generic;

namespace Seq2SeqSharp.Metrics
{
    public class LengthRatioMetric : IMetric
    {
        private double[] m_counts;
        public string Name => "Length Ratio (Hyp:Ref)";

        public LengthRatioMetric()
        {
            this.ClearStatus();
        }

        public void ClearStatus()
        {
            this.m_counts = new double[2];
        }

        public void Evaluate(List<List<string>> refTokens, List<string> hypTokens)
        {
            this.m_counts[0] += hypTokens.Count;
            this.m_counts[1] += BleuMetric.GetClosestRefLength(refTokens, hypTokens);
        }

        public string GetScoreStr()
        {
            return this.GetPrimaryScore().ToString("F");
        }

        public double GetPrimaryScore()
        {
            var lr = this.m_counts[0] / this.m_counts[1];
            return 100.0 * lr;
        }
    }
}
