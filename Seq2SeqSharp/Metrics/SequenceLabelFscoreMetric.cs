using System.Collections.Generic;

namespace Seq2SeqSharp.Metrics
{
    public class SequenceLabelFscoreMetric : IMetric
    {
        private double[] m_count;
        private readonly string m_classLabel;

        public string Name => $"SequenceLabelFscore ({this.m_classLabel})";

        public SequenceLabelFscoreMetric(string classLabel)
        {
            this.m_count = new double[3];
            this.m_classLabel = classLabel;
        }

        public void ClearStatus()
        {
            this.m_count = new double[3];
        }

        public void Evaluate(List<List<string>> allRefTokens, List<string> hypTokens)
        {
            foreach (var refTokens in allRefTokens)
            {
                for (var i = 0; i < hypTokens.Count; i++)
                {
                    if (hypTokens[i] == this.m_classLabel)
                    {
                        this.m_count[1]++;
                    }
                    if (refTokens[i] == this.m_classLabel)
                    {
                        this.m_count[2]++;
                    }
                    if (hypTokens[i] == this.m_classLabel && refTokens[i] == this.m_classLabel)
                    {
                        this.m_count[0]++;
                    }
                }
            }
        }

        public string GetScoreStr()
        {
            var precision = this.m_count[0] / this.m_count[1];
            var recall = this.m_count[0] / this.m_count[2];
            var objective = 0.0;
            if (precision > 0.0 && recall > 0.0)
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return $"F-score = '{(100.0 * objective).ToString("F")}' Precision = '{(100.0 * precision).ToString("F")}' Recall = '{(100.0 * recall).ToString("F")}'";
        }

        public double GetPrimaryScore()
        {
            var precision = this.m_count[0] / this.m_count[1];
            var recall = this.m_count[0] / this.m_count[2];
            var objective = 0.0;
            if (precision > 0.0 && recall > 0.0)
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return 100.0 * objective;
        }
    }
}
