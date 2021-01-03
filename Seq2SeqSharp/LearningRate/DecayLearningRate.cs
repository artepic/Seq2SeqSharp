using AdvUtils;
using System;

namespace Seq2SeqSharp
{
    public class DecayLearningRate : ILearningRate
    {
        private readonly float m_startLearningRate = 0.001f;
        private int m_weightsUpdateCount = 0;
        private readonly int m_warmupSteps = 8000;

        public DecayLearningRate(float startLearningRate, int warmupSteps, int weightsUpdatesCount)
        {
            Logger.WriteLine($"Creating decay learning rate. StartLearningRate = '{startLearningRate}', WarmupSteps = '{warmupSteps}', WeightsUpdatesCount = '{weightsUpdatesCount}'");
            this.m_startLearningRate = startLearningRate;
            this.m_warmupSteps = warmupSteps;
            this.m_weightsUpdateCount = weightsUpdatesCount;
        }

        public float GetCurrentLearningRate()
        {
            this.m_weightsUpdateCount++;
            var lr = this.m_startLearningRate * (float)(Math.Min(Math.Pow(this.m_weightsUpdateCount, -0.5), Math.Pow(this.m_warmupSteps, -1.5) * this.m_weightsUpdateCount) / Math.Pow(this.m_warmupSteps, -0.5));
            return lr;
        }
    }
}
