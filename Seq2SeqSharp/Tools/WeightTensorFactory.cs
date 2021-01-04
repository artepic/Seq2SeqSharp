using System.Collections.Generic;

namespace Seq2SeqSharp.Tools
{
    public class WeightTensorFactory : IWeightFactory
    {
        private readonly List<WeightTensor> weights = new();

        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false, string name = "", bool isTrainable = false, IComputeGraph graphToBind = null)
        {
            var r = new WeightTensor(new long[2] { row, column }, deviceId, name, isTrainable, graphToBind: graphToBind);

            if (cleanWeights)
            {
                r.CleanWeight();
            }

            this.weights.Add(r);

            return r;
        }

        public WeightTensor CreateWeightTensor(long[] sizes, int deviceId, bool cleanWeights = false, string name = "", IComputeGraph graphToBind = null)
        {
            var r = new WeightTensor(sizes, deviceId, name, graphToBind: graphToBind);

            if (cleanWeights)
            {
                r.CleanWeight();
            }

            this.weights.Add(r);

            return r;
        }

        public void Dispose()
        {
            foreach (var item in this.weights)
            {
                item.Dispose();
            }

            this.weights.Clear();
        }
    }
}
