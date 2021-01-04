using Seq2SeqSharp.Tools;

using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class PositionwiseFeedForward
    {
        private readonly LayerNormalization layerNorm2;
        private readonly FeedForwardLayer feedForwardLayer1;
        private readonly FeedForwardLayer feedForwardLayer2;

        private readonly int m_hiddenDim;
        private readonly string m_name;
        private readonly float m_dropoutRatio;

        public PositionwiseFeedForward(string name, int hiddenDim, float dropoutRatio, int deviceId, bool isTrainable)
        {
            this.m_name = name;
            this.m_hiddenDim = hiddenDim;
            this.m_dropoutRatio = dropoutRatio;

            this.layerNorm2 = new LayerNormalization($"{name}.{nameof(this.layerNorm2)}", hiddenDim, deviceId, isTrainable);
            this.feedForwardLayer1 = new FeedForwardLayer($"{name}.{nameof(this.feedForwardLayer1)}", hiddenDim, hiddenDim * 4, this.m_dropoutRatio, deviceId, isTrainable);
            this.feedForwardLayer2 = new FeedForwardLayer($"{name}.{nameof(this.feedForwardLayer2)}", hiddenDim * 4, hiddenDim, this.m_dropoutRatio, deviceId, isTrainable);
        }
      
        public IWeightTensor Perform(IWeightTensor input, int batchSize, IComputeGraph graph)
        {
            using (var g = graph.CreateSubGraph($"{this.m_name}_PositionwiseFeedForward"))
            {
                var inputNorm = this.layerNorm2.Norm(input, g);

                //Feed forward
                var ffnResult = this.feedForwardLayer1.Process(inputNorm, batchSize, g);
                var reluFFNResult = g.Relu(ffnResult, true);
                var ffn2Result = this.feedForwardLayer2.Process(reluFFNResult, batchSize, g);

                //Skip connection and layer normaliztion
                var addFFNResult = graph.Add(ffn2Result, input);

                return addFFNResult;
            }

        }

        public virtual List<IWeightTensor> getParams()
        {
            var response = new List<IWeightTensor>();

            response.AddRange(this.layerNorm2.getParams());
            response.AddRange(this.feedForwardLayer1.GetParams());
            response.AddRange(this.feedForwardLayer2.GetParams());

            return response;
        }


        public void Save(Stream stream)
        {
            this.layerNorm2.Save(stream);
            this.feedForwardLayer1.Save(stream);
            this.feedForwardLayer2.Save(stream);
        }


        public void Load(Stream stream)
        {
            this.layerNorm2.Load(stream);
            this.feedForwardLayer1.Load(stream);
            this.feedForwardLayer2.Load(stream);
        }
    }
}
