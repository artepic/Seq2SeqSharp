using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    [Serializable]
    internal class LayerNormalization
    {
        private readonly IWeightTensor m_alpha;
        private readonly IWeightTensor m_beta;
        private readonly string m_name;

        public LayerNormalization(string name, int dim, int deviceId, bool isTrainable)
        {
            this.m_name = name;
            this.m_alpha = new WeightTensor(new long[2] { 1, dim }, 1.0f, deviceId, name: $"{name}.{nameof(this.m_alpha)}", isTrainable: isTrainable);
            this.m_beta = new WeightTensor(new long[2] { 1, dim }, 0, deviceId, name: $"{name}.{nameof(this.m_beta)}", isTrainable: isTrainable);
        }

        public IWeightTensor Norm(IWeightTensor input, IComputeGraph g)
        {
            return g.LayerNorm(input, this.m_alpha, this.m_beta, 1e-06f);
        }

        /// <summary>
        /// LayerNorm (input1 + input2)
        /// </summary>
        /// <param name="input1"></param>
        /// <param name="input2"></param>
        /// <param name="g"></param>
        /// <returns></returns>
        public IWeightTensor AddNorm(IWeightTensor input1, IWeightTensor input2, IComputeGraph g)
        {
            return g.AddLayerNorm(input1, input2, this.m_alpha, this.m_beta);
        }

        public virtual List<IWeightTensor> getParams()
        {
            var response = new List<IWeightTensor>
            {
                this.m_alpha, this.m_beta
            };

            return response;
        }

        public void Save(Stream stream)
        {
            this.m_alpha.Save(stream);
            this.m_beta.Save(stream);
        }


        public void Load(Stream stream)
        {
            this.m_alpha.Load(stream);
            this.m_beta.Load(stream);
        }
    }
}
