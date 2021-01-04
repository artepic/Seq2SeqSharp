using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class TransformerEncoder : IEncoder
    {
        private readonly List<MultiHeadAttention> m_encoders = new List<MultiHeadAttention>();
        private readonly List<PositionwiseFeedForward> m_posFFNs = new List<PositionwiseFeedForward>();

        private readonly int m_inputDim;
        private readonly float m_dropoutRatio;
        private readonly string m_name;
        private readonly int m_multiHeadNum;
        private readonly int m_hiddenDim;
        private readonly int m_depth;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly LayerNormalization layerNorm;

        public TransformerEncoder(string name, int multiHeadNum, int hiddenDim, int inputDim, int depth, float dropoutRatio, int deviceId, bool isTrainable)
        {
            Logger.WriteLine($"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}'");

            this.m_name = name;
            this.m_multiHeadNum = multiHeadNum;
            this.m_hiddenDim = hiddenDim;
            this.m_inputDim = inputDim;
            this.m_depth = depth;
            this.m_dropoutRatio = dropoutRatio;
            this.m_deviceId = deviceId;
            this.m_isTrainable = isTrainable;

            if (hiddenDim != inputDim)
            {
                throw new ArgumentException($"hiddenDim is not equal to inputDim in TransformerEncoder.");
            }

            this.m_encoders.Add(new MultiHeadAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, this.m_dropoutRatio, deviceId, isTrainable, true));
            for (var i = 1; i < depth; i++)
            {
                this.m_encoders.Add(new MultiHeadAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, this.m_dropoutRatio, deviceId, isTrainable, true));              
            }

            for (var i = 0; i < depth; i++)
            {
                this.m_posFFNs.Add(new PositionwiseFeedForward($"{name}.PosFFN_{i}", hiddenDim, this.m_dropoutRatio, deviceId, isTrainable));
            }

            this.layerNorm = new LayerNormalization($"{name}.{nameof(this.layerNorm)}", hiddenDim, deviceId, isTrainable);

        }

        public int GetDeviceId()
        {
            return this.m_deviceId;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
        }

        /// <summary>
        /// Transformer encoder
        /// </summary>
        /// <param name="rawInputs"></param>
        /// <param name="g"></param>
        /// <returns></returns>
        public IWeightTensor Encode(IWeightTensor inputs, int batchSize, IComputeGraph g, IWeightTensor srcSelfMask)
        {
            using (var subg = g.CreateSubGraph($"{this.m_name}_Encoder"))
            {
                for (var k = 0; k < this.m_encoders.Count; k++)
                {
                    inputs = this.m_encoders[k].Perform(inputs, srcSelfMask, batchSize, subg);
                    inputs = this.m_posFFNs[k].Perform(inputs, batchSize, subg);
                }
                inputs.UnbindFromComputeGraph();
            }

            inputs = this.layerNorm.Norm(inputs, g);

            return inputs;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new TransformerEncoder(this.m_name, this.m_multiHeadNum, this.m_hiddenDim, this.m_inputDim, this.m_depth, this.m_dropoutRatio, deviceId, this.m_isTrainable);
        }

        public List<IWeightTensor> GetParams()
        {
            var response = new List<IWeightTensor>();

            foreach (var item in this.m_encoders)
            {
                response.AddRange(item.getParams());
            }

            foreach (var item in this.m_posFFNs)
            {
                response.AddRange(item.getParams());
            }

            response.AddRange(this.layerNorm.getParams());

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in this.m_encoders)
            {
                item.Save(stream);
            }

            foreach (var item in this.m_posFFNs)
            {
                item.Save(stream);
            }

            this.layerNorm.Save(stream);
        }

        public void Load(Stream stream)
        {
            foreach (var item in this.m_encoders)
            {
                item.Load(stream);
            }

            foreach (var item in this.m_posFFNs)
            {
                item.Load(stream);
            }

            this.layerNorm.Load(stream);
        }
    }
}
