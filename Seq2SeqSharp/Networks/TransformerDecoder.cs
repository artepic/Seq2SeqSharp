using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class TransformerDecoder : IDecoder
    {
        private readonly List<MultiHeadAttention> m_selfAttns = new();
        private readonly List<MultiHeadAttention> m_encAttns = new();
        private readonly List<PositionwiseFeedForward> m_posFFNs = new();

        private readonly FeedForwardLayer m_decoderFFLayer;
        private readonly int m_inputDim;
        private readonly int m_outputDim;
        private readonly float m_dropoutRatio;
        private readonly string m_name;
        private readonly int m_multiHeadNum;
        private readonly int m_hiddenDim;
        private readonly int m_depth;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly LayerNormalization layerNorm;

        public TransformerDecoder(string name, int multiHeadNum, int hiddenDim, int inputDim, int outputDim, int depth, float dropoutRatio, int deviceId, bool isTrainable)
        {
            Logger.WriteLine($"Creating transformer decoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}'");

            this.m_name = name;
            this.m_multiHeadNum = multiHeadNum;
            this.m_hiddenDim = hiddenDim;
            this.m_inputDim = inputDim;
            this.m_outputDim = outputDim;
            this.m_depth = depth;
            this.m_dropoutRatio = dropoutRatio;
            this.m_deviceId = deviceId;
            this.m_isTrainable = isTrainable;

            if (hiddenDim != inputDim)
            {
                throw new ArgumentException($"hiddenDim is not equal to inputDim in TransformerEncoder.");
            }

            this.m_selfAttns.Add(new MultiHeadAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, this.m_dropoutRatio, deviceId, isTrainable, true));
            for (var i = 1; i < depth; i++)
            {
                this.m_selfAttns.Add(new MultiHeadAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, this.m_dropoutRatio, deviceId, isTrainable, true));
            }

            this.m_encAttns.Add(new MultiHeadAttention($"{name}.EncAttn_0", multiHeadNum, hiddenDim, inputDim, this.m_dropoutRatio, deviceId, isTrainable));
            for (var i = 1; i < depth; i++)
            {
                this.m_encAttns.Add(new MultiHeadAttention($"{name}.EncAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, this.m_dropoutRatio, deviceId, isTrainable));
            }

            for (var i = 0; i < depth; i++)
            {
                this.m_posFFNs.Add(new PositionwiseFeedForward($"{name}.PosFFN_{i}", hiddenDim, this.m_dropoutRatio, deviceId, isTrainable));
            }

            this.layerNorm = new LayerNormalization($"{name}.{nameof(this.layerNorm)}", hiddenDim, deviceId, isTrainable);

            this.m_decoderFFLayer = new FeedForwardLayer($"{name}.FeedForward", hiddenDim, outputDim, 0.0f, deviceId, isTrainable);

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
        /// 

        public IWeightTensor Decode(IWeightTensor tgtInputs, IWeightTensor encOutputBatchFirst, IWeightTensor tgtSelfMask, IWeightTensor srcTgtMask, int batchSize, IComputeGraph g)
        {
            using (var subg = g.CreateSubGraph($"{this.m_name}_Decoder"))
            {
                for (var k = 0; k < this.m_selfAttns.Count; k++)
                {
                    tgtInputs = this.m_selfAttns[k].Perform(tgtInputs, tgtSelfMask, batchSize, subg);
                    tgtInputs = this.m_encAttns[k].Perform(tgtInputs, encOutputBatchFirst, encOutputBatchFirst, srcTgtMask, batchSize, subg);
                    tgtInputs = this.m_posFFNs[k].Perform(tgtInputs, batchSize, subg);
                }

                tgtInputs = this.layerNorm.Norm(tgtInputs, subg);

                tgtInputs.UnbindFromComputeGraph();
            }
            

            tgtInputs = this.m_decoderFFLayer.Process(tgtInputs, batchSize, g);

            return tgtInputs;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new TransformerDecoder(this.m_name, this.m_multiHeadNum, this.m_hiddenDim, this.m_inputDim, this.m_outputDim, this.m_depth, this.m_dropoutRatio, deviceId, this.m_isTrainable);
        }

        public List<IWeightTensor> GetParams()
        {
            var response = new List<IWeightTensor>();

            foreach (var item in this.m_selfAttns)
            {
                response.AddRange(item.getParams());
            }

            foreach (var item in this.m_encAttns)
            {
                response.AddRange(item.getParams());
            }

            foreach (var item in this.m_posFFNs)
            {
                response.AddRange(item.getParams());
            }

            response.AddRange(this.layerNorm.getParams());
            response.AddRange(this.m_decoderFFLayer.GetParams());

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in this.m_selfAttns)
            {
                item.Save(stream);
            }

            foreach (var item in this.m_encAttns)
            {
                item.Save(stream);
            }

            foreach (var item in this.m_posFFNs)
            {
                item.Save(stream);
            }

            this.layerNorm.Save(stream);
            this.m_decoderFFLayer.Save(stream);
        }

        public void Load(Stream stream)
        {
            foreach (var item in this.m_selfAttns)
            {
                item.Load(stream);
            }

            foreach (var item in this.m_encAttns)
            {
                item.Load(stream);
            }

            foreach (var item in this.m_posFFNs)
            {
                item.Load(stream);
            }

            this.layerNorm.Load(stream);
            this.m_decoderFFLayer.Load(stream);
        }
    }
}
