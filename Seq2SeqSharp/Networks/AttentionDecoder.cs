
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Seq2SeqSharp
{


    [Serializable]
    public class AttentionDecoder : IDecoder
    {
        private readonly List<LSTMAttentionDecoderCell> m_decoders = new List<LSTMAttentionDecoderCell>();
        private readonly FeedForwardLayer m_decoderFFLayer;
        private readonly int m_hdim;
        private readonly int m_embDim;
        private readonly int m_outputDim;
        private readonly float m_dropoutRatio;
        private readonly int m_depth;
        private readonly int m_context;
        private readonly int m_deviceId;
        private readonly AttentionUnit m_attentionLayer;
        private readonly string m_name;
        private readonly bool m_enableCoverageModel;
        private readonly bool m_isTrainable;

        public AttentionDecoder(string name, int hiddenDim, int embeddingDim, int contextDim, int outputDim, float dropoutRatio, int depth, int deviceId, bool enableCoverageModel, bool isTrainable)
        {
            this.m_name = name;
            this.m_hdim = hiddenDim;
            this.m_embDim = embeddingDim;
            this.m_context = contextDim;
            this.m_depth = depth;
            this.m_deviceId = deviceId;
            this.m_outputDim = outputDim;
            this.m_dropoutRatio = dropoutRatio;
            this.m_enableCoverageModel = enableCoverageModel;
            this.m_isTrainable = isTrainable;

            this.m_attentionLayer = new AttentionUnit($"{name}.AttnUnit", hiddenDim, contextDim, deviceId, enableCoverageModel, isTrainable: isTrainable);

            this.m_decoders.Add(new LSTMAttentionDecoderCell($"{name}.LSTMAttn_0", hiddenDim, embeddingDim, contextDim, deviceId, isTrainable));
            for (var i = 1; i < depth; i++)
            {
                this.m_decoders.Add(new LSTMAttentionDecoderCell($"{name}.LSTMAttn_{i}", hiddenDim, hiddenDim, contextDim, deviceId, isTrainable));
            }

            this.m_decoderFFLayer = new FeedForwardLayer($"{name}.FeedForward", hiddenDim, outputDim, 0.0f, deviceId: deviceId, isTrainable: isTrainable);

        }

        public int GetDeviceId()
        {
            return this.m_deviceId;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new AttentionDecoder(this.m_name, this.m_hdim, this.m_embDim, this.m_context, this.m_outputDim, this.m_dropoutRatio, this.m_depth, deviceId, this.m_enableCoverageModel, this.m_isTrainable);
        }


        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            foreach (var item in this.m_decoders)
            {
                item.Reset(weightFactory, batchSize);
            }
        }

        public AttentionPreProcessResult PreProcess(IWeightTensor encOutputs, int batchSize, IComputeGraph g)
        {
            return this.m_attentionLayer.PreProcess(encOutputs, batchSize, g);
        }


        public IWeightTensor Decode(IWeightTensor input, AttentionPreProcessResult attenPreProcessResult, int batchSize, IComputeGraph g)
        {
            var V = input;
            var lastStatus = this.m_decoders.LastOrDefault().Cell;
            var context = this.m_attentionLayer.Perform(lastStatus, attenPreProcessResult, batchSize, g);

            foreach (var decoder in this.m_decoders)
            {
                var e = decoder.Step(context, V, g);
                V = e;
            }

            var eOutput = g.Dropout(V, batchSize, this.m_dropoutRatio, false);
            eOutput = this.m_decoderFFLayer.Process(eOutput, batchSize, g);

            return eOutput;
        }


        public List<IWeightTensor> GetCTs()
        {
            var res = new List<IWeightTensor>();
            foreach (var decoder in this.m_decoders)
            {
                res.Add(decoder.Cell);
            }

            return res;
        }

        public List<IWeightTensor> GetHTs()
        {
            var res = new List<IWeightTensor>();
            foreach (var decoder in this.m_decoders)
            {
                res.Add(decoder.Hidden);
            }

            return res;
        }

        public void SetCTs(List<IWeightTensor> l)
        {
            for (var i = 0; i < l.Count; i++)
            {
                this.m_decoders[i].Cell = l[i];
            }
        }

        public void SetHTs(List<IWeightTensor> l)
        {
            for (var i = 0; i < l.Count; i++)
            {
                this.m_decoders[i].Hidden = l[i];
            }
        }

        public List<IWeightTensor> GetParams()
        {
            var response = new List<IWeightTensor>();

            foreach (var item in this.m_decoders)
            {
                response.AddRange(item.getParams());
            }
            response.AddRange(this.m_attentionLayer.GetParams());
            response.AddRange(this.m_decoderFFLayer.GetParams());

            return response;
        }

        public void Save(Stream stream)
        {
            this.m_attentionLayer.Save(stream);
            foreach (var item in this.m_decoders)
            {
                item.Save(stream);
            }

            this.m_decoderFFLayer.Save(stream);
        }

        public void Load(Stream stream)
        {
            this.m_attentionLayer.Load(stream);
            foreach (var item in this.m_decoders)
            {
                item.Load(stream);
            }

            this.m_decoderFFLayer.Load(stream);
        }
    }
}
