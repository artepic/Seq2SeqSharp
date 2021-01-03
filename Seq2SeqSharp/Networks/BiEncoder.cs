
using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Seq2SeqSharp
{

    [Serializable]
    public class BiEncoder : IEncoder
    {
        private readonly List<LSTMCell> m_forwardEncoders;
        private readonly List<LSTMCell> m_backwardEncoders;
        private readonly string m_name;
        private readonly int m_hiddenDim;
        private readonly int m_inputDim;
        private readonly int m_depth;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;

        public BiEncoder(string name, int hiddenDim, int inputDim, int depth, int deviceId, bool isTrainable)
        {
            Logger.WriteLine($"Creating BiLSTM encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', IsTrainable = '{isTrainable}'");

            this.m_forwardEncoders = new List<LSTMCell>();
            this.m_backwardEncoders = new List<LSTMCell>();

            this.m_forwardEncoders.Add(new LSTMCell($"{name}.Forward_LSTM_0", hiddenDim, inputDim, deviceId, isTrainable: isTrainable));
            this.m_backwardEncoders.Add(new LSTMCell($"{name}.Backward_LSTM_0", hiddenDim, inputDim, deviceId, isTrainable: isTrainable));

            for (var i = 1; i < depth; i++)
            {
                this.m_forwardEncoders.Add(new LSTMCell($"{name}.Forward_LSTM_{i}", hiddenDim, hiddenDim * 2, deviceId, isTrainable: isTrainable));
                this.m_backwardEncoders.Add(new LSTMCell($"{name}.Backward_LSTM_{i}", hiddenDim, hiddenDim * 2, deviceId, isTrainable: isTrainable));
            }

            this.m_name = name;
            this.m_hiddenDim = hiddenDim;
            this.m_inputDim = inputDim;
            this.m_depth = depth;
            this.m_deviceId = deviceId;
            this.m_isTrainable = isTrainable;
        }

        public int GetDeviceId()
        {
            return this.m_deviceId;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new BiEncoder(this.m_name, this.m_hiddenDim, this.m_inputDim, this.m_depth, deviceId, this.m_isTrainable);
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            foreach (var item in this.m_forwardEncoders)
            {
                item.Reset(weightFactory, batchSize);
            }

            foreach (var item in this.m_backwardEncoders)
            {
                item.Reset(weightFactory, batchSize);
            }
        }

        public IWeightTensor Encode(IWeightTensor rawInputs, int batchSize, IComputeGraph g, IWeightTensor srcSelfMask)
        {
            var seqLen = rawInputs.Rows / batchSize;

            rawInputs = g.TransposeBatch(rawInputs, seqLen);

            var inputs = new List<IWeightTensor>();
            for (var i = 0; i < seqLen; i++)
            {
                var emb_i = g.PeekRow(rawInputs, i * batchSize, batchSize);
                inputs.Add(emb_i);
            }

            var forwardOutputs = new List<IWeightTensor>();
            var backwardOutputs = new List<IWeightTensor>();

            var layerOutputs = inputs.ToList();
            for (var i = 0; i < this.m_depth; i++)
            {
                for (var j = 0; j < seqLen; j++)
                {
                    var forwardOutput = this.m_forwardEncoders[i].Step(layerOutputs[j], g);
                    forwardOutputs.Add(forwardOutput);

                    var backwardOutput = this.m_backwardEncoders[i].Step(layerOutputs[inputs.Count - j - 1], g);
                    backwardOutputs.Add(backwardOutput);
                }

                backwardOutputs.Reverse();
                layerOutputs.Clear();
                for (var j = 0; j < seqLen; j++)
                {
                    var concatW = g.ConcatColumns(forwardOutputs[j], backwardOutputs[j]);
                    layerOutputs.Add(concatW);
                }

            }

            var result = g.ConcatRows(layerOutputs);

            return g.TransposeBatch(result, batchSize);
        }


        public List<IWeightTensor> GetParams()
        {
            var response = new List<IWeightTensor>();

            foreach (var item in this.m_forwardEncoders)
            {
                response.AddRange(item.getParams());
            }


            foreach (var item in this.m_backwardEncoders)
            {
                response.AddRange(item.getParams());
            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in this.m_forwardEncoders)
            {
                item.Save(stream);
            }

            foreach (var item in this.m_backwardEncoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in this.m_forwardEncoders)
            {
                item.Load(stream);
            }

            foreach (var item in this.m_backwardEncoders)
            {
                item.Load(stream);
            }
        }
    }
}
