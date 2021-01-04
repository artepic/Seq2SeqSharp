using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    [Serializable]
    public class LSTMAttentionDecoderCell
    {
        public IWeightTensor Hidden { get; set; }
        public IWeightTensor Cell { get; set; }

        private readonly int m_hiddenDim;
        private readonly int m_inputDim;
        private readonly int m_deviceId;
        private readonly string m_name;
        private readonly IWeightTensor m_Wxhc;
        private readonly IWeightTensor m_b;
        private readonly LayerNormalization m_layerNorm1;
        private readonly LayerNormalization m_layerNorm2;

        public LSTMAttentionDecoderCell(string name, int hiddenDim, int inputDim, int contextDim, int deviceId, bool isTrainable)
        {
            this.m_name = name;
            this.m_hiddenDim = hiddenDim;
            this.m_inputDim = inputDim;
            this.m_deviceId = deviceId;

            Logger.WriteLine($"Create LSTM attention decoder cell '{name}' HiddemDim = '{hiddenDim}', InputDim = '{inputDim}', ContextDim = '{contextDim}', DeviceId = '{deviceId}'");

            this.m_Wxhc = new WeightTensor(new long[2] { inputDim + hiddenDim + contextDim, hiddenDim * 4 }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(this.m_Wxhc)}", isTrainable: isTrainable);
            this.m_b = new WeightTensor(new long[2] { 1, hiddenDim * 4 }, 0, deviceId, $"{name}.{nameof(this.m_b)}", isTrainable);

            this.m_layerNorm1 = new LayerNormalization($"{name}.{nameof(this.m_layerNorm1)}", hiddenDim * 4, deviceId, isTrainable);
            this.m_layerNorm2 = new LayerNormalization($"{name}.{nameof(this.m_layerNorm2)}", hiddenDim, deviceId, isTrainable);
        }

        /// <summary>
        /// Update LSTM-Attention cells according to given weights
        /// </summary>
        /// <param name="context">The context weights for attention</param>
        /// <param name="input">The input weights</param>
        /// <param name="computeGraph">The compute graph to build workflow</param>
        /// <returns>Update hidden weights</returns>
        public IWeightTensor Step(IWeightTensor context, IWeightTensor input, IComputeGraph g)
        {
            using (var computeGraph = g.CreateSubGraph(this.m_name))
            {
                var cell_prev = this.Cell;
                var hidden_prev = this.Hidden;

                var hxhc = computeGraph.ConcatColumns(input, hidden_prev, context);
                var hhSum = computeGraph.Affine(hxhc, this.m_Wxhc, this.m_b);
                var hhSum2 = this.m_layerNorm1.Norm(hhSum, computeGraph);

                var (gates_raw, cell_write_raw) = computeGraph.SplitColumns(hhSum2, this.m_hiddenDim * 3, this.m_hiddenDim);
                var gates = computeGraph.Sigmoid(gates_raw);
                var cell_write = computeGraph.Tanh(cell_write_raw);

                var (input_gate, forget_gate, output_gate) = computeGraph.SplitColumns(gates, this.m_hiddenDim, this.m_hiddenDim, this.m_hiddenDim);

                // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
                this.Cell = g.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
                var ct2 = this.m_layerNorm2.Norm(this.Cell, computeGraph);

                this.Hidden = g.EltMul(output_gate, computeGraph.Tanh(ct2));


                return this.Hidden;
            }
        }

        public List<IWeightTensor> getParams()
        {
            var response = new List<IWeightTensor>
            {
                this.m_Wxhc, this.m_b
            };

            response.AddRange(this.m_layerNorm1.getParams());
            response.AddRange(this.m_layerNorm2.getParams());

            return response;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            this.Hidden = weightFactory.CreateWeightTensor(batchSize, this.m_hiddenDim, this.m_deviceId, true, $"{this.m_name}.{nameof(this.Hidden)}", true);
            this.Cell = weightFactory.CreateWeightTensor(batchSize, this.m_hiddenDim, this.m_deviceId, true, $"{this.m_name}.{nameof(this.Cell)}", true);
        }

        public void Save(Stream stream)
        {
            this.m_Wxhc.Save(stream);
            this.m_b.Save(stream);

            this.m_layerNorm1.Save(stream);
            this.m_layerNorm2.Save(stream);
        }


        public void Load(Stream stream)
        {
            this.m_Wxhc.Load(stream);
            this.m_b.Load(stream);

            this.m_layerNorm1.Load(stream);
            this.m_layerNorm2.Load(stream);
        }
    }
}


