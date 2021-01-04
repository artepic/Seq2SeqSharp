using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    [Serializable]
    public class LSTMCell
    {
        private readonly IWeightTensor m_Wxh;
        private readonly IWeightTensor m_b;
        private IWeightTensor m_hidden;
        private IWeightTensor m_cell;
        private readonly int m_hdim;
        private readonly int m_dim;
        private readonly int m_deviceId;
        private readonly string m_name;
        private readonly LayerNormalization m_layerNorm1;
        private readonly LayerNormalization m_layerNorm2;

        public IWeightTensor Hidden => this.m_hidden;

        public LSTMCell(string name, int hdim, int dim, int deviceId, bool isTrainable)
        {
            this.m_name = name;

            this.m_Wxh = new WeightTensor(new long[2] { dim + hdim, hdim * 4 }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(this.m_Wxh)}", isTrainable: isTrainable);
            this.m_b = new WeightTensor(new long[2] { 1, hdim * 4 }, 0, deviceId, $"{name}.{nameof(this.m_b)}", isTrainable);

            this.m_hdim = hdim;
            this.m_dim = dim;
            this.m_deviceId = deviceId;

            this.m_layerNorm1 = new LayerNormalization($"{name}.{nameof(this.m_layerNorm1)}", hdim * 4, deviceId, isTrainable);
            this.m_layerNorm2 = new LayerNormalization($"{name}.{nameof(this.m_layerNorm2)}", hdim, deviceId, isTrainable);
        }

        public IWeightTensor Step(IWeightTensor input, IComputeGraph g)
        {
            using (var innerGraph = g.CreateSubGraph(this.m_name))
            {
                var hidden_prev = this.m_hidden;
                var cell_prev = this.m_cell;

                var inputs = innerGraph.ConcatColumns(input, hidden_prev);
                var hhSum = innerGraph.Affine(inputs, this.m_Wxh, this.m_b);
                var hhSum2 = this.m_layerNorm1.Norm(hhSum, innerGraph);

                var (gates_raw, cell_write_raw) = innerGraph.SplitColumns(hhSum2, this.m_hdim * 3, this.m_hdim);
                var gates = innerGraph.Sigmoid(gates_raw);
                var cell_write = innerGraph.Tanh(cell_write_raw);

                var (input_gate, forget_gate, output_gate) = innerGraph.SplitColumns(gates, this.m_hdim, this.m_hdim, this.m_hdim);

                // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
                this.m_cell = g.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
                var ct2 = this.m_layerNorm2.Norm(this.m_cell, innerGraph);

                // compute hidden state as gated, saturated cell activations
                this.m_hidden = g.EltMul(output_gate, innerGraph.Tanh(ct2));

                return this.m_hidden;
            }
        }

        public virtual List<IWeightTensor> getParams()
        {
            var response = new List<IWeightTensor>
            {
                this.m_Wxh, this.m_b
            };

            response.AddRange(this.m_layerNorm1.getParams());
            response.AddRange(this.m_layerNorm2.getParams());

            return response;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            if (this.m_hidden != null)
            {
                this.m_hidden.Dispose();
                this.m_hidden = null;
            }

            if (this.m_cell != null)
            {
                this.m_cell.Dispose();
                this.m_cell = null;
            }

            this.m_hidden = weightFactory.CreateWeightTensor(batchSize, this.m_hdim, this.m_deviceId, true, $"{this.m_name}.{nameof(this.m_hidden)}", true);
            this.m_cell = weightFactory.CreateWeightTensor(batchSize, this.m_hdim, this.m_deviceId, true, $"{this.m_name}.{nameof(this.m_cell)}", true);
        }

        public void Save(Stream stream)
        {
            this.m_Wxh.Save(stream);
            this.m_b.Save(stream);

            this.m_layerNorm1.Save(stream);
            this.m_layerNorm2.Save(stream);

        }


        public void Load(Stream stream)
        {
            this.m_Wxh.Load(stream);
            this.m_b.Load(stream);

            this.m_layerNorm1.Load(stream);
            this.m_layerNorm2.Load(stream);
        }
    }

}
