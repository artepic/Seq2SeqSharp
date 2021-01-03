using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class MultiHeadAttention
    {
        private readonly IWeightTensor W0;
        private readonly IWeightTensor b0;

        private readonly IWeightTensor Q;
        private readonly IWeightTensor K;
        private readonly IWeightTensor V;

        private readonly IWeightTensor Qb;
        private readonly IWeightTensor Kb;
        private readonly IWeightTensor Vb;


        private readonly IWeightTensor QKV;

        private readonly LayerNormalization layerNormQ;

        private readonly int m_hiddenDim;
        private readonly int m_d;
        private readonly int m_multiHeadNum;
        private readonly string m_name;
        private readonly float m_dropoutRatio;
        private readonly float m_inputDim;
        private readonly bool m_sharedQKV = false;

        public MultiHeadAttention(string name, int multiHeadNum, int hiddenDim, int inputDim, float dropoutRatio, int deviceId, bool isTrainable, bool sharedQKV = false)
        {
            this.m_name = name;
            this.m_hiddenDim = hiddenDim;
            this.m_inputDim = inputDim;
            this.m_multiHeadNum = multiHeadNum;
            this.m_d = this.m_hiddenDim / this.m_multiHeadNum;
            this.m_dropoutRatio = dropoutRatio;
            this.m_sharedQKV = sharedQKV;

            this.W0 = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, name: $"{name}.{nameof(this.W0)}", isTrainable: isTrainable, normal: NormType.Uniform);
            this.b0 = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(this.b0)}", isTrainable: isTrainable);

            if (this.m_sharedQKV == false)
            {
                this.Q = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(this.Q)}", isTrainable: isTrainable, normal: NormType.Uniform);
                this.Qb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(this.Qb)}", isTrainable: isTrainable);

                this.K = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(this.K)}", isTrainable: isTrainable, normal: NormType.Uniform);
                this.Kb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(this.Kb)}", isTrainable: isTrainable);

                this.V = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(this.V)}", isTrainable: isTrainable, normal: NormType.Uniform);
                this.Vb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(this.Vb)}", isTrainable: isTrainable);
            }
            else
            {
                this.QKV = new WeightTensor(new long[] { 3, inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(this.QKV)}", isTrainable: isTrainable, normal: NormType.Uniform);
            }

            this.layerNormQ = new LayerNormalization($"{name}.{nameof(this.layerNormQ)}", this.m_hiddenDim, deviceId, isTrainable);
        }

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="inputK">The input K tensor</param>
        /// <param name="inputV">The input V tensor</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public IWeightTensor Perform(IWeightTensor inputQ, IWeightTensor inputK, IWeightTensor inputV, IWeightTensor keyMask, int batchSize, IComputeGraph graph)
        {
            if (this.m_sharedQKV)
            {
                throw new ArgumentException($"Layer '{this.m_name}' is in shared QKV mode, please call antoher Perform function with single input tensor.");
            }

            using (var g = graph.CreateSubGraph($"{this.m_name}_MultiHeadAttention"))
            {
                var seqLenQ = inputQ.Rows / batchSize;

                // SeqLenK must be euqal to SeqLenV
                var seqLenK = inputK.Rows / batchSize;
                var seqLenV = inputV.Rows / batchSize;

                var inputQNorm = this.layerNormQ.Norm(inputQ, g);
                //Input projections
                var scale = 1.0f / (float)(this.m_inputDim);
                var allQ = g.View(g.Affine(inputQNorm, this.Q, this.Qb, scale), dims: new long[] { batchSize, seqLenQ, this.m_multiHeadNum, this.m_d });
                var allK = g.View(g.Affine(inputK, this.K, this.Kb, scale), dims: new long[] { batchSize, seqLenK, this.m_multiHeadNum, this.m_d });
                var allV = g.View(g.Affine(inputV, this.V, this.Vb, scale), dims: new long[] { batchSize, seqLenV, this.m_multiHeadNum, this.m_d });

                //Multi-head attentions
                var Qs = g.View(g.Permute(allQ, 2, 0, 1, 3), dims: new long[] { this.m_multiHeadNum * batchSize, seqLenQ, this.m_d });
                var Ks = g.View(g.Permute(allK, 2, 0, 3, 1), dims: new long[] { this.m_multiHeadNum * batchSize, this.m_d, seqLenK });
                var Vs = g.View(g.Permute(allV, 2, 0, 1, 3), dims: new long[] { this.m_multiHeadNum * batchSize, seqLenV, this.m_d });

                // Scaled softmax
                scale = 1.0f / (float)(this.m_d);
                var attn = g.MulBatch(Qs, Ks, this.m_multiHeadNum * batchSize, scale);
                var softmax = g.Softmax(attn, keyMask, inPlace: true);
                var o = g.View(g.MulBatch(softmax, Vs, this.m_multiHeadNum * batchSize), dims: new long[] { this.m_multiHeadNum, batchSize, seqLenQ, this.m_d });

                var W = g.View(g.Permute(o, 1, 2, 0, 3), dims: new long[] { batchSize * seqLenQ, this.m_multiHeadNum * this.m_d });

                // Output projection
                var finalAttResults = g.Dropout(g.Affine(W, this.W0, this.b0), batchSize, this.m_dropoutRatio, inPlace: true);

                return graph.Add(finalAttResults, inputQ);
            }
        }


        public IWeightTensor Perform(IWeightTensor inputQ, IWeightTensor keyMask, int batchSize, IComputeGraph graph)
        {
            if (this.m_sharedQKV == false)
            {
                throw new ArgumentException($"Layer '{this.m_name}' is not in shared QKV mode, please call another Perform function with three separated input tensors.");
            }

            using (var g = graph.CreateSubGraph($"{this.m_name}_MultiHeadAttention_SharedQKV"))
            {
                var seqLenQ = inputQ.Rows / batchSize;
                var inputQNorm = this.layerNormQ.Norm(inputQ, g);

                //Input projections
                var scale = 1.0f / (float)(this.m_inputDim);
                IWeightTensor mulQ, mulK, mulV;

                using (var inputQNormView = g.View(inputQNorm, dims: new long[] { 1, inputQ.Rows, inputQ.Columns }))
                {
                    using (var inputQNormViewExp = g.Expand(inputQNormView, dims: new long[] { 3, inputQ.Rows, inputQ.Columns }))
                    {
                        using (var mulQKV = g.MulBatch(inputQNormViewExp, this.QKV, 3, scale))
                        {
                            mulQ = g.Select(mulQKV, 0, 0);
                            mulK = g.Select(mulQKV, 0, 1);
                            mulV = g.Select(mulQKV, 0, 2);
                        }
                    }
                }

                var allQ = g.View(mulQ, dims: new long[] { batchSize, seqLenQ, this.m_multiHeadNum, this.m_d });
                var allK = g.View(mulK, dims: new long[] { batchSize, seqLenQ, this.m_multiHeadNum, this.m_d });
                var allV = g.View(mulV, dims: new long[] { batchSize, seqLenQ, this.m_multiHeadNum, this.m_d });

                //Multi-head attentions
                var Qs = g.View(g.Permute(allQ, 2, 0, 1, 3), dims: new long[] { this.m_multiHeadNum * batchSize, seqLenQ, this.m_d });
                var Ks = g.View(g.Permute(allK, 2, 0, 3, 1), dims: new long[] { this.m_multiHeadNum * batchSize, this.m_d, seqLenQ });
                var Vs = g.View(g.Permute(allV, 2, 0, 1, 3), dims: new long[] { this.m_multiHeadNum * batchSize, seqLenQ, this.m_d });

                // Scaled softmax
                scale = 1.0f / (float)(this.m_d);
                var attn = g.MulBatch(Qs, Ks, this.m_multiHeadNum * batchSize, scale);
                var softmax = g.Softmax(attn, keyMask, inPlace: true);
                var o = g.View(g.MulBatch(softmax, Vs, this.m_multiHeadNum * batchSize), dims: new long[] { this.m_multiHeadNum, batchSize, seqLenQ, this.m_d });

                var W = g.View(g.Permute(o, 1, 2, 0, 3), dims: new long[] { batchSize * seqLenQ, this.m_multiHeadNum * this.m_d });

                // Output projection
                var finalAttResults = g.Dropout(g.Affine(W, this.W0, this.b0), batchSize, this.m_dropoutRatio, inPlace: true);

                return graph.Add(finalAttResults, inputQ);
            }
        }


        public virtual List<IWeightTensor> getParams()
        {
            var response = new List<IWeightTensor>
            {
                this.W0, this.b0
            };

            if (this.m_sharedQKV)
            {
                response.Add(this.QKV);
            }
            else
            {
                response.Add(this.Q);
                response.Add(this.Qb);

                response.Add(this.K);
                response.Add(this.Kb);

                response.Add(this.V);
                response.Add(this.Vb);
            }

            response.AddRange(this.layerNormQ.getParams());

            return response;
        }


        public void Save(Stream stream)
        {
            if (this.m_sharedQKV)
            {
                this.QKV.Save(stream);
            }
            else
            {
                this.Q.Save(stream);
                this.Qb.Save(stream);

                this.K.Save(stream);
                this.Kb.Save(stream);

                this.V.Save(stream);
                this.Vb.Save(stream);
            }

            this.W0.Save(stream);
            this.b0.Save(stream);

            this.layerNormQ.Save(stream);


        }


        public void Load(Stream stream)
        {
            if (this.m_sharedQKV)
            {
                this.QKV.Load(stream);
            }
            else
            {
                this.Q.Load(stream);
                this.Qb.Load(stream);

                this.K.Load(stream);
                this.Kb.Load(stream);

                this.V.Load(stream);
                this.Vb.Load(stream);
            }

            this.W0.Load(stream);
            this.b0.Load(stream);

            this.layerNormQ.Load(stream);
        }
    }
}
