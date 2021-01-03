using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace Seq2SeqSharp
{

    public class AttentionPreProcessResult
    {
        public IWeightTensor Uhs;
        public IWeightTensor encOutput;
    }

    [Serializable]
    public class AttentionUnit : INeuralUnit
    {
        private readonly IWeightTensor m_V;
        private readonly IWeightTensor m_Ua;
        private readonly IWeightTensor m_bUa;
        private readonly IWeightTensor m_Wa;
        private readonly IWeightTensor m_bWa;

        private readonly string m_name;
        private readonly int m_hiddenDim;
        private readonly int m_contextDim;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;

        private bool m_enableCoverageModel;
        private readonly IWeightTensor m_Wc;
        private readonly IWeightTensor m_bWc;
        private readonly LSTMCell m_coverage;

        private readonly int k_coverageModelDim = 16;

        public AttentionUnit(string name, int hiddenDim, int contextDim, int deviceId, bool enableCoverageModel, bool isTrainable)
        {
            this.m_name = name;
            this.m_hiddenDim = hiddenDim;
            this.m_contextDim = contextDim;
            this.m_deviceId = deviceId;
            this.m_enableCoverageModel = enableCoverageModel;
            this.m_isTrainable = isTrainable;

            Logger.WriteLine($"Creating attention unit '{name}' HiddenDim = '{hiddenDim}', ContextDim = '{contextDim}', DeviceId = '{deviceId}', EnableCoverageModel = '{enableCoverageModel}'");

            this.m_Ua = new WeightTensor(new long[2] { contextDim, hiddenDim }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(this.m_Ua)}", isTrainable: isTrainable);
            this.m_Wa = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(this.m_Wa)}", isTrainable: isTrainable);
            this.m_bUa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(this.m_bUa)}", isTrainable: isTrainable);
            this.m_bWa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(this.m_bWa)}", isTrainable: isTrainable);
            this.m_V = new WeightTensor(new long[2] { hiddenDim, 1 }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(this.m_V)}", isTrainable: isTrainable);

            if (this.m_enableCoverageModel)
            {
                this.m_Wc = new WeightTensor(new long[2] { this.k_coverageModelDim, hiddenDim }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(this.m_Wc)}", isTrainable: isTrainable);
                this.m_bWc = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(this.m_bWc)}", isTrainable: isTrainable);
                this.m_coverage = new LSTMCell(name: $"{name}.{nameof(this.m_coverage)}", hdim: this.k_coverageModelDim, dim: 1 + contextDim + hiddenDim, deviceId: deviceId, isTrainable: isTrainable);
            }
        }

        public int GetDeviceId()
        {
            return this.m_deviceId;
        }

        public AttentionPreProcessResult PreProcess(IWeightTensor encOutput, int batchSize, IComputeGraph g)
        {
            var srcSeqLen = encOutput.Rows / batchSize;

            var r = new AttentionPreProcessResult
            {
                encOutput = encOutput
            };

            r.Uhs = g.Affine(r.encOutput, this.m_Ua, this.m_bUa);
            r.Uhs = g.View(r.Uhs, dims: new long[] { batchSize, srcSeqLen, -1 });


            if (this.m_enableCoverageModel)
            {
                this.m_coverage.Reset(g.GetWeightFactory(), r.encOutput.Rows);
            }

            return r;
        }

        public IWeightTensor Perform(IWeightTensor state, AttentionPreProcessResult attnPre, int batchSize, IComputeGraph graph)
        {
            var srcSeqLen = attnPre.encOutput.Rows / batchSize;

            using (var g = graph.CreateSubGraph(this.m_name))
            {
                // Affine decoder state
                var wc = g.Affine(state, this.m_Wa, this.m_bWa);

                // Expand dims from [batchSize x decoder_dim] to [batchSize x srcSeqLen x decoder_dim]
                var wc1 = g.View(wc, dims: new long[] { batchSize, 1, wc.Columns });
                var wcExp = g.Expand(wc1, dims: new long[] { batchSize, srcSeqLen, wc.Columns });

                IWeightTensor ggs = null;
                if (this.m_enableCoverageModel)
                {
                    // Get coverage model status at {t-1}
                    var wCoverage = g.Affine(this.m_coverage.Hidden, this.m_Wc, this.m_bWc);
                    var wCoverage1 = g.View(wCoverage, dims: new long[] { batchSize, srcSeqLen, -1 });

                    ggs = g.AddTanh(attnPre.Uhs, wcExp, wCoverage1);
                }
                else
                {
                    ggs = g.AddTanh(attnPre.Uhs, wcExp);
                }

                var ggss = g.View(ggs, dims: new long[] { batchSize * srcSeqLen, -1 });
                var atten = g.Mul(ggss, this.m_V);

                var attenT = g.Transpose(atten);
                var attenT2 = g.View(attenT, dims: new long[] { batchSize, srcSeqLen });

                var attenSoftmax1 = g.Softmax(attenT2, inPlace: true);

                var attenSoftmax = g.View(attenSoftmax1, dims: new long[] { batchSize, 1, srcSeqLen });
                var inputs2 = g.View(attnPre.encOutput, dims: new long[] { batchSize, srcSeqLen, attnPre.encOutput.Columns });

                var contexts = graph.MulBatch(attenSoftmax, inputs2, batchSize);

                contexts = graph.View(contexts, dims: new long[] { batchSize, attnPre.encOutput.Columns });

                if (this.m_enableCoverageModel)
                {
                    // Concatenate tensor as input for coverage model
                    var aCoverage = g.View(attenSoftmax1, dims: new long[] { attnPre.encOutput.Rows, 1 });


                    var state2 = g.View(state, dims: new long[] { batchSize, 1, state.Columns });
                    var state3 = g.Expand(state2, dims: new long[] { batchSize, srcSeqLen, state.Columns });
                    var state4 = g.View(state3, dims: new long[] { batchSize * srcSeqLen, -1 });


                    var concate = g.ConcatColumns(aCoverage, attnPre.encOutput, state4);
                    this.m_coverage.Step(concate, graph);
                }


                return contexts;
            }
        }


        public virtual List<IWeightTensor> GetParams()
        {
            var response = new List<IWeightTensor>
            {
                this.m_Ua,
                this.m_Wa,
                this.m_bUa,
                this.m_bWa,
                this.m_V
            };

            if (this.m_enableCoverageModel)
            {
                response.Add(this.m_Wc);
                response.Add(this.m_bWc);
                response.AddRange(this.m_coverage.getParams());
            }

            return response;
        }

        public void Save(Stream stream)
        {
            this.m_Ua.Save(stream);
            this.m_Wa.Save(stream);
            this.m_bUa.Save(stream);
            this.m_bWa.Save(stream);
            this.m_V.Save(stream);

            if (this.m_enableCoverageModel)
            {
                this.m_Wc.Save(stream);
                this.m_bWc.Save(stream);
                this.m_coverage.Save(stream);
            }
        }


        public void Load(Stream stream)
        {
            this.m_Ua.Load(stream);
            this.m_Wa.Load(stream);
            this.m_bUa.Load(stream);
            this.m_bWa.Load(stream);
            this.m_V.Load(stream);

            if (this.m_enableCoverageModel)
            {
                this.m_Wc.Load(stream);
                this.m_bWc.Load(stream);
                this.m_coverage.Load(stream);
            }
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            var a = new AttentionUnit(this.m_name, this.m_hiddenDim, this.m_contextDim, deviceId, this.m_enableCoverageModel, this.m_isTrainable);
            return a;
        }
    }
}



