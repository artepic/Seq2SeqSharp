using AdvUtils;
using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class FeedForwardLayer : INeuralUnit
    {
        private readonly IWeightTensor m_Whd;
        private readonly IWeightTensor m_Bd;
        private readonly string m_name;
        private readonly float m_dropoutRatio;
        private readonly int m_inputDim;
        private readonly int m_outputDim;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;

        public FeedForwardLayer(string name, int inputDim, int outputDim, float dropoutRatio, int deviceId, bool isTrainable)
        {
            Logger.WriteLine($"Create feed forward layer '{name}' InputDim = '{inputDim}', OutputDim = '{outputDim}', DropoutRatio = '{dropoutRatio}', DeviceId = '{deviceId}'");

            this.m_name = name;
            this.m_inputDim = inputDim;
            this.m_outputDim = outputDim;
            this.m_dropoutRatio = dropoutRatio;
            this.m_deviceId = deviceId;
            this.m_isTrainable = isTrainable;

            this.m_Whd = new WeightTensor(new long[2] { inputDim, outputDim }, deviceId, $"{name}.{nameof(this.m_Whd)}", normal: NormType.Uniform, isTrainable: isTrainable);
            this.m_Bd = new WeightTensor(new long[2] { 1, outputDim }, 0, deviceId, $"{name}.{nameof(this.m_Bd)}", isTrainable);
        }

        public int GetDeviceId()
        {
            return this.m_deviceId;
        }

        public IWeightTensor Process(IWeightTensor inputT, int batchSize, IComputeGraph g)
        {            
            var res = g.Affine(inputT, this.m_Whd, this.m_Bd);
            return g.Dropout(res, batchSize, this.m_dropoutRatio, true);
        }

        public virtual List<IWeightTensor> GetParams()
        {
            var response = new List<IWeightTensor>
            {
                this.m_Whd, this.m_Bd
            };

            return response;
        }

        public void Save(Stream stream)
        {
            this.m_Whd.Save(stream);
            this.m_Bd.Save(stream);
        }


        public void Load(Stream stream)
        {
            this.m_Whd.Load(stream);
            this.m_Bd.Load(stream);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new FeedForwardLayer(this.m_name, this.m_inputDim, this.m_outputDim, this.m_dropoutRatio, deviceId, this.m_isTrainable);
        }
    }
}
