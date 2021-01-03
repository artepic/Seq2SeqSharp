
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{

    [Serializable]
    public class Encoder
    {
        public List<LSTMCell> encoders = new List<LSTMCell>();
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }

        public Encoder(string name, int hdim, int dim, int depth, int deviceId, bool isTrainable)
        {
            this.encoders.Add(new LSTMCell($"{name}.LSTM_0", hdim, dim, deviceId, isTrainable));

            for (var i = 1; i < depth; i++)
            {
                this.encoders.Add(new LSTMCell($"{name}.LSTM_{i}", hdim, hdim, deviceId, isTrainable));

            }
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            foreach (var item in this.encoders)
            {
                item.Reset(weightFactory, batchSize);
            }

        }

        public IWeightTensor Encode(IWeightTensor V, IComputeGraph g)
        {
            foreach (var encoder in this.encoders)
            {
                var e = encoder.Step(V, g);
                V = e;
            }

            return V;
        }


        public List<IWeightTensor> getParams()
        {
            var response = new List<IWeightTensor>();

            foreach (var item in this.encoders)
            {
                response.AddRange(item.getParams());

            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in this.encoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in this.encoders)
            {
                item.Load(stream);
            }
        }
    }
}
