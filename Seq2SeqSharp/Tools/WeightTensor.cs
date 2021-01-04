using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    public enum NormType
    {
        None,
        Uniform,
        Normal
    }

    [Serializable]
    public class WeightTensor : IWeightTensor, IDisposable
    {
        public long[] Sizes { get; set; }

        public int Rows
        {
            get => (int)this.Sizes[0];
            set => this.Sizes[0] = value;
        }
        public int Columns
        {
            get => (int)this.Sizes[1];
            set => this.Sizes[1] = value;
        }

        public string Name { get; set; }
        public bool IsTrainable { get; set; }

        public int DeviceId { get; set; }

        private IAllocator m_allocator;

        private Tensor m_TWeight;
        private Tensor m_TGradient;
        private static readonly object locker = new();

        private bool releasedWeight;
        private bool releasedGradient;
        private IComputeGraph m_computeGraphToBind;

        private string m_GradientSetName = "None";

        public Tensor TWeight
        {
            get
            {
                if (this.releasedWeight)
                {
                    throw new Exception($"The weight '{this.Name}' has been released, you cannot access it.");
                }

                if (this.m_TWeight == null)
                {
                    this.m_TWeight = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
                }

                return this.m_TWeight;
            }
            set
            {
                if (this.m_TWeight != null)
                {
                    throw new Exception($"Please call ReleaseWeight function before assign a new value to weight '{this.Name}'.");
                }

                this.m_TWeight = value;
                this.releasedWeight = false;
            }
        }

        public Tensor TGradient
        {
            get
            {
                if (this.releasedGradient)
                {
                    throw new Exception($"The gradient '{this.Name}' has been released, you cannot access it.");
                }

                if (this.m_TGradient == null)
                {
                    this.m_TGradient = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
                    Ops.Fill(this.m_TGradient, 0.0f);

                    this.m_GradientSetName = "Get";
                }

                return this.m_TGradient;
            }

            set
            {
                if (this.m_TGradient != null)
                {                   
                    throw new Exception($"Please call ReleaseGradient function before assign a new value to gradient '{this.Name}'. This gradient was set by '{this.m_GradientSetName}'");
                }

                this.m_TGradient = value;
                this.releasedGradient = false;
            }
        }


        public WeightTensor(long[] sizes, int deviceId, string name = "", bool isTrainable = false, NormType normal = NormType.None, bool fanIn = false, bool fanOut = false, IComputeGraph graphToBind = null)
        {
            this.Name = name;
            this.DeviceId = deviceId;
            this.IsTrainable = isTrainable;
            this.m_allocator = TensorAllocator.Allocator(this.DeviceId);
            this.Sizes = sizes;

            if (graphToBind != null)
            {
                this.m_computeGraphToBind = graphToBind;
                this.m_computeGraphToBind.Bind(this);
            }

            if (normal == NormType.Uniform)
            {
                var scale = (float)Math.Sqrt(6.0 / (double)(this.Rows + this.Columns));

                if (fanIn && !fanOut)
                {
                    scale = (float)Math.Sqrt(3.0 / (double)this.Rows);
                }
                else if (!fanIn && fanOut)
                {
                    scale = (float)Math.Sqrt(3.0 / (double)this.Columns);
                }

                var w = TensorSharp.RandomGenerator.BuildRandomUniformWeight(this.Sizes, -scale, scale);
                this.SetWeightArray(w);               
            }
            else if (normal == NormType.Normal)
            {
                var w = TensorSharp.RandomGenerator.BuildRandomUniformWeight(this.Sizes, -1.0f, 1.0f);
                this.SetWeightArray(w);
            }
        }

        public WeightTensor(long[] sizes, float c, int deviceId, string name = "", bool isTrainable = false)
        {
            this.Name = name;
            this.DeviceId = deviceId;
            this.IsTrainable = isTrainable;
            this.Sizes = sizes;
            this.m_allocator = TensorAllocator.Allocator(this.DeviceId);

            this.TWeight = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
            Ops.Fill(this.TWeight, c);
        }


        public void UnbindFromComputeGraph()
        {
            if (this.m_computeGraphToBind != null)
            {
                this.m_computeGraphToBind.Unbind(this);
            }
        }

        public int GetDeviceId()
        {
            return this.DeviceId;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new WeightTensor(this.Sizes, deviceId, this.Name, this.IsTrainable);
        }

        public void ZeroGradient()
        {
            Ops.Fill(this.TGradient, 0.0f);
        }

        public void CleanWeight()
        {
            Ops.Fill(this.TWeight, 0.0f);
        }

        public float GetWeightAt(int offset)
        {
            return this.TWeight.GetElementAsFloat(0, offset);
        }

        public void SetWeightAt(float val, int offset)
        {
            this.TWeight.SetElementAsFloat(val, 0, offset);
        }


        public void SetGradientAt(float val, int offset)
        {
            this.TGradient.SetElementAsFloat(val, 0, offset);
        }

        public void SetWeightAtRow(int row, float[] val)
        {
            this.TWeight.SetElementsAsFloat(val, row, 0);
        }

        public void CopyWeightsToGradients(IWeightTensor src)
        {
            var m = src as WeightTensor;

            if (this.m_TGradient != null)
            {
                this.m_TGradient.Dispose();
            }

            this.m_TGradient = m.TWeight.CopyRef();

            this.m_GradientSetName = "CopyWeightsToGradients";
        }

        public void CopyWeightsFrom(IWeightTensor src)
        {
            var m = src as WeightTensor;

            Ops.Copy(this.TWeight, m.TWeight);
        }

        public void AddGradientFrom(IWeightTensor src)
        {
            var m = src as WeightTensor;

            lock (locker)
            {
                var t = new Tensor(this.TGradient.Allocator, DType.Float32, this.Sizes);
                Ops.Copy(t, m.TGradient);
                Ops.Add(this.TGradient, this.TGradient, t);

                t.Dispose();
            }
        }

        public float[] ToWeightArray()
        {
            return this.TWeight.GetElementsAsFloat(this.Rows * this.Columns);
        }

        public float[] ToGradientArray()
        {
            return this.TGradient.GetElementsAsFloat(this.Rows * this.Columns);
        }


        public void AddSoftmaxGradient(WeightTensor src, bool inPlace = false)
        {
            if (this.m_TGradient == null)
            {
                this.m_allocator = TensorAllocator.Allocator(this.DeviceId);
                this.m_TGradient = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
                Ops.SoftmaxGrad(this.m_TGradient, src.TGradient, src.TWeight, false);

                this.m_GradientSetName = "AddSoftmaxGradient";
            }
            else
            {
                Ops.SoftmaxGrad(this.m_TGradient, src.TGradient, src.TWeight, !inPlace);
            }
        }

        public void CopyOrAddGradient(WeightTensor src)
        {
            if (this.m_TGradient == null)
            {
                this.m_allocator = TensorAllocator.Allocator(this.DeviceId);
                this.m_TGradient = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
                Ops.Copy(this.m_TGradient, src.TGradient);

                this.m_GradientSetName = "CopyOrAddGradient_WeightTensor";
            }
            else
            {
                Ops.Add(this.m_TGradient, this.m_TGradient, src.TGradient);
            }
        }

        public void CopyOrAddGradient(Tensor src, string callerName = "")
        {
            if (this.m_TGradient == null)
            {
                this.m_allocator = TensorAllocator.Allocator(this.DeviceId);
                this.m_TGradient = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
                Ops.Copy(this.m_TGradient, src);

                this.m_GradientSetName = $"CopyOrAddGradient_Tensor_CalledBy_{callerName}";
            }
            else
            {
                Ops.Add(this.m_TGradient, this.m_TGradient, src);
            }
        }

        public void AddMulGradient(Tensor w, Tensor g, bool inPlace = false)
        {
            if (this.m_TGradient == null)
            {
                this.m_allocator = TensorAllocator.Allocator(this.DeviceId);
                this.m_TGradient = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
                Ops.Mul(this.m_TGradient, w, g);

                this.m_GradientSetName = "AddMulGrdient";
            }
            else
            {
                if (inPlace)
                {
                    Ops.Mul(this.m_TGradient, w, g);
                }
                else
                {
                    Ops.AddMul(this.m_TGradient, this.m_TGradient, w, g);
                }
            }
        }

        public void AddSigmoidGradient(WeightTensor src)
        {
            if (this.m_TGradient == null)
            {
                this.m_allocator = TensorAllocator.Allocator(this.DeviceId);
                this.m_TGradient = new Tensor(this.m_allocator, DType.Float32, this.Sizes);
                Ops.SigmoidD(this.m_TGradient, src.TWeight, src.TGradient);

                this.m_GradientSetName = "AddSigmoidGradient";
            }
            else
            {
                Ops.AddSigmoidD(this.m_TGradient, this.m_TGradient, src.TWeight, src.TGradient);
            }
        }


        public void AddTanhGradient(WeightTensor src)
        {
            if (this.m_TGradient == null)
            {
                this.m_allocator = TensorAllocator.Allocator(this.DeviceId);
                this.m_TGradient = new Tensor(this.m_allocator, DType.Float32, this.Sizes);

                Ops.TanhD(this.m_TGradient, src.TWeight, src.TGradient);

                this.m_GradientSetName = "AddTanhGradient";
            }
            else
            {
                Ops.AddTanhD(this.m_TGradient, this.m_TGradient, src.TWeight, src.TGradient);
            }
        }

        public List<int> GetTopNMaxWeightIdx(int topN)
        {
            var weights = this.ToWeightArray();
            var q = new FixedSizePriorityQueue<ComparableItem<int>>(topN, new ComparableItemComparer<int>(true));

            for (var i = 0; i < weights.Length; i++)
            {
                q.Enqueue(new ComparableItem<int>(weights[i], i));
            }

            return q.Select(x => x.Value).ToList();
        }

        public void SetWeightArray(float[] v)
        {
            this.TWeight.SetElementsAsFloat(v);
        }

        public void SetGradientArray(float[] v)
        {
            this.TGradient.SetElementsAsFloat(v);
        }

        public WeightTensor CopyWeightsRef(string name)
        {
            var result = new WeightTensor(this.Sizes, this.DeviceId, name)
            {
                m_TWeight = this.m_TWeight.CopyRef()
            };

            return result;
        }

        public void Dispose()
        {
            this.ReleaseWeight();
            this.ReleaseGradient();
        }

        public void ReleaseWeight()
        {
            if (this.m_TWeight != null)
            {
                this.m_TWeight.Dispose();
                this.m_TWeight = null;
                this.releasedWeight = true;
            }
        }

        public void ReleaseGradient()
        {
            if (this.m_TGradient != null)
            {
                this.m_TGradient.Dispose();
                this.m_TGradient = null;
                this.releasedGradient = true;
            }
        }

        public void Save(Stream stream)
        {
            var floatArray1 = this.ToWeightArray();

            // create a byte array and copy the floats into it...
            var byteArray = new byte[floatArray1.Length * 4];
            Buffer.BlockCopy(floatArray1, 0, byteArray, 0, byteArray.Length);

            stream.Write(byteArray, 0, byteArray.Length);
        }

        public void Load(Stream stream)
        {
            var size = this.Rows * this.Columns;
            var byteArray = new byte[size * 4];
            stream.Read(byteArray, 0, byteArray.Length);

            var floatArray2 = new float[byteArray.Length / 4];
            Buffer.BlockCopy(byteArray, 0, floatArray2, 0, byteArray.Length);

            this.SetWeightArray(floatArray2);
        }

        public List<IWeightTensor> GetParams()
        {
            return this.IsTrainable ? new List<IWeightTensor> { this } : new List<IWeightTensor>();
        }
    }
}
