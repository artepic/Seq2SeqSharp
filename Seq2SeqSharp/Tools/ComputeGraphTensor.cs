//using Microsoft.Msagl.Drawing;
//using Microsoft.Msagl.Layout.Incremental;
//using Microsoft.Msagl.Layout.Layered;
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using TensorSharp;

/// <summary>
/// Tensor based computing graph written by Zhongkai Fu.
/// The graph includes the following key features.
/// #1. Include several key operations for neural network.
/// #2. Support both CPU and GPU (CUDA)
/// #3. Support automatic differentiation and back propagation
/// #4. Support networks (operations) visualization
/// #5. and so on...
/// </summary>
namespace Seq2SeqSharp.Tools
{
    public class ConcurrentList<T>
    {
        private const int MaxSize = 1024000;
        private T[] array;
        private int count;
        public int Count => this.count;

        public T this[int key]
        {
            get => this.array[key];
            set => this.array[key] = value;
        }

        public ConcurrentList()
        {
            this.array = new T[MaxSize];
        }

        public void Add(T item)
        {
            var n = System.Threading.Interlocked.Increment(ref this.count);
            this.array[n - 1] = item;
        }

        public void RemoveLastItem()
        {
            System.Threading.Interlocked.Decrement(ref this.count);
        }

        private readonly object locker = new();
        public void Clear()
        {
            lock (this.locker)
            {
                this.count = 0;
                this.array = new T[MaxSize];
            }
        }
    }

    public class ComputeGraphTensor : IComputeGraph
    {
        private readonly WeightTensorFactory m_weightTensorFactory;
        private readonly ConcurrentList<Action> m_backprop;
        private readonly bool m_needsBackprop;
      //  private readonly bool m_visNeuralNetwork;
        private readonly int m_deviceId;
        private readonly bool m_isSubGraph;

        // Visualization for neural network
       // private Microsoft.Msagl.Drawing.Graph m_opsViz;
       // private HashSet<string> m_setEdges;
        //private Microsoft.Msagl.Drawing.Subgraph m_subGraph = null;
       // private Dictionary<string, Microsoft.Msagl.Drawing.Subgraph> m_name2SubGraph = null;

        private List<IWeightTensor> m_tensorsBindToCurrentGraph;

        public ComputeGraphTensor(IWeightFactory weightFactory, int deviceId, bool needBack = true, ConcurrentList<Action> backprop = null, bool isSubGraph = false)
        {
            this.m_backprop = backprop != null ? backprop : new ConcurrentList<Action>();
            this.m_weightTensorFactory = weightFactory as WeightTensorFactory;
            this.m_needsBackprop = needBack;
            this.m_deviceId = deviceId;
            //m_visNeuralNetwork = visNetwork;
            this.m_isSubGraph = isSubGraph;

            //m_name2SubGraph = new Dictionary<string, Subgraph>();
            //if (m_visNeuralNetwork)
            //{
            //    // Initialize parameters for neural network visualization
            //    m_opsViz = new Microsoft.Msagl.Drawing.Graph();
            //    m_setEdges = new HashSet<string>();
            //}

            this.m_tensorsBindToCurrentGraph = new List<IWeightTensor>();
        }

        public IWeightFactory GetWeightFactory()
        {
            return this.m_weightTensorFactory;
        }

        public IComputeGraph CreateSubGraph(string name)
        {
            var subGraph = new ComputeGraphTensor(this.m_weightTensorFactory, this.m_deviceId, this.m_needsBackprop, this.m_backprop, true);
            //if (m_visNeuralNetwork)
            //{
            //    // Create parameters for neural network visualization
            //    subGraph.m_opsViz = m_opsViz;
            //    subGraph.m_setEdges = m_setEdges;
            //    subGraph.m_name2SubGraph = m_name2SubGraph;
            //    if (m_name2SubGraph.ContainsKey(name) == false)
            //    {
            //        int index = name.LastIndexOf(".");
            //        subGraph.m_subGraph = new Subgraph(name)
            //        {
            //            LabelText = name.Substring(index + 1)
            //        };

            //        m_name2SubGraph.Add(name, subGraph.m_subGraph);

            //        if (m_subGraph == null)
            //        {
            //            m_opsViz.RootSubgraph.AddSubgraph(subGraph.m_subGraph);
            //        }
            //        else
            //        {
            //            m_subGraph.AddSubgraph(subGraph.m_subGraph);
            //        }
            //    }
            //    else
            //    {
            //        subGraph.m_subGraph = m_name2SubGraph[name];
            //    }
            //}

            return subGraph;
        }

        public void Backward()
        {
            for (var i = this.m_backprop.Count - 1; i >= 0; i--)
            {
                this.m_backprop[i](); // tick!
            }

            this.m_backprop.Clear();
        }

        public void RunTopBackward()
        {
            if (this.m_needsBackprop)
            {
                this.m_backprop[this.m_backprop.Count - 1]();
                this.m_backprop.RemoveLastItem();
            }
        }

        public IWeightTensor Sigmoid(IWeightTensor w)
        {
            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m.Sizes, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.Sigmoid");
            this.VisualizeNodes(w, res);

            Ops.Sigmoid(res.TWeight, m.TWeight);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    m.AddSigmoidGradient(res);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m1.Sizes, this.m_deviceId, name: $"{this.GetHashString(w1.Name, w2.Name)}.AddTanh");
            this.VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.AddTanh(res.TWeight, m1.TWeight, m2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    m1.AddTanhGradient(res);
                    m2.AddTanhGradient(res);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var m3 = w3 as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m1.Sizes, this.m_deviceId, name: $"{this.GetHashString(w1.Name, w2.Name, w3.Name)}.AddTanh");
            this.VisualizeNodes(new IWeightTensor[] { w1, w2, w3 }, res);

            Ops.AddTanh3(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    m1.AddTanhGradient(res);
                    m2.AddTanhGradient(res);
                    m3.AddTanhGradient(res);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor Mul(IWeightTensor w, float v)
        {
            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m.Sizes, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.MulV", graphToBind: this);
            this.VisualizeNodes(w, res);

            Ops.Mul(res.TWeight, m.TWeight, v);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    Ops.AddMulV(m.TGradient, m.TGradient, res.TGradient, v);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        /// <summary>
        /// Operation r = w1 + w2 * v
        /// </summary>
        /// <param name="w1"></param>
        /// <param name="w2"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public IWeightTensor AddMul(IWeightTensor w1, IWeightTensor w2, float v, bool runGradientW1 = true, bool runGradientW2 = true)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;

            var res = this.m_weightTensorFactory.CreateWeightTensor(m1.Sizes, this.m_deviceId, name: $"{this.GetHashString(w1.Name, w2.Name)}.AddMulV", graphToBind: this);
            this.VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.AddMulV(res.TWeight, m1.TWeight, m2.TWeight, v);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    if (runGradientW1)
                    {
                        m1.CopyOrAddGradient(res);
                    }

                    if (runGradientW2)
                    {
                        Ops.AddMulV(m2.TGradient, m2.TGradient, res.TGradient, v);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }



        public IWeightTensor MaskFill(IWeightTensor w1, IWeightTensor m, float v)
        {
            var m1 = w1 as WeightTensor;
            var mask = m as WeightTensor;

            var res = this.m_weightTensorFactory.CreateWeightTensor(m1.Sizes, this.m_deviceId, name: $"{this.GetHashString(w1.Name, m.Name)}.MaskFill", graphToBind: this);
            this.VisualizeNodes(new IWeightTensor[] { w1, m }, res);


            Ops.MaskFill(res.TWeight, m1.TWeight, mask.TWeight, v);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var tGrad = Ops.MaskFill(null, res.TGradient, mask.TWeight, 0.0f))
                    {
                        m1.CopyOrAddGradient(tGrad);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public void Bind(IWeightTensor w)
        {
            this.m_tensorsBindToCurrentGraph.Add(w);
        }

        public void Unbind(IWeightTensor w)
        {
            this.m_tensorsBindToCurrentGraph.Remove(w);

        }

        /// <summary>
        /// Result = w1 * w2 + w3 * w4
        /// </summary>
        /// <param name="w1"></param>
        /// <param name="w2"></param>
        /// <param name="w3"></param>
        /// <param name="w4"></param>
        /// <returns></returns>
        public IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var m3 = w3 as WeightTensor;
            var m4 = w4 as WeightTensor;

            var res = this.m_weightTensorFactory.CreateWeightTensor(m1.Sizes, this.m_deviceId, name: $"{this.GetHashString(w1.Name, w2.Name, w3.Name, w4.Name)}.EltMulMulAdd", graphToBind: this);
            this.VisualizeNodes(new IWeightTensor[] { w1, w2, w3, w4 }, res);

            Ops.MulMulAdd(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight, m4.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    m1.AddMulGradient(m2.TWeight, res.TGradient);
                    m2.AddMulGradient(m1.TWeight, res.TGradient);

                    m3.AddMulGradient(m4.TWeight, res.TGradient);
                    m4.AddMulGradient(m3.TWeight, res.TGradient);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);

                // These tensors' weights will be used during back-propogation, so we unbind them from the computing graph
                m1.UnbindFromComputeGraph();
                m2.UnbindFromComputeGraph();
                m3.UnbindFromComputeGraph();
                m4.UnbindFromComputeGraph();
            }


            return res;
        }

        public IWeightTensor EltMul(IWeightTensor w1, IWeightTensor w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m1.Sizes, this.m_deviceId, name: $"{this.GetHashString(w1.Name, w2.Name)}.EltMul", graphToBind: this);
            this.VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.Mul(res.TWeight, m1.TWeight, m2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    m1.AddMulGradient(m2.TWeight, res.TGradient);
                    m2.AddMulGradient(m1.TWeight, res.TGradient);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);

                m1.UnbindFromComputeGraph();
                m2.UnbindFromComputeGraph();
            }

            return res;
        }

        public float UpdateCost(IWeightTensor m, int[] ids)
        {
            var t = m as WeightTensor;

            using (var idsTensor = new Tensor(TensorAllocator.Allocator(this.m_deviceId), DType.Int32, 1, ids.Length))
            {
                idsTensor.SetElementsAsInt(ids);
                using (var costs = Ops.UpdateCost(null, t.TWeight, idsTensor))
                {
                    return Ops.SumAll(costs);
                }
            }
        }


        public IWeightTensor Add(IWeightTensor w1, IWeightTensor w2, bool runGradient1 = true, bool runGradient2 = true)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m1.Sizes, this.m_deviceId, name: $"{this.GetHashString(w1.Name, w2.Name)}.Add", graphToBind: this);

            this.VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);


            Ops.Add(res.TWeight, m1.TWeight, m2.TWeight);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    if (runGradient1)
                    {
                        m1.CopyOrAddGradient(res);
                    }

                    if (runGradient2)
                    {
                        m2.CopyOrAddGradient(res);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Tanh(IWeightTensor w)
        {
            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m.Sizes, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.Tanh");
            this.VisualizeNodes(w, res);

            Ops.Tanh(res.TWeight, m.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    m.AddTanhGradient(res);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor Relu(IWeightTensor w, bool inPlace = false)
        {
            var m = w as WeightTensor;
            WeightTensor res = null;
            res = inPlace ? m.CopyWeightsRef($"{this.GetHashString(w.Name)}.Relu") : this.m_weightTensorFactory.CreateWeightTensor(m.Sizes, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.Relu", graphToBind: this);

            this.VisualizeNodes(w, res);


            Ops.Relu(res.TWeight, m.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    if (inPlace)
                    {
                        m.TGradient = res.TGradient.CopyRef();
                        Ops.ReluD(m.TGradient, m.TWeight, m.TGradient);
                    }
                    else
                    {
                        Ops.AddReluD(m.TGradient, m.TGradient, m.TWeight, res.TGradient);
                    }
                    res.Dispose();
                };
                this.m_backprop.Add(backward);

                m.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor MulBatch(IWeightTensor m1, IWeightTensor m2, int batchSize, float alpha = 1.0f)
        {
            var t1 = m1 as WeightTensor;
            var t2 = m2 as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(new long[] { batchSize, t1.TWeight.Sizes[1], t2.TWeight.Sizes[2] }, this.m_deviceId, name: $"{this.GetHashString(m1.Name, m2.Name)}.MulBatch", graphToBind: this);
            this.VisualizeNodes(new IWeightTensor[] { m1, m2 }, res);

            var t1W = t1.TWeight;
            var t2W = t2.TWeight;

            Ops.AddmmBatch(res.TWeight, 0.0f, res.TWeight, alpha, t1W, t2W);


            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var tW2 = t2W.Transpose(1, 2))
                    {
                        Ops.AddmmBatch(t1.TGradient, 1.0f, t1.TGradient, alpha, res.TGradient, tW2);
                    }

                    using (var tW1 = t1W.Transpose(1, 2))
                    {
                        Ops.AddmmBatch(t2.TGradient, 1.0f, t2.TGradient, alpha, tW1, res.TGradient);
                    }

                    res.Dispose();

                };
                this.m_backprop.Add(backward);

                t1.UnbindFromComputeGraph();
                t2.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor Mul(IWeightTensor m1, IWeightTensor m2)
        {
            var t1 = m1 as WeightTensor;
            var t2 = m2 as WeightTensor;
            var n = t1.Rows;
            var d = t2.Columns;
            WeightTensor res;

            res = this.m_weightTensorFactory.CreateWeightTensor(n, d, this.m_deviceId, name: $"{this.GetHashString(m1.Name, m2.Name)}.Mul", graphToBind: this);
            this.VisualizeNodes(new IWeightTensor[] { m1, m2 }, res);

            Ops.Addmm(res.TWeight, 0.0f, res.TWeight, 1.0f, t1.TWeight, t2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var tW2 = t2.TWeight.Transpose())
                    {
                        Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);
                    }

                    using (var tW1 = t1.TWeight.Transpose())
                    {
                        Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);

                t1.UnbindFromComputeGraph();
                t2.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor Affine(IWeightTensor m1, IWeightTensor m2, IWeightTensor mbias, float alpha = 1.0f)
        {
            if (m1 == null)
            {
                throw new ArgumentNullException($"m1 tensor is null");
            }

            if (m2 == null)
            {
                throw new ArgumentNullException($"m2 tensor is null");
            }

            if (mbias == null)
            {
                throw new ArgumentNullException($"mbias tensor is null");
            }

            var t1 = m1 as WeightTensor;
            var t2 = m2 as WeightTensor;
            var t3 = mbias as WeightTensor;

            var n = t1.Rows;
            var d = t2.Columns;
            var res = this.m_weightTensorFactory.CreateWeightTensor(n, d, this.m_deviceId, name: $"{this.GetHashString(m1.Name, m2.Name, mbias.Name)}.Affine", graphToBind: this);
            this.VisualizeNodes(new IWeightTensor[] { m1, m2, mbias }, res);

            using (var t3WExp = t3.TWeight.Expand(n, d))
            {
                Ops.Addmm(res.TWeight, 1.0f, t3WExp, alpha, t1.TWeight, t2.TWeight);
            }

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var t3G = t3.TGradient.Expand(n, d))
                    {
                        Ops.Add(t3G, t3G, res.TGradient);
                    }

                    using (var tW2 = t2.TWeight.Transpose())
                    {
                        Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, alpha, res.TGradient, tW2);
                    }

                    using (var tW1 = t1.TWeight.Transpose())
                    {
                        Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, alpha, tW1, res.TGradient);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);

                t1.UnbindFromComputeGraph();
                t2.UnbindFromComputeGraph();
            }

            return res;

        }

        public IWeightTensor Transpose(IWeightTensor w)
        {
            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m.Columns, m.Rows, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.Transpose", graphToBind: this);
            this.VisualizeNodes(w, res);

            res.TWeight = m.TWeight.Transpose();
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();
                    using (var gT = res.TGradient.Transpose())
                    {
                        m.CopyOrAddGradient(gT, res.Name);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public int[] Argmax(IWeightTensor w, int dim)
        {
            int[] idx = null;
            var m = w as WeightTensor;
            using (var argMaxT = Ops.Argmax(null, m.TWeight, dim))
            {
                var res = new float[argMaxT.ElementCount()];
                argMaxT.CopyToArray(res);

                idx = new int[res.Length];
                for (var i = 0; i < res.Length; i++)
                {
                    idx[i] = (int)res[i];
                }
            }

            return idx;
        }

      
        public IWeightTensor Softmax(IWeightTensor w, IWeightTensor mask = null, bool runGradients = true, bool inPlace = false)
        {
            var t = w as WeightTensor;
            WeightTensor res = null;

            res = inPlace ? t.CopyWeightsRef($"{this.GetHashString(w.Name)}.SoftmaxMask") : this.m_weightTensorFactory.CreateWeightTensor(t.Sizes, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.SoftmaxMask");

            this.VisualizeNodes(w, res);

            if (mask != null)
            {
                var m = mask as WeightTensor;
                Ops.SoftmaxMask(res.TWeight, t.TWeight, m.TWeight);
            }
            else
            {
                Ops.Softmax(res.TWeight, t.TWeight);
            }
            
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    if (runGradients)
                    {
                        if (inPlace)
                        {
                            t.TGradient = res.TGradient.CopyRef();
                        }
                        t.AddSoftmaxGradient(res, inPlace);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor PeekRow(IWeightTensor w, int ix, int num = 1, bool runGradients = true)
        {
            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(num, m.Columns, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.PeekRow", graphToBind: this);
            res.TWeight = m.TWeight.Narrow(0, ix, num);
            res.TGradient = runGradients ? m.TGradient.Narrow(0, ix, num) : null;

            this.VisualizeNodes(w, res);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Select(IWeightTensor w, int dim, int index)
        {
            var m = w as WeightTensor;

            var selWeights = m.TWeight.Select(dim, index);

            var res = this.m_weightTensorFactory.CreateWeightTensor(selWeights.Sizes, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.Select", graphToBind: this);
            res.TWeight = selWeights;

            this.VisualizeNodes(w, res);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var mGrad = m.TGradient.Select(dim, index))
                    {
                        Ops.Add(mGrad, mGrad, res.TGradient);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        private byte[] GetHash(string inputString)
        {
            HashAlgorithm algorithm = SHA256.Create();
            return algorithm.ComputeHash(Encoding.UTF8.GetBytes(inputString));
        }

        private string GetHashString(params string[] inputStrings)
        {
            //if (m_visNeuralNetwork)
            //{
            //    string inputString = string.Join("_", inputStrings);
            //    StringBuilder sb = new StringBuilder();
            //    foreach (byte b in GetHash(inputString))
            //    {
            //        sb.Append(b.ToString("X2"));
            //    }

            //    return sb.ToString();
            //}
            return string.Empty;
        }

        private void VisualizeNodes(IWeightTensor sourceNode, IWeightTensor targetNode)
        {
            this.VisualizeNodes(new IWeightTensor[] { sourceNode }, targetNode);
        }

        private void VisualizeNodes(IEnumerable<IWeightTensor> sourceNodes, IWeightTensor targetNode)
        {
            //if (!m_visNeuralNetwork || m_deviceId != 0)
            //{
            //    return;
            //}

            //// Create node for target tensor
            //int index = targetNode.Name.LastIndexOf('.');
            //Microsoft.Msagl.Drawing.Node tgtNode = m_opsViz.AddNode(targetNode.Name);
            //tgtNode.LabelText = targetNode.Name.Substring(index + 1);

            //if (targetNode.IsTrainable)
            //{
            //    tgtNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;
            //}

            //if (m_subGraph != null)
            //{
            //    // Current compute graph is a sub-graph
            //    m_subGraph.AddNode(tgtNode);
            //}

            //// Create edges for each source node and target node
            //foreach (IWeightTensor sourceNode in sourceNodes)
            //{
            //    if (!string.IsNullOrEmpty(sourceNode.Name) && !string.IsNullOrEmpty(targetNode.Name))
            //    {
            //        string key = $"{sourceNode.Name}->{targetNode.Name}";
            //        if (m_setEdges.Contains(key))
            //        {
            //            continue;
            //        }

            //        int srcIndex = sourceNode.Name.LastIndexOf('.');
            //        Microsoft.Msagl.Drawing.Node srcNode = m_opsViz.AddNode(sourceNode.Name);
            //        srcNode.LabelText = sourceNode.Name.Substring(srcIndex + 1);
            //        if (sourceNode.IsTrainable)
            //        {
            //            srcNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;

            //            if (m_subGraph != null)
            //            {
            //                m_subGraph.AddNode(srcNode);
            //            }
            //        }

            //        Edge edge = m_opsViz.AddEdge(sourceNode.Name, targetNode.Name);

            //        m_setEdges.Add(key);
            //    }
            //}
        }

        public void VisualizeNeuralNetToFile(string neuralNetPicFilePath)
        {
            //FastIncrementalLayoutSettings fastSettings = new FastIncrementalLayoutSettings
            //{
            //    AvoidOverlaps = true,
            //    NodeSeparation = 30,
            //    RouteEdges = true
            //};

            //SugiyamaLayoutSettings settings = new SugiyamaLayoutSettings
            //{
            //    FallbackLayoutSettings = fastSettings
            //};

            //m_opsViz.LayoutAlgorithmSettings = settings;

            //Microsoft.Msagl.GraphViewerGdi.GraphRenderer renderer = new Microsoft.Msagl.GraphViewerGdi.GraphRenderer(m_opsViz);
            //renderer.CalculateLayout();

            //System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap((int)m_opsViz.Width, (int)m_opsViz.Height, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
            //renderer.Render(bitmap);

            //bitmap.Save(neuralNetPicFilePath);

            //bitmap.Dispose();
        }


        public IWeightTensor RepeatRows(IWeightTensor w, int n, bool runGradient = true)
        {
            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m.Rows * n, m.Columns, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.RepeatRows", graphToBind: this);
            this.VisualizeNodes(w, res);

            res.TWeight = m.TWeight.RepeatTensor(n, 1);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    if (runGradient)
                    {
                        res.ReleaseWeight();
                        for (var i = 0; i < n; i++)
                        {
                            using (var resG_i = res.TGradient.Narrow(0, m.Rows * i, m.Rows))
                            {
                                m.CopyOrAddGradient(resG_i, res.Name);
                            }
                        }
                    }
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor ConcatRows(List<IWeightTensor> wl)
        {
            if (wl.Count == 1)
            {
                return wl[0];
            }

            var wlNameList = new List<string>();
            var twl = new List<Tensor>();
            var sx = 0;
            var sy = 0;
            foreach (var item in wl)
            {
                var m = item as WeightTensor;
                sx += m.Rows;
                sy = m.Columns;

                twl.Add(m.TWeight);
                wlNameList.Add(item.Name);
            }

            var wlName = string.Join("_", wlNameList);
            var res = this.m_weightTensorFactory.CreateWeightTensor(sx, sy, this.m_deviceId, name: $"{this.GetHashString(wlName)}.ConcatRows", graphToBind: this);
            this.VisualizeNodes(wl, res);

            Ops.Concat(res.TWeight, 0, twl.ToArray());

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    sx = 0;
                    foreach (var item in wl)
                    {
                        var m = item as WeightTensor;
                        using (var tTmp = res.TGradient.Narrow(0, sx, m.Rows))
                        {
                            m.CopyOrAddGradient(tTmp, res.Name);
                            sx += m.Rows;
                        }
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor TransposeBatch(IWeightTensor m, int batchSize)
        {
            var t = m as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(t.Sizes, this.m_deviceId, name: $"{this.GetHashString(m.Name)}.TransposeBatch", graphToBind: this);
            this.VisualizeNodes(m, res);

            var sizeEveryBatch = m.Rows / batchSize;
            using (var tWView = t.TWeight.View(sizeEveryBatch, batchSize, m.Columns))
            {
                using (var tWViewPermute = tWView.Permute(1, 0, 2))
                {
                    using (var tW2 = Ops.AsContiguous(tWViewPermute))
                    {
                        res.TWeight = tW2.View(m.Rows, m.Columns);
                    }
                }
            }

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var g = t.TGradient.View(sizeEveryBatch, batchSize, m.Columns))
                    {
                        using (var t2 = res.TGradient.View(batchSize, sizeEveryBatch, m.Columns))
                        {
                            using (var t2Permute = t2.Permute(1, 0, 2))
                            {
                                Ops.Add(g, g, t2Permute);
                            }
                        }
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor ConcatColumns(params IWeightTensor[] wl)
        {
            if (wl.Length == 1)
            {
                return wl[0];
            }

            var srcNameList = new List<string>();
            var twl = new List<Tensor>();
            var sx = 0;
            var sy = 0;

            foreach (var item in wl)
            {
                var m = item as WeightTensor;
                sx = m.Rows;
                sy += m.Columns;

                twl.Add(m.TWeight);
                srcNameList.Add(item.Name);
            }

            var srcNames = string.Join("_", srcNameList);
            var res = this.m_weightTensorFactory.CreateWeightTensor(sx, sy, this.m_deviceId, name: $"{this.GetHashString(srcNames)}.ConcatColumns", graphToBind: this);
            this.VisualizeNodes(wl, res);

            Ops.Concat(res.TWeight, 1, twl.ToArray());
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    sy = 0;
                    foreach (var item in wl)
                    {
                        var m = item as WeightTensor;
                        using (var tTmp = res.TGradient.Narrow(1, sy, m.Columns))
                        {
                            m.CopyOrAddGradient(tTmp, res.Name);
                            sy += m.Columns;
                        }
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public List<IWeightTensor> SplitColumns2(IWeightTensor w, params int[] sizes)
        {
            var m = w as WeightTensor;
            var resList = new List<IWeightTensor>();

            var x = 0;
            foreach (var size in sizes)
            {
                var res = this.m_weightTensorFactory.CreateWeightTensor(m.Rows, size, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.SplitColumn", graphToBind: this);
                this.VisualizeNodes(w, res);

                res.TWeight = m.TWeight.Narrow(1, x, size);
                resList.Add(res);

                x += size;
            }


            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    x = 0;
                    var i = 0;
                    foreach (var item in resList)
                    {
                        var item_i = item as WeightTensor;
                        using (var mG = m.TGradient.Narrow(1, x, sizes[i]))
                        {
                            Ops.Add(mG, mG, item_i.TGradient);
                        }

                        item.Dispose();

                        x += sizes[i];
                        i++;
                    }
                };
                this.m_backprop.Add(backward);
            }


            return resList;
        }

        public IWeightTensor Permute(IWeightTensor w, params int[] dims)
        {
            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(m.Sizes, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.Permute", graphToBind: this);
            this.VisualizeNodes(w, res);

            res.TWeight = m.TWeight.Permute(dims);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var gT = m.TGradient.Permute(dims))
                    {
                        Ops.Add(gT, gT, res.TGradient);
                    }
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor View(IWeightTensor w, bool runGradient = true, params long[] dims)
        {
            var hasNegOne = false;
            var negOneIdx = 0;
            long totalGivenSize = 1;
            for (var i = 0; i < dims.Length; i++)
            {
                var dim = dims[i];
                if (dim == -1)
                {
                    if (hasNegOne)
                    {
                        throw new ArgumentException($"View operation only allows single -1 in dims.");
                    }

                    hasNegOne = true;
                    negOneIdx = i;
                }
                else
                {
                    totalGivenSize *= dim;
                }
            }

            if (hasNegOne)
            {
                long totalSrcSize = 1;
                foreach (int size in w.Sizes)
                {
                    totalSrcSize *= size;
                }

                dims[negOneIdx] = totalSrcSize / totalGivenSize;
            }


            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(dims, this.m_deviceId, name: w.Name, graphToBind: this);
            //  VisualizeNodes(w, res);


            var congtiW = Ops.AsContiguous(m.TWeight);
            m.ReleaseWeight();
            m.TWeight = congtiW;

            res.TWeight = congtiW.View(dims);


            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    if (runGradient)
                    {
                        res.ReleaseWeight();
                        using (var resG = res.TGradient.View(m.Sizes))
                        {
                            m.CopyOrAddGradient(resG, res.Name);
                        }
                    }
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor Expand(IWeightTensor w, bool runGradient = true, params long[] dims)
        {

            var m = w as WeightTensor;
            var res = this.m_weightTensorFactory.CreateWeightTensor(dims, this.m_deviceId, name: $"{this.GetHashString(w.Name)}.Expand", graphToBind: this);
            this.VisualizeNodes(w, res);

            res.TWeight = m.TWeight.Expand(dims);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    if (runGradient)
                    {
                        res.ReleaseWeight();
                        using (var mGExp = m.TGradient.Expand(dims))
                        {
                            Ops.Add(mGExp, mGExp, res.TGradient);
                        }
                    }
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public (IWeightTensor r1, IWeightTensor r2) SplitColumns(IWeightTensor w, int size1, int size2)
        {
            var res = this.SplitColumns2(w, size1, size2);

            return (res[0], res[1]);
        }

        public (IWeightTensor r1, IWeightTensor r2, IWeightTensor r3) SplitColumns(IWeightTensor w, int size1, int size2, int size3)
        {
            var res = this.SplitColumns2(w, size1, size2, size3);

            return (res[0], res[1], res[2]);
        }

        private Tensor BuildRandomTensor(int rows, int columns, int batchSize, float prob)
        {
            using (var noise = new Tensor(TensorAllocator.Allocator(this.m_deviceId), DType.Float32, rows / batchSize, columns))
            {
                var w = TensorSharp.RandomGenerator.BuildRandomBernoulliWeight(new long[] {rows / batchSize, columns }, prob);                
                noise.SetElementsAsFloat(w);

                return rows / batchSize == 1 ? noise.Expand(rows, columns) : noise.RepeatTensor(batchSize, 1);
            }
        }

        public IWeightTensor LayerNorm(IWeightTensor src, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-9f)
        {
            var srcT = src as WeightTensor;
            var alphaT = alpha as WeightTensor;
            var betaT = beta as WeightTensor;

            var res = this.m_weightTensorFactory.CreateWeightTensor(srcT.Sizes, this.m_deviceId, name: $"{this.GetHashString(src.Name, alpha.Name, beta.Name)}.LayerNorm");
            this.VisualizeNodes(new IWeightTensor[] { src, alpha, beta }, res);

                Ops.LayerNorm(res.TWeight, srcT.TWeight, alphaT.TWeight, betaT.TWeight, eps);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                        Ops.LayerNormGrad(srcT.TGradient, alphaT.TGradient, betaT.TGradient, res.TGradient, res.TWeight, srcT.TWeight, alphaT.TWeight, betaT.TWeight, eps);


                    res.Dispose();
                };
                this.m_backprop.Add(backward);

                srcT.UnbindFromComputeGraph();

                alphaT.UnbindFromComputeGraph();
                betaT.UnbindFromComputeGraph();
            }

            return res;
        }



        /// <summary>
        /// LayerNorm (src1 + src2)
        /// </summary>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <param name="alpha"></param>
        /// <param name="beta"></param>
        /// <param name="eps"></param>
        /// <returns></returns>
        public IWeightTensor AddLayerNorm(IWeightTensor src1, IWeightTensor src2, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-09f)
        {
            var src1T = src1 as WeightTensor;
            var src2T = src2 as WeightTensor;
            var alphaT = alpha as WeightTensor;
            var betaT = beta as WeightTensor;

            var res = this.m_weightTensorFactory.CreateWeightTensor(src1T.Sizes, this.m_deviceId, name: $"{this.GetHashString(src1.Name, src2.Name, alpha.Name, beta.Name)}.AddLayerNorm");
            this.VisualizeNodes(new IWeightTensor[] { src1, src2, alpha, beta }, res);

            Ops.AddLayerNorm(res.TWeight, src1T.TWeight, src2T.TWeight, alphaT.TWeight, betaT.TWeight, eps);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    Ops.AddLayerNormGrad(src1T.TGradient, src2T.TGradient, alphaT.TGradient, betaT.TGradient, res.TGradient, res.TWeight, src1T.TWeight, src2T.TWeight, alphaT.TWeight, betaT.TWeight, eps);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);

                src1T.UnbindFromComputeGraph();
                src2T.UnbindFromComputeGraph();

                alphaT.UnbindFromComputeGraph();
                betaT.UnbindFromComputeGraph();
            }

            return res;
        }


        public IWeightTensor Dropout(IWeightTensor V, int batchSize, float drop_prob, bool inPlace = false)
        {
            if (drop_prob == 0 || !this.m_needsBackprop)
            {
                return V;
            }

            // Generate noise tensor
            var p = 1.0f - drop_prob;
            var noise = this.BuildRandomTensor(V.Rows, V.Columns, batchSize, p);

            var w = V as WeightTensor;
            WeightTensor res = null;
            res = inPlace ? w.CopyWeightsRef($"{this.GetHashString(V.Name)}.Dropout") : this.m_weightTensorFactory.CreateWeightTensor(w.Sizes, this.m_deviceId, name: $"{this.GetHashString(V.Name)}.Dropout", graphToBind: this);

            this.VisualizeNodes(V, res);

            Ops.Mul(res.TWeight, w.TWeight, noise);
            
            Action backward = () =>
             {
                 res.ReleaseWeight();

                 if (inPlace)
                 {
                     w.TGradient = res.TGradient.CopyRef();
                 }

                 w.AddMulGradient(noise, res.TGradient, inPlace);

                 res.Dispose();
                 noise.Dispose();
             };
            this.m_backprop.Add(backward);


            return res;
        }

        public void Dispose()
        {
            // We only dispose root computing graph, For sub graph, we don't do it.
            if (this.m_isSubGraph == false)
            {
                if (this.m_backprop != null)
                {
                    this.m_backprop.Clear();
                }

                if (this.m_weightTensorFactory != null)
                {
                    this.m_weightTensorFactory.Dispose();
                }

                //if (m_setEdges != null)
                //{
                //    m_setEdges.Clear();
                //}

                //if (m_name2SubGraph != null)
                //{
                //    m_name2SubGraph.Clear();
                //}
            }
            else
            {
                foreach (WeightTensor item in this.m_tensorsBindToCurrentGraph)
                {
                    item.ReleaseWeight();
                }
            }

            this.m_tensorsBindToCurrentGraph.Clear();
        }
    }
}
