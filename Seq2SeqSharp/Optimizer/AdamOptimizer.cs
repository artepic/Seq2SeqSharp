using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{

    public class AdamOptimizer
    {
        private static float m_beta1 = 0.9f;
        private static float m_beta2 = 0.98f;
        private static readonly float m_smoothEps = 1e-9f;
        private readonly ConcurrentDictionary<string, Tensor> m_cacheName2V;
        private readonly ConcurrentDictionary<string, Tensor> m_cacheName2M;
        private readonly float m_clipval;

        public AdamOptimizer(float clipval, float beta1 = 0.9f, float beta2 = 0.98f)
        {
            Logger.WriteLine($"Creating Adam optimizer. GradClip = '{clipval}', Beta1 = '{beta1}', Beta2 = '{beta2}'");

            this.m_cacheName2V = new ConcurrentDictionary<string, Tensor>();
            this.m_cacheName2M = new ConcurrentDictionary<string, Tensor>();

            this.m_clipval = clipval;
            m_beta1 = beta1;
            m_beta2 = beta2;
        }

        public void UpdateWeights(List<IWeightTensor> model, int batchSize, float step_size, float regc, int iter)
        {
            var id2Models = new Dictionary<int, List<IWeightTensor>>();
            var setWeightsName = new HashSet<string>();
            foreach (var item in model)
            {
                if (!item.IsTrainable)
                {
                    continue;
                }

                if (setWeightsName.Contains(item.Name))
                {
                    throw new ArgumentException($"Found duplicated weights name '{item.Name}'");
                }
                setWeightsName.Add(item.Name);

                if (id2Models.ContainsKey(item.DeviceId) == false)
                {
                    id2Models.Add(item.DeviceId, new List<IWeightTensor>());
                }
                id2Models[item.DeviceId].Add(item);

                if (this.m_cacheName2V.ContainsKey(item.Name) == false)
                {
                    var allocator = TensorAllocator.Allocator(item.DeviceId);
                    this.m_cacheName2V[item.Name] = new Tensor(allocator, DType.Float32, item.Sizes);
                    Ops.Fill(this.m_cacheName2V[item.Name], 0.0f);

                    this.m_cacheName2M[item.Name] = new Tensor(allocator, DType.Float32, item.Sizes);
                    Ops.Fill(this.m_cacheName2M[item.Name], 0.0f);

                    Logger.WriteLine($"Added weight '{item.Name}' to optimizer.");
                }
            }

            Parallel.ForEach(id2Models, kv =>
            {
                foreach (var item in kv.Value)
                {
                    var m = item as WeightTensor;
                    this.UpdateWeightsTensor(m, batchSize, step_size, this.m_clipval, regc, iter);
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float clipval, float regc, int iter)
        {
            // Ops.RMSProp(m.TWeight, m.TGradient, m.TV, batchSize, step_size, clipval, regc, decay_rate, smooth_eps);
            Ops.Adam(m.TWeight, m.TGradient, this.m_cacheName2V[m.Name], this.m_cacheName2M[m.Name], batchSize, step_size, clipval, regc, m_beta2, m_beta1, iter, m_smoothEps);
        }

    }
}
