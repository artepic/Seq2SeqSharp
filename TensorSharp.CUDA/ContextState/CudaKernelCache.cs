using AdvUtils;
using ManagedCuda;
using System;
using System.Collections.Generic;

namespace TensorSharp.CUDA.ContextState
{
    [Serializable]
    public class CudaKernelCache : IDisposable
    {
        private readonly Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel> activeKernels = new Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel>();

        public CudaKernelCache()
        {
        }

        private readonly object locker = new object();

        public void Dispose()
        {
            lock (this.locker)
            {
                foreach (var kvp in this.activeKernels)
                {
                    var ctx = kvp.Key.Item1;
                    var kernel = kvp.Value;

                    ctx.UnloadKernel(kernel);
                }
            }
        }



        public CudaKernel Get(CudaContext context, byte[] ptx, string kernelName)
        {
            lock (this.locker)
            {
                try
                {
                    if (this.activeKernels.TryGetValue(Tuple.Create(context, ptx, kernelName), out var value))
                    {
                        return value;
                    }
                    else
                    {
                        value = context.LoadKernelPTX(ptx, kernelName);
                        this.activeKernels.Add(Tuple.Create(context, ptx, kernelName), value);
                        return value;
                    }
                }
                catch (Exception err)
                {
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: '{err.Message}'");
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Call stack: '{err.StackTrace}'");

                    throw err;
                }
            }
        }
    }

}
