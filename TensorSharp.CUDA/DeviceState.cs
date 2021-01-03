using ManagedCuda;
using ManagedCuda.CudaBlas;
using System;
using TensorSharp.CUDA.ContextState;
using TensorSharp.CUDA.Util;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Used by TSCudaContext to maintain per-device state
    /// </summary>
    public class DeviceState : IDisposable
    {
        private const int ScratchSpacePerSMStream = 4 * sizeof(float);


        public readonly CudaContext CudaContext;
        public readonly CudaDeviceProperties DeviceInfo;

        public readonly ObjectPool<CudaBlas> BlasHandles;
        // public readonly ObjectPool<ManagedCuda.CudaDNN.CudaDNNContext> DnnHandles;

        public readonly IDeviceAllocator MemoryAllocator;
        public readonly ScratchSpace ScratchSpace;


        public DeviceState(int deviceId, float memoryUsageRatio = 0.9f)
        {
            this.CudaContext = new CudaContext(deviceId);
            this.DeviceInfo = this.CudaContext.GetDeviceInfo();

            this.BlasHandles = new ObjectPool<CudaBlas>(1, () =>
                                                        {
                                                            this.CudaContext.SetCurrent();
                                                            return new CudaBlas();
                                                        },
                                                        blas => blas.Dispose());

            this.MemoryAllocator = new PoolingDeviceAllocator(this.CudaContext, memoryUsageRatio);
            this.ScratchSpace = AllocScratchSpace(this.CudaContext, this.DeviceInfo);
        }

        public void Dispose()
        {
            this.BlasHandles.Dispose();
            this.CudaContext.Dispose();
            this.MemoryAllocator.Dispose();
        }

        private static ScratchSpace AllocScratchSpace(CudaContext context, CudaDeviceProperties deviceProps)
        {
            var size = ScratchSpacePerSMStream * deviceProps.MultiProcessorCount;
            var buffer = context.AllocateMemory(size);
            return new ScratchSpace() { size = size, buffer = buffer };
        }
    }
}
