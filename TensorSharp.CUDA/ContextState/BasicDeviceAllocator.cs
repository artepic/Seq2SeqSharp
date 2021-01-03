using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;

namespace TensorSharp.CUDA.ContextState
{
    /// <summary>
    /// This allocator simply forwards all alloc/free requests to CUDA. This will generally be slow
    /// because calling cudaMalloc causes GPU synchronization
    /// </summary>
    public class BasicDeviceAllocator : IDeviceAllocator
    {
        private readonly CudaContext context;

        public BasicDeviceAllocator(CudaContext cudaContext)
        {
            this.context = cudaContext;
        }

        public void Dispose()
        {
        }


        public IDeviceMemory Allocate(long byteCount)
        {
            var buffer = this.context.AllocateMemory(byteCount);
            return new BasicDeviceMemory(buffer, () => this.context.FreeMemory(buffer));
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }

    public class BasicDeviceMemory : IDeviceMemory
    {
        private readonly CUdeviceptr pointer;
        private readonly Action freeHandler;

        public CUdeviceptr Pointer => this.pointer;


        public BasicDeviceMemory(CUdeviceptr pointer, Action freeHandler)
        {
            this.pointer = pointer;
            this.freeHandler = freeHandler;
        }

        public void Free()
        {
            this.freeHandler();
        }
    }
}
