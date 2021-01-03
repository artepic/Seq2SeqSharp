using System;

namespace TensorSharp.CUDA
{
    [Serializable]
    public class CudaAllocator : IAllocator
    {
        private readonly TSCudaContext context;
        private readonly int deviceId;

        public CudaAllocator(TSCudaContext context, int deviceId)
        {
            this.context = context;
            this.deviceId = deviceId;
        }

        public TSCudaContext Context => this.context;
        public int DeviceId => this.deviceId;

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CudaStorage(this, this.context, this.context.CudaContextForDevice(this.deviceId), elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            return this.Context.AllocatorForDevice(this.DeviceId).GetAllocatedMemoryRatio();
        }
    }
}
