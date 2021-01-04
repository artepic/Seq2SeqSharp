using AdvUtils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using TensorSharp.CUDA.ContextState;

namespace TensorSharp.CUDA
{
    [Serializable]
    public class CudaStorage : Storage
    {
        private readonly CudaContext context;

        private IDeviceMemory bufferHandle;
        private readonly CUdeviceptr deviceBuffer;


        public CudaStorage(IAllocator allocator, TSCudaContext tsContext, CudaContext context, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            this.TSContext = tsContext;
            this.context = context;

            this.bufferHandle = tsContext.AllocatorForDevice(this.DeviceId).Allocate(this.ByteLength);
            this.deviceBuffer = this.bufferHandle.Pointer;
        }

        public TSCudaContext TSContext { get; private set; }

        protected override void Destroy()
        {
            if (this.bufferHandle != null)
            {
                this.bufferHandle.Free();
                this.bufferHandle = null;
            }
        }

        public override int[] GetElementsAsInt(long index, int length)
        {
            var ptr = this.DevicePtrAtElement(index);

            if (this.ElementType == DType.Int32) { var result = new int[length];
                this.context.CopyToHost(result, ptr); return result; }
            else
            {
                throw new NotSupportedException("Element type " + this.ElementType + " not supported");
            }
        }

        public override void SetElementsAsInt(long index, int[] value)
        {
            var ptr = this.DevicePtrAtElement(index);

            if (this.ElementType == DType.Int32) {
                this.context.CopyToDevice(ptr, value); }
            else
            {
                throw new NotSupportedException("Element type " + this.ElementType + " not supported");
            }
        }


        public override string LocationDescription()
        {
            return "CUDA:" + this.context.DeviceId;
        }

        public int DeviceId => this.context.DeviceId;

        public CUdeviceptr DevicePtrAtElement(long index)
        {
            var offset = this.ElementType.Size() * index;
            return new CUdeviceptr(this.deviceBuffer.Pointer + offset);
        }

        public override float GetElementAsFloat(long index)
        {
            var ptr = this.DevicePtrAtElement(index);

            try
            {
                if (this.ElementType == DType.Float32) { var result = new float[1];
                    this.context.CopyToHost(result, ptr); return result[0]; }
                else if (this.ElementType == DType.Float64) { var result = new double[1];
                    this.context.CopyToHost(result, ptr); return (float)result[0]; }
                else if (this.ElementType == DType.Int32) { var result = new int[1];
                    this.context.CopyToHost(result, ptr); return result[0]; }
                else if (this.ElementType == DType.UInt8) { var result = new byte[1];
                    this.context.CopyToHost(result, ptr); return result[0]; }
                else
                {
                    throw new NotSupportedException("Element type " + this.ElementType + " not supported");
                }
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Failed to get element as float from addr = '{ptr.Pointer}'");
                Logger.WriteLine($"Exception: {err.Message}");
                Logger.WriteLine($"Call stack: {err.StackTrace}");

                throw err;
            }
        }


        public override float[] GetElementsAsFloat(long index, int length)
        {
            var ptr = this.DevicePtrAtElement(index);

            if (this.ElementType == DType.Float32) { var result = new float[length];
                this.context.CopyToHost(result, ptr); return result; }
            else
            {
                throw new NotSupportedException("Element type " + this.ElementType + " not supported");
            }
        }

        public override void SetElementAsFloat(long index, float value)
        {
            var ptr = this.DevicePtrAtElement(index);

            if (this.ElementType == DType.Float32) {
                this.context.CopyToDevice(ptr, value); }
            else if (this.ElementType == DType.Float64) {
                this.context.CopyToDevice(ptr, (double)value); }
            else if (this.ElementType == DType.Int32) {
                this.context.CopyToDevice(ptr, (int)value); }
            else if (this.ElementType == DType.UInt8) {
                this.context.CopyToDevice(ptr, (byte)value); }
            else
            {
                throw new NotSupportedException("Element type " + this.ElementType + " not supported");
            }
        }

        public override void SetElementsAsFloat(long index, float[] value)
        {
            var ptr = this.DevicePtrAtElement(index);

            if (this.ElementType == DType.Float32) {
                this.context.CopyToDevice(ptr, value); }
            else
            {
                throw new NotSupportedException("Element type " + this.ElementType + " not supported");
            }
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            var dstPtr = this.DevicePtrAtElement(storageIndex);
            this.context.SetCurrent();
            this.context.CopyToDevice(dstPtr, src, byteCount);
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            var srcPtr = this.DevicePtrAtElement(storageIndex);

            // Call this method directly instead of CudaContext.CopyToHost because this method supports a long byteCount
            // CopyToHost only supports uint byteCount.
            var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dst, srcPtr, byteCount);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
        }
    }
}
