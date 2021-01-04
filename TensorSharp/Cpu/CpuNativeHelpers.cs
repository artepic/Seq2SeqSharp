using System;

namespace TensorSharp.Cpu
{
    public static class CpuNativeHelpers
    {
        public static IntPtr GetBufferStart(Tensor tensor)
        {
            var buffer = ((CpuStorage)tensor.Storage).buffer;
            return PtrAdd(buffer, tensor.StorageOffset * tensor.ElementType.Size());
        }

        private static IntPtr PtrAdd(IntPtr ptr, long offset)
        {
            return new(ptr.ToInt64() + offset);
        }

    }
}
