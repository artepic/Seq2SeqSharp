using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cpu
{
    public class CpuStorage : Storage
    {
        public IntPtr buffer;


        public CpuStorage(IAllocator allocator, DType ElementType, long elementCount)
            : base(allocator, ElementType, elementCount)
        {
            this.buffer = Marshal.AllocHGlobal(new IntPtr(this.ByteLength));
        }

        protected override void Destroy()
        {
            Marshal.FreeHGlobal(this.buffer);
            this.buffer = IntPtr.Zero;
        }

        public override string LocationDescription()
        {
            return "CPU";
        }

        public IntPtr PtrAtElement(long index)
        {
            return new IntPtr(this.buffer.ToInt64() + (index * this.ElementType.Size()));
        }

        public override int[] GetElementsAsInt(long index, int length)
        {
            unsafe
            {
                if (this.ElementType == DType.Int32)
                {
                    var p = ((int*)this.buffer.ToPointer());
                    var array = new int[length];

                    for (var i = 0; i < length; i++)
                    {
                        array[i] = *(p + i);
                    }
                    return array;
                }
                else
                {
                    throw new NotSupportedException("Element type " + this.ElementType + " not supported");
                }
            }
        }

        public override float GetElementAsFloat(long index)
        {
            unsafe
            {
                if (this.ElementType == DType.Float32)
                {
                    return ((float*)this.buffer.ToPointer())[index];
                }
                else if (this.ElementType == DType.Float64)
                {
                    return (float)((double*)this.buffer.ToPointer())[index];
                }
                else if (this.ElementType == DType.Int32)
                {
                    return ((int*)this.buffer.ToPointer())[index];
                }
                else if (this.ElementType == DType.UInt8)
                {
                    return ((byte*)this.buffer.ToPointer())[index];
                }
                else
                {
                    throw new NotSupportedException("Element type " + this.ElementType + " not supported");
                }
            }
        }

        public override float[] GetElementsAsFloat(long index, int length)
        {
            unsafe
            {
                if (this.ElementType == DType.Float32)
                {
                    var p = ((float*)this.buffer.ToPointer());
                    var array = new float[length];

                    for (var i = 0; i < length; i++)
                    {
                        array[i] = *(p + i);
                    }
                    return array;
                }
                else
                {
                    throw new NotSupportedException("Element type " + this.ElementType + " not supported");
                }
            }
        }

        public override void SetElementAsFloat(long index, float value)
        {
            unsafe
            {
                if (this.ElementType == DType.Float32)
                {
                    ((float*)this.buffer.ToPointer())[index] = value;
                }
                else if (this.ElementType == DType.Float64)
                {
                    ((double*)this.buffer.ToPointer())[index] = value;
                }
                else if (this.ElementType == DType.Int32)
                {
                    ((int*)this.buffer.ToPointer())[index] = (int)value;
                }
                else if (this.ElementType == DType.UInt8)
                {
                    ((byte*)this.buffer.ToPointer())[index] = (byte)value;
                }
                else
                {
                    throw new NotSupportedException("Element type " + this.ElementType + " not supported");
                }
            }
        }

        public override void SetElementsAsInt(long index, int[] value)
        {
            unsafe
            {
                if (this.ElementType == DType.Int32)
                {
                    for (var i = 0; i < value.Length; i++)
                    {
                        ((int*)this.buffer.ToPointer())[index + i] = value[i];
                    }
                }
                else
                {
                    throw new NotSupportedException("Element type " + this.ElementType + " not supported");
                }
            }
        }

        public override void SetElementsAsFloat(long index, float[] value)
        {
            unsafe
            {
                if (this.ElementType == DType.Float32)
                {
                    for (var i = 0; i < value.Length; i++)
                    {
                        ((float*)this.buffer.ToPointer())[index + i] = value[i];
                    }
                }
                else
                {
                    throw new NotSupportedException("Element type " + this.ElementType + " not supported");
                }
            }
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            var dstPtr = this.PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(src.ToPointer(), dstPtr.ToPointer(), byteCount, byteCount);
            }
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            var srcPtr = this.PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(srcPtr.ToPointer(), dst.ToPointer(), byteCount, byteCount);
            }
        }
    }
}
