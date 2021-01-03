﻿using System;

namespace TensorSharp
{
    [Serializable]
    public abstract class Storage : RefCounted
    {
        public Storage(IAllocator allocator, DType elementType, long elementCount)
        {
            this.Allocator = allocator;
            this.ElementType = elementType;
            this.ElementCount = elementCount;
        }

        /// <summary>
        /// Gets a reference to the allocator that constructed this Storage object.
        /// </summary>
        public IAllocator Allocator { get; private set; }

        public DType ElementType { get; private set; }
        public long ElementCount { get; private set; }

        public long ByteLength => this.ElementCount * this.ElementType.Size();

        public bool IsOwnerExclusive()
        {
            return this.GetCurrentRefCount() == 1;
        }


        public abstract int[] GetElementsAsInt(long index, int length);
        public abstract void SetElementsAsInt(long index, int[] value);


        public abstract string LocationDescription();

        public abstract float GetElementAsFloat(long index);
        public abstract float[] GetElementsAsFloat(long index, int length);
        public abstract void SetElementAsFloat(long index, float value);
        public abstract void SetElementsAsFloat(long index, float[] value);

        public abstract void CopyToStorage(long storageIndex, IntPtr src, long byteCount);
        public abstract void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount);

    }
}
