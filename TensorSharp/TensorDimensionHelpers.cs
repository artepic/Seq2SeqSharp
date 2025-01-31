﻿using System.Linq;

namespace TensorSharp
{
    public static class TensorDimensionHelpers
    {
        public static long ElementCount(long[] sizes)
        {
            return sizes.Length == 0 ? 0 : sizes.Aggregate(1L, (current, t) => current * t);
        }

        public static long GetStorageSize(long[] sizes, long[] strides)
        {
            long offset = 0;
            for (var i = 0; i < sizes.Length; ++i)
            {
                offset += (sizes[i] - 1) * strides[i];
            }
            return offset + 1; // +1 to count last element, which is at *index* equal to offset
        }

        // Returns the stride required for a tensor to be contiguous.
        // If a tensor is contiguous, then the elements in the last dimension are contiguous in memory,
        // with lower numbered dimensions having increasingly large strides.
        public static long[] GetContiguousStride(long[] dims)
        {
            long acc = 1;
            var stride = new long[dims.Length];

            for (var i = dims.Length - 1; i >= 0; --i)
            {
                stride[i] = acc;
                acc *= dims[i];
            }

            //if (dims[dims.Length - 1] == 1)
            //{
            //    stride[dims.Length - 1] = 0;
            //}

            return stride;
        }
    }
}
