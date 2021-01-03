﻿using System;
using System.Reflection;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    [OpsClass]
    public class CpuIndexingOps
    {
        public CpuIndexingOps()
        {
        }

        private readonly MethodInfo gather_func = NativeWrapper.GetMethod("TS_Gather");
        private readonly MethodInfo scatter_func = NativeWrapper.GetMethod("TS_Scatter");
        private readonly MethodInfo scatterFill_func = NativeWrapper.GetMethod("TS_ScatterFill");


        [RegisterOpStorageType("gather", typeof(CpuStorage))]
        public Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            if (result != null && result.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("result and src must have same number of dimensions");
            }

            if (result != null && dim < 0 && dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dim");
            }

            if (indices.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("src and indices must have same number of dimensions");
            }

            if (result != null && !result.IsSameSizeAs(indices))
            {
                throw new InvalidOperationException("result and indices must be the same size");
            }

            if (result != null && !TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException("result and src must be the same size except in dimension dim");
            }

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, indices.Allocator, src.ElementType, false, indices.Sizes);

            NativeWrapper.InvokeTypeMatch(this.gather_func, writeTarget, src, dim, indices);
            return writeTarget;
        }

        [RegisterOpStorageType("scatter", typeof(CpuStorage))]
        public Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices)
        {
            if (result == null)
            {
                throw new ArgumentNullException("result");
            }

            if (result.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("result and src must have same number of dimensions");
            }

            if (dim < 0 && dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dim");
            }

            if (indices.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("src and indices must have same number of dimensions");
            }

            if (!src.IsSameSizeAs(indices))
            {
                throw new InvalidOperationException("src and indices must be the same size");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException("result and src must be the same size except in dimension dim");
            }

            var writeTarget = result;

            NativeWrapper.InvokeTypeMatch(this.scatter_func, writeTarget, src, dim, indices);
            return writeTarget;
        }

        [RegisterOpStorageType("scatter_fill", typeof(CpuStorage))]
        public Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices)
        {
            if (result == null)
            {
                throw new ArgumentNullException("result");
            }

            if (dim < 0 && dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dim");
            }

            if (indices.DimensionCount != result.DimensionCount)
            {
                throw new InvalidOperationException("result and indices must have same number of dimensions");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(indices.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException("result and indices must be the same size except in dimension dim");
            }

            var writeTarget = result;

            NativeWrapper.InvokeTypeMatch(this.scatterFill_func, writeTarget, value, dim, indices);
            return writeTarget;
        }
    }
}
