﻿using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using TensorSharp.Core;

namespace TensorSharp.CUDA.MatrixMul
{
    public static class CudaMatrixMulDot
    {
        public static Tensor Dot(TSCudaContext context, Tensor result, Tensor lhs, Tensor rhs)
        {
            // ReSharper disable once ArrangeRedundantParentheses
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
            {
                throw new InvalidOperationException("All tensors must have the same element type");
            }

            CudaHelpers.ThrowIfDifferentDevices(result, lhs, rhs);

            if (result != null && !(result.Storage is CudaStorage))
            {
                throw new ArgumentException("result must be a CUDA tensor", nameof(result));
            }

            if (!(lhs.Storage is CudaStorage))
            {
                throw new ArgumentException("lhs must be a CUDA tensor", nameof(lhs));
            }

            if (!(rhs.Storage is CudaStorage))
            {
                throw new ArgumentException("rhs must be a CUDA tensor", nameof(rhs));
            }

            if (lhs.DimensionCount != 1)
            {
                throw new ArgumentException("lhs must have 1 dimension (ie. be a vector)", nameof(lhs));
            }

            if (rhs.DimensionCount != 1)
            {
                throw new ArgumentException("rhs must have 1 dimension (ie. be a vector)", nameof(rhs));
            }

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, 1);

            if (writeTarget.ElementType == DType.Float32)
            {
                Run_Dot_float(context, writeTarget, lhs, rhs);
            }
            else if (writeTarget.ElementType == DType.Float64)
            {
                Run_Dot_double(context, writeTarget, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException("CUDA vector dot product with element type " + result.ElementType + " not supported");
            }

            return writeTarget;
        }

        private static void Run_Dot_float(TSCudaContext context, Tensor result, Tensor lhs, Tensor rhs)
        {
            using var blas = context.BlasForTensor(lhs);
            //var resultPtr = CudaNativeHelpers.GetBufferStart(result);
            var lhsPtr = CudaHelpers.GetBufferStart(lhs);
            var rhsPtr = CudaHelpers.GetBufferStart(rhs);

            var n = (int)lhs.Sizes[0];
            var incx = (int)lhs.Strides[0];
            var incy = (int)rhs.Strides[0];

            float resultVal = 0;
            var _status = CudaBlasNativeMethods.cublasSdot_v2(blas.Value.CublasHandle, n, lhsPtr, incx, rhsPtr, incy, ref resultVal);
            if (_status != CublasStatus.Success)
            {
                throw new CudaBlasException(_status);
            }

            result.Storage.SetElementAsFloat(result.StorageOffset, resultVal);
        }

        private static void Run_Dot_double(TSCudaContext context, Tensor result, Tensor lhs, Tensor rhs)
        {
            using var blas = context.BlasForTensor(lhs);
            //var resultPtr = CudaNativeHelpers.GetBufferStart(result);
            var lhsPtr = CudaHelpers.GetBufferStart(lhs);
            var rhsPtr = CudaHelpers.GetBufferStart(rhs);

            var n = (int)lhs.Sizes[0];
            var incx = (int)lhs.Strides[0];
            var incy = (int)rhs.Strides[0];

            // TODO add SetElementAsDouble to prevent need to round to float here
            double resultVal = 0;
            var _status = CudaBlasNativeMethods.cublasDdot_v2(blas.Value.CublasHandle, n, lhsPtr, incx, rhsPtr, incy, ref resultVal);
            if (_status != CublasStatus.Success)
            {
                throw new CudaBlasException(_status);
            }

            result.Storage.SetElementAsFloat(result.StorageOffset, (float)resultVal);
        }
    }
}
