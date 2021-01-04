﻿using System;
using TensorSharp.Core;
using TensorSharp.CUDA.DeviceCode;
using TensorSharp.CUDA.KernelOps;
using TensorSharp.CUDA.MatrixMul;

namespace TensorSharp.CUDA
{
    [OpsClass]
    public class CudaBasicOps
    {
        private readonly CopyOps copyOps;

        private readonly ElementwiseKernels elementwiseKernels = new ElementwiseKernels();
        private readonly ElementwiseOpKernels elementwiseOpKernels = new ElementwiseOpKernels();
        private readonly ElementwiseTriKernels elementwiseTriKernels = new ElementwiseTriKernels();
        private readonly ElementwiseActKernels elementwiseActKernels = new ElementwiseActKernels();

        private readonly FillCopyKernels fillCopyKernels = new FillCopyKernels();

        private readonly CudaReduceKernels cudaReduceKernels = new CudaReduceKernels();
        private readonly CudaReduceAllKernels cudaReduceAllKernels = new CudaReduceAllKernels();

        private readonly VarStdKernels varStdKernels = new VarStdKernels();
        private readonly ReduceDimIndexKernels reduceDimIndexKernels = new ReduceDimIndexKernels();

        private readonly AdvFuncKernels advFuncKernels = new AdvFuncKernels();

        public CudaBasicOps()
        {
            this.copyOps = new CopyOps(this.fillCopyKernels);
        }


        /*
        public Tensor NewContiguous(Tensor src)
        {
            var result = new Tensor(src.Allocator, src.ElementType, (long[])src.Sizes.Clone());
            Copy(result, src);
            return result;
        }

        public Tensor AsContiguous(Tensor src)
        {
            if (src.IsContiguous())
                return src.CopyRef();
            else
                return NewContiguous(src);
        }

        public Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }


        public float SumAll(Tensor src) { using (var resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float ProdAll(Tensor src) { using (var resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float MinAll(Tensor src) { using (var resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float MaxAll(Tensor src) { using (var resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        public float MeanAll(Tensor src) { using (var resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float VarAll(Tensor src) { using (var resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float StdAll(Tensor src) { using (var resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float NormAll(Tensor src, float value) { using (var resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }

        */



        [RegisterOpStorageType("buildpadselftrimask", typeof(CudaStorage))]
        public Tensor BuildPadSelfTriMask(Tensor originalLengths, int batchSize, int paddedLength)
        {
            return this.advFuncKernels.BuildPadSelfTriMask(originalLengths, batchSize, paddedLength);
        }


        [RegisterOpStorageType("buildsrctgtmask", typeof(CudaStorage))]
        public Tensor BuildSrcTgtMask(Tensor originalSrcLengths, Tensor originalTgtLengths, int batchSize, int paddedSrcLength, int paddedTgtLength)
        {
            return this.advFuncKernels.BuildSrcTgtMask(originalSrcLengths, originalTgtLengths, batchSize, paddedSrcLength, paddedTgtLength);
        }



        [RegisterOpStorageType("updatecost", typeof(CudaStorage))]
        public Tensor UpdateCost(Tensor costs, Tensor weight, Tensor ids)
        {
            return this.advFuncKernels.UpdateCost(costs, weight, ids);
        }

        [RegisterOpArgCount("copy")]
        public void CopyGpu(
            [OpArgStorageType(typeof(CudaStorage))] Tensor result,
            [OpArgStorageType(typeof(CudaStorage))] Tensor src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
            {
                throw new InvalidOperationException("Tensors must have equal numbers of elements");
            }

            if (src.DimensionCount == 0)
            {
                return;
            }

            this.copyOps.CopyGpu(result, src, totalElements);
        }

        [RegisterOpArgCount("copy")]
        public void CopyCpuToGpu(
            [OpArgStorageType(typeof(CudaStorage))] Tensor result,
            [OpArgStorageType(typeof(Cpu.CpuStorage))] Tensor src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
            {
                throw new InvalidOperationException("Tensors must have equal numbers of elements");
            }

            if (src.DimensionCount == 0)
            {
                return;
            }

            this.copyOps.CopyCpuToGpu(result, src, totalElements);
        }

        [RegisterOpArgCount("copy")]
        public void CopyGpuToCpu(
            [OpArgStorageType(typeof(Cpu.CpuStorage))] Tensor result,
            [OpArgStorageType(typeof(CudaStorage))] Tensor src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
            {
                throw new InvalidOperationException("Tensors must have equal numbers of elements");
            }

            if (src.DimensionCount == 0)
            {
                return;
            }

            this.copyOps.CopyGpuToCpu(result, src, totalElements);
        }


        [RegisterOpStorageType("fill", typeof(CudaStorage))]
        public void Fill(Tensor result, float value)
        {
            FillOp.Invoke(this.fillCopyKernels, result, value);
        }


        [RegisterOpStorageType("dot", typeof(CudaStorage))]
        public Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            var context = CudaHelpers.TSContextForTensor(lhs);
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return CudaMatrixMulDot.Dot(context, result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 1)
            {
                return CudaMatrixMulMV.Mul_M_V(context, result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return CudaMatrixMulMM.Mul_M_M(context, result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        [RegisterOpStorageType("addmm", typeof(CudaStorage))]
        public Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            // ReSharper disable once ArrangeRedundantParentheses
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException("All tensors must have the same element type");
            }

            if (result != null && !(result.Storage is CudaStorage))
            {
                throw new ArgumentException("result must be a CUDA tensor", "result");
            }

            if (!(m1.Storage is CudaStorage))
            {
                throw new ArgumentException("m1 must be a CUDA tensor", "m1");
            }

            if (!(m2.Storage is CudaStorage))
            {
                throw new ArgumentException("m2 must be a CUDA tensor", "m2");
            }

            if (src.DimensionCount != 2)
            {
                throw new ArgumentException("src must be a matrix", "src");
            }

            if (m1.DimensionCount != 2)
            {
                throw new ArgumentException("m1 must be a matrix", "m1");
            }

            if (m2.DimensionCount != 2)
            {
                throw new ArgumentException("m2 must be a matrix", "m2");
            }

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
            {
                throw new InvalidOperationException($"Size mismatch, srcSize0 = {src.Sizes[0]}, m1Size0 = {m1.Sizes[0]}, srcSize1 = {src.Sizes[1]}, m2Size1 = {m2.Sizes[1]}, m1Size1 = '{m1.Sizes[1]}', m2Size0 = '{m2.Sizes[0]}'");
            }

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            CudaMatrixMulMM.Gemm(context, alpha, m1, m2, beta, writeTarget);


            return writeTarget;
        }



        [RegisterOpStorageType("addmmbatch", typeof(CudaStorage))]
        public Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            // ReSharper disable once ArrangeRedundantParentheses
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException("All tensors must have the same element type");
            }

            if (result != null && !(result.Storage is CudaStorage))
            {
                throw new ArgumentException("result must be a CUDA tensor", "result");
            }

            if (!(m1.Storage is CudaStorage))
            {
                throw new ArgumentException("m1 must be a CUDA tensor", "m1");
            }

            if (!(m2.Storage is CudaStorage))
            {
                throw new ArgumentException("m2 must be a CUDA tensor", "m2");
            }

            if (src.DimensionCount != 3)
            {
                throw new ArgumentException("src must be a matrix", "src");
            }

            if (m1.DimensionCount != 3)
            {
                throw new ArgumentException("m1 must be a matrix", "m1");
            }

            if (m2.DimensionCount != 3)
            {
                throw new ArgumentException("m2 must be a matrix", "m2");
            }

            if (src.Sizes[1] != m1.Sizes[1] || src.Sizes[2] != m2.Sizes[2] || m1.Sizes[2] != m2.Sizes[1])
            {
                throw new InvalidOperationException($"Size mismatch, srcSize0 = {src.Sizes[0]}, m1Size0 = {m1.Sizes[0]}, srcSize1 = {src.Sizes[1]}, m2Size1 = {m2.Sizes[1]}, m1Size1 = '{m1.Sizes[1]}', m2Size0 = '{m2.Sizes[0]}'");
            }

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            CudaMatrixMulMM.GemmBatch(context, alpha, m1, m2, beta, writeTarget);


            return writeTarget;
        }

        [RegisterOpStorageType("abs", typeof(CudaStorage))]
        public Tensor Abs(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "abs", result, src); }
        [RegisterOpStorageType("neg", typeof(CudaStorage))]
        public Tensor Neg(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "neg", result, src); }
        [RegisterOpStorageType("sign", typeof(CudaStorage))]
        public Tensor Sign(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "sign", result, src); }

        [RegisterOpStorageType("sqrt", typeof(CudaStorage))]
        public Tensor Sqrt(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "sqrt", result, src); }





        [RegisterOpStorageType("rsqrt", typeof(CudaStorage))]
        public Tensor Rsqrt(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "rsqrt", result, src); }


        [RegisterOpStorageType("exp", typeof(CudaStorage))]
        public Tensor Exp(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "exp", result, src); }
        [RegisterOpStorageType("log", typeof(CudaStorage))]
        public Tensor Log(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "log", result, src); }
        [RegisterOpStorageType("log1p", typeof(CudaStorage))]
        public Tensor Log1p(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "log1p", result, src); }
        [RegisterOpStorageType("floor", typeof(CudaStorage))]
        public Tensor Floor(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "floor", result, src); }
        [RegisterOpStorageType("ceil", typeof(CudaStorage))]
        public Tensor Ceil(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "ceil", result, src); }
        [RegisterOpStorageType("round", typeof(CudaStorage))]
        public Tensor Round(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "round", result, src); }
        [RegisterOpStorageType("trunc", typeof(CudaStorage))]
        public Tensor Trunc(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "trunc", result, src); }
        [RegisterOpStorageType("frac", typeof(CudaStorage))]
        public Tensor Frac(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseKernels, "frac", result, src); }

        [RegisterOpStorageType("sin", typeof(CudaStorage))]
        public Tensor Sin(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "sin", result, src); }
        [RegisterOpStorageType("cos", typeof(CudaStorage))]
        public Tensor Cos(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "cos", result, src); }
        [RegisterOpStorageType("tan", typeof(CudaStorage))]
        public Tensor Tan(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "tan", result, src); }

        [RegisterOpStorageType("asin", typeof(CudaStorage))]
        public Tensor Asin(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "asin", result, src); }
        [RegisterOpStorageType("acos", typeof(CudaStorage))]
        public Tensor Acos(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "acos", result, src); }
        [RegisterOpStorageType("atan", typeof(CudaStorage))]
        public Tensor Atan(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "atan", result, src); }

        [RegisterOpStorageType("sinh", typeof(CudaStorage))]
        public Tensor Sinh(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "sinh", result, src); }
        [RegisterOpStorageType("cosh", typeof(CudaStorage))]
        public Tensor Cosh(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "cosh", result, src); }
        [RegisterOpStorageType("tanh", typeof(CudaStorage))]
        public Tensor Tanh(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseTriKernels, "tanh", result, src); }

        [RegisterOpStorageType("addtanhD", typeof(CudaStorage))]
        public Tensor AddTanhD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return ElementwiseTTTTOp.Invoke(this.elementwiseTriKernels, "addtanhD", result, t, resW, resG); }

        [RegisterOpStorageType("tanhD", typeof(CudaStorage))]
        public Tensor TanhD(Tensor result, Tensor resW, Tensor resG) { return ElementwiseTTTOp.Invoke(this.elementwiseTriKernels, "tanhD", result, resW, resG); }


        [RegisterOpStorageType("addtanh", typeof(CudaStorage))]
        public Tensor AddTanh(Tensor result, Tensor x, Tensor y) { return ElementwiseTTTOp.Invoke(this.elementwiseTriKernels, "addtanh", result, x, y); }


        [RegisterOpStorageType("addtanh3", typeof(CudaStorage))]
        public Tensor AddTanh3(Tensor result, Tensor x, Tensor y, Tensor z) { return ElementwiseTTTTOp.Invoke(this.elementwiseTriKernels, "addtanh3", result, x, y, z); }

        [RegisterOpStorageType("sigmoidD", typeof(CudaStorage))]
        public Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG) { return ElementwiseTTTOp.Invoke(this.elementwiseActKernels, "sigmoidD", result, resW, resG); }

        [RegisterOpStorageType("sigmoid", typeof(CudaStorage))]
        public Tensor Sigmoid(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseActKernels, "sigmoid", result, src); }

        [RegisterOpStorageType("addsigmoidD", typeof(CudaStorage))]
        public Tensor AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return ElementwiseTTTTOp.Invoke(this.elementwiseActKernels, "addsigmoidD", result, t, resW, resG); }

        [RegisterOpStorageType("relu", typeof(CudaStorage))]
        public Tensor Relu(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(this.elementwiseActKernels, "relu", result, src); }

        [RegisterOpStorageType("relud", typeof(CudaStorage))]
        public Tensor ReluD(Tensor result, Tensor w, Tensor g) { return ElementwiseTTTOp.Invoke(this.elementwiseActKernels, "relud", result, w, g); }

        [RegisterOpStorageType("addrelud", typeof(CudaStorage))]
        public Tensor AddReluD(Tensor result, Tensor t, Tensor w, Tensor g) { return ElementwiseTTTTOp.Invoke(this.elementwiseActKernels, "addrelud", result, t, w, g); }

        [RegisterOpStorageType("mulmuladd", typeof(CudaStorage))]
        public Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return ElementwiseTTTTTOp.Invoke(this.elementwiseKernels, "mulmuladd", result, x, y, z, w); }

        [RegisterOpStorageType("addmul", typeof(CudaStorage))]
        public Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z) { return ElementwiseTTTTOp.Invoke(this.elementwiseKernels, "addmul", result, x, y, z); }
        [RegisterOpStorageType("addmulv", typeof(CudaStorage))]
        public Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z) { return ElementwiseTTTSOp.Invoke(this.elementwiseKernels, "addmulv", result, x, y, z); }


        [RegisterOpStorageType("maskfill", typeof(CudaStorage))]
        public Tensor MaskFill(Tensor result, Tensor t, Tensor mask, float defValue) { return ElementwiseTTTSOp.Invoke(this.elementwiseKernels, "maskfill", result, t, mask, defValue); }



        [RegisterOpStorageType("atan2", typeof(CudaStorage))]
        public Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return Atan2Op.Invoke(this.elementwiseTriKernels, result, srcY, srcX); }
        [RegisterOpStorageType("pow", typeof(CudaStorage))]
        public Tensor Pow(Tensor result, Tensor src, float value) { return ElementwiseTTSOp.Invoke(this.elementwiseKernels, "pow", result, src, value); }
        [RegisterOpStorageType("tpow", typeof(CudaStorage))]
        public Tensor Tpow(Tensor result, float value, Tensor src) { return ElementwiseTTSOp.Invoke(this.elementwiseKernels, "tpow", result, src, value); }
        [RegisterOpStorageType("lerp", typeof(CudaStorage))]
        public Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return LerpOp.Invoke(this.elementwiseKernels, result, srcA, srcB, weight); }
        [RegisterOpStorageType("clamp", typeof(CudaStorage))]
        public Tensor Clamp(Tensor result, Tensor src, float min, float max) { return ClampOp.Invoke(this.elementwiseKernels, result, src, min, max); }

        [RegisterOpStorageType("addv", typeof(CudaStorage))]
        public Tensor Add(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "add", result, rhs, lhs); }
        [RegisterOpStorageType("subv", typeof(CudaStorage))]
        public Tensor Sub(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "sub", result, rhs, lhs); }
        [RegisterOpStorageType("rsubv", typeof(CudaStorage))]
        public Tensor Sub(Tensor result, float rhs, Tensor lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "rsub", result, lhs, rhs); }
        [RegisterOpStorageType("mulv", typeof(CudaStorage))]
        public Tensor Mul(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "mul", result, rhs, lhs); }
        [RegisterOpStorageType("divv", typeof(CudaStorage))]
        public Tensor Div(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "div", result, rhs, lhs); }
        [RegisterOpStorageType("rdivv", typeof(CudaStorage))]
        public Tensor Div(Tensor result, float rhs, Tensor lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "rdiv", result, lhs, rhs); }
        [RegisterOpStorageType("modv", typeof(CudaStorage))]
        public Tensor Mod(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "mod", result, rhs, lhs); }

        [RegisterOpStorageType("gtValue", typeof(CudaStorage))]
        public Tensor GreaterThan(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "gt", result, rhs, lhs); }
        [RegisterOpStorageType("ltValue", typeof(CudaStorage))]
        public Tensor LessThan(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "lt", result, rhs, lhs); }
        [RegisterOpStorageType("geValue", typeof(CudaStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "ge", result, rhs, lhs); }
        [RegisterOpStorageType("leValue", typeof(CudaStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "le", result, rhs, lhs); }
        [RegisterOpStorageType("eqValue", typeof(CudaStorage))]
        public Tensor EqualTo(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "eq", result, rhs, lhs); }
        [RegisterOpStorageType("neValue", typeof(CudaStorage))]
        public Tensor NotEqual(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(this.elementwiseOpKernels, "ne", result, rhs, lhs); }


        [RegisterOpStorageType("addt", typeof(CudaStorage))]
        public Tensor Add(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cadd", result, rhs, lhs); }
        [RegisterOpStorageType("subt", typeof(CudaStorage))]
        public Tensor Sub(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "csub", result, rhs, lhs); }
        [RegisterOpStorageType("mult", typeof(CudaStorage))]
        public Tensor Mul(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cmul", result, rhs, lhs); }
        [RegisterOpStorageType("divt", typeof(CudaStorage))]
        public Tensor Div(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cdiv", result, rhs, lhs); }
        [RegisterOpStorageType("modt", typeof(CudaStorage))]
        public Tensor Mod(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cmod", result, rhs, lhs); }

        [RegisterOpStorageType("gtTensor", typeof(CudaStorage))]
        public Tensor GreaterThan(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cgt", result, rhs, lhs); }
        [RegisterOpStorageType("ltTensor", typeof(CudaStorage))]
        public Tensor LessThan(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "clt", result, rhs, lhs); }
        [RegisterOpStorageType("geTensor", typeof(CudaStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cge", result, rhs, lhs); }
        [RegisterOpStorageType("leTensor", typeof(CudaStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cle", result, rhs, lhs); }
        [RegisterOpStorageType("eqTensor", typeof(CudaStorage))]
        public Tensor EqualTo(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "ceq", result, rhs, lhs); }
        [RegisterOpStorageType("neTensor", typeof(CudaStorage))]
        public Tensor NotEqual(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(this.elementwiseOpKernels, "cne", result, rhs, lhs); }


        [RegisterOpStorageType("sum", typeof(CudaStorage))]
        public Tensor Sum(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(this.cudaReduceKernels, "sum", 0.0f, ReduceInitType.GivenValue, result, src, dimension); }
        [RegisterOpStorageType("prod", typeof(CudaStorage))]
        public Tensor Prod(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(this.cudaReduceKernels, "prod", 1.0f, ReduceInitType.GivenValue, result, src, dimension); }
        [RegisterOpStorageType("min", typeof(CudaStorage))]
        public Tensor Min(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(this.cudaReduceKernels, "min", 0.0f, ReduceInitType.MaxValue, result, src, dimension); }
        [RegisterOpStorageType("max", typeof(CudaStorage))]
        public Tensor Max(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(this.cudaReduceKernels, "max", 0.0f, ReduceInitType.MinValue, result, src, dimension); }

        [RegisterOpStorageType("argmin", typeof(CudaStorage))]
        public Tensor Argmin(Tensor result, Tensor src, int dimension) { return this.reduceDimIndexKernels.ArgMin(result, src, dimension); }

        [RegisterOpStorageType("argmax", typeof(CudaStorage))]
        public Tensor Argmax(Tensor result, Tensor src, int dimension) { return this.reduceDimIndexKernels.ArgMax(result, src, dimension); }


        [RegisterOpStorageType("mean", typeof(CudaStorage))]
        public Tensor Mean(Tensor result, Tensor src, int dimension)
        {
            var requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dimension] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, requiredOutputSize);

            this.Sum(writeTarget, src, dimension);
            this.Div(writeTarget, writeTarget, src.Sizes[dimension]);
            return writeTarget;
        }

        [RegisterOpStorageType("norm", typeof(CudaStorage))]
        public Tensor Norm(Tensor result, Tensor src, int dimension, float value)
        {
            if (value == 0)
            {
                return ReductionOp.Invoke(this.cudaReduceKernels, "e0_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
            }
            else if (value == 1)
            {
                return ReductionOp.Invoke(this.cudaReduceKernels, "e1_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
            }
            else if (value == 2)
            {
                var writeTarget = ReductionOp.Invoke(this.cudaReduceKernels, "e2_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
                this.Pow(writeTarget, writeTarget, 0.5f);
                return writeTarget;
            }
            else
            {
                var writeTarget = ReductionOp.Invoke(this.cudaReduceKernels, "en_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension, value);
                this.Pow(writeTarget, writeTarget, 1.0f / value);
                return writeTarget;
            }
        }

        [RegisterOpStorageType("std", typeof(CudaStorage))]
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return this.varStdKernels.Std(result, src, dimension, normByN); }
        [RegisterOpStorageType("var", typeof(CudaStorage))]
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return this.varStdKernels.Var(result, src, dimension, normByN); }


        [RegisterOpStorageType("softmax", typeof(CudaStorage))]
        public Tensor Softmax(Tensor result, Tensor src) { return this.advFuncKernels.Softmax(result, src); }


        [RegisterOpStorageType("softmaxmask", typeof(CudaStorage))]
        public Tensor SoftmaxMask(Tensor result, Tensor src, Tensor mask) { return this.advFuncKernels.SoftmaxMask(result, src, mask); }


        [RegisterOpStorageType("softmaxgrad", typeof(CudaStorage))]
        public Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true) { return this.advFuncKernels.SoftmaxGrad(grad, adj, val, addGrad); }

        [RegisterOpStorageType("layernorm", typeof(CudaStorage))]
        public Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f) { return this.advFuncKernels.LayerNorm(result, src, alpha, beta, eps); }
        [RegisterOpStorageType("layernormgrad", typeof(CudaStorage))]
        public Tensor LayerNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-09f) { return this.advFuncKernels.LayerNormGrad(outGrad, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps); }


        [RegisterOpStorageType("addlayernorm", typeof(CudaStorage))]
        public Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-09f) { return this.advFuncKernels.AddLayerNorm(result, src1, src2, alpha, beta, eps); }
        [RegisterOpStorageType("addlayernormgrad", typeof(CudaStorage))]
        public void AddLayerNormGrad(Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-09f) {
            this.advFuncKernels.AddLayerNormGrad(out1Grad, out2Grad, alphaGrad, betaGrad, inGrad, y, x1, x2, alpha, beta, eps); }

        [RegisterOpStorageType("adam", typeof(CudaStorage))]
        public Tensor Adam(Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            return this.advFuncKernels.Adam(weight, gradient, v, m, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
        }


        [RegisterOpStorageType("rmsprop", typeof(CudaStorage))]
        public Tensor RMSProp(Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            return this.advFuncKernels.RMSProp(weight, gradient, cache, batchSize, step_size, clipval, regc, decay_rate, eps);
        }


        [RegisterOpStorageType("sumall", typeof(CudaStorage))]
        public Tensor SumAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "sumAll", result, src);
        }

        [RegisterOpStorageType("prodall", typeof(CudaStorage))]
        public Tensor ProdAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(this.cudaReduceAllKernels, 1.0f, ReduceInitType.GivenValue, "prodAll", result, src);
        }

        [RegisterOpStorageType("minall", typeof(CudaStorage))]
        public Tensor MinAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0, ReduceInitType.MaxValue, "minAll", result, src);
        }

        [RegisterOpStorageType("maxall", typeof(CudaStorage))]
        public Tensor MaxAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0, ReduceInitType.MinValue, "maxAll", result, src);
        }


        [RegisterOpStorageType("meanall", typeof(CudaStorage))]
        public Tensor MeanAll(Tensor result, Tensor src)
        {
            if (src.DimensionCount == 0 || src.ElementCount() == 0)
            {
                throw new ArgumentException("src must be a non-empty tensor");
            }

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            this.SumAll(writeTarget, src);
            this.Div(writeTarget, writeTarget, src.ElementCount());
            return writeTarget;
        }

        [RegisterOpStorageType("normall", typeof(CudaStorage))]
        public Tensor NormAll(Tensor result, Tensor src, float value)
        {
            if (value == 0)
            {
                return ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e0_norm", result, src);
            }
            else if (value == 1)
            {
                return ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e1_norm", result, src);
            }
            else if (value == 2)
            {
                var writeTarget = ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e2_norm", result, src);
                this.Pow(writeTarget, writeTarget, 0.5f);
                return writeTarget;
            }
            else
            {
                var writeTarget = ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_norm", result, src, value);
                this.Pow(writeTarget, writeTarget, 1.0f / value);
                return writeTarget;
            }
        }


        [RegisterOpStorageType("varall", typeof(CudaStorage))]
        public Tensor VarAll(Tensor result, Tensor src)
        {
            if (src.DimensionCount == 0 || src.ElementCount() == 0)
            {
                throw new ArgumentException("src must be a non-empty tensor");
            }

            var mean = Ops.MeanAll(src);
            var writeTarget = ReduceAllOp.Invoke(this.cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_norm", result, src, mean);
            this.Div(writeTarget, writeTarget, src.ElementCount() - 1);
            return writeTarget;
        }

        [RegisterOpStorageType("stdall", typeof(CudaStorage))]
        public Tensor StdAll(Tensor result, Tensor src)
        {
            var writeTarget = this.VarAll(result, src);
            this.Pow(writeTarget, writeTarget, 0.5f);
            return writeTarget;
        }

    }
}
