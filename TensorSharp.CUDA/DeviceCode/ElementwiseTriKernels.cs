﻿namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class ElementwiseTriKernels : CudaCode
    {
        public ElementwiseTriKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math")
        {
        }

        private static string GetFullCode()
        {
            var result = new PermutationGenerator();

            AppendTTFunc(result, "sin", "sin");
            AppendTTFunc(result, "cos", "cos");
            AppendTTFunc(result, "tan", "tan");
            AppendTTFunc(result, "asin", "asin");
            AppendTTFunc(result, "acos", "acos");
            AppendTTFunc(result, "atan", "atan");
            AppendTTFunc(result, "sinh", "sinh");
            AppendTTFunc(result, "cosh", "cosh");
            AppendTTFunc(result, "tanh", "tanhf");

            result.AddApplyTTT("atan2", "*a = atan2f(*b, *c);");

            AppendTTTFunc(result, "addtanh", "AddTanh");
            AppendTTTTFunc(result, "addtanh3", "AddTanh3");
            AppendTTTTFunc(result, "addtanhD", "AddTanhD");
            AppendTTTFunc(result, "tanhD", "TanhD");

            return result.ToString();
        }

        private static void AppendTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyT("t1_" + kernelBaseName, $"*v = {func}(*v);");
            pg.AddApplyTT("t2_" + kernelBaseName, $"*a = {func}(*b);");
        }

        private static void AppendTTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTT("t1_" + kernelBaseName, $"*a = {func}(*a, *b);");
            pg.AddApplyTTT("t2_" + kernelBaseName, $"*a = {func}(*b, *c);");
        }

        private static void AppendTTTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTTT("t1_" + kernelBaseName, $"*a = {func}(*a, *b, *c);");
            pg.AddApplyTTTT("t2_" + kernelBaseName, $"*a = {func}(*b, *c, *d);");
        }

    }
}
