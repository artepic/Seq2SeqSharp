namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class ElementwiseActKernels : CudaCode
    {
        public ElementwiseActKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math")
        {
        }

        private static string GetFullCode()
        {
            var result = new PermutationGenerator();

            AppendTTFunc(result, "sigmoid", "Sigmoid");
            AppendTTTTFunc(result, "addsigmoidD", "AddSigmoidD");
            AppendTTTFunc(result, "sigmoidD", "SigmoidD");

            AppendTTFunc(result, "relu", "relu");
            AppendTTTFunc(result, "relud", "relud");
            AppendTTTTFunc(result, "addrelud", "addrelud");

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
