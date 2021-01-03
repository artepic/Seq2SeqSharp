namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class FillCopyKernels : CudaCode
    {
        public FillCopyKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply")
        {
        }

        private static string GetFullCode()
        {
            var result = new PermutationGenerator();
            result.AddApplyTS("fill", "*a = b;");

            result.AddApplyTT("copy", "*a = *b;");

            return result.ToString();
        }
    }

}
