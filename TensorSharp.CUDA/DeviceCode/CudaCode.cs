using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.DeviceCode
{
    public abstract class CudaCode : IPrecompilable
    {
        private readonly string code;
        private readonly string[] requiredHeaders;
        private byte[] ptx = null;

        protected CudaCode(string code, params string[] requiredHeaders)
        {
            this.code = code;
            this.requiredHeaders = requiredHeaders;
        }

        public byte[] GetPtx(CudaCompiler compiler)
        {
            if (this.ptx == null)
            {
                this.Precompile(compiler);
            }
            return this.ptx;
        }

        public void Precompile(CudaCompiler compiler)
        {
            this.ptx = compiler.CompileToPtx(this.code, this.requiredHeaders);
        }
    }

}
