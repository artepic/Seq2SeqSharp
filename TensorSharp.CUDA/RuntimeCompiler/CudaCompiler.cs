using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    [Serializable]
    public class CudaCompiler
    {
        private readonly Dictionary<string, string> includes = new Dictionary<string, string>();
        private readonly KernelDiskCache diskCache;
        private readonly string[] m_options;

        public CudaCompiler(KernelDiskCache diskCache, string[] options = null)
        {
            this.diskCache = diskCache;
            this.m_options = options;
            this.RegisterAttributeHeaders(Assembly.GetExecutingAssembly());
        }

        public byte[] CompileToPtx(string code, params string[] prependIncludes)
        {
            // We manually prepend include files here, so that the header content forms part of the hash of the source
            // code. This means that changes to headers will correctly trigger a recompile.
            var finalCode = new StringBuilder();
            foreach (var includeName in prependIncludes)
            {
                finalCode.Append(this.includes[includeName]).Append('\n');
            }
            finalCode.Append(code);
            var finalCodeString = finalCode.ToString();

            return this.diskCache.Get(finalCodeString, this.DoCompile);
        }

        private byte[] DoCompile(string fullSource)
        {
            var rtc = new ManagedCuda.NVRTC.CudaRuntimeCompiler(fullSource, null);

            try
            {
                if (this.m_options == null ||
                    this.m_options.Length == 0)
                {
                    rtc.Compile(new string[] { });
                }
                else
                {
                    Logger.WriteLine($"Compiler Options: {string.Join(" ", this.m_options)}");
                    rtc.Compile(this.m_options);
                    //rtc.Compile(new string[] { "--use_fast_math", "--gpu-architecture=compute_60" });
                }
            }
            catch
            {
                throw new ApplicationException("Error compiling CUDA code: " + rtc.GetLogAsString());
            }

            return rtc.GetPTX();
        }

        public void RegisterHeader(string name, string content)
        {
            this.includes.Add(name, content);
        }


        private void RegisterAttributeHeaders(Assembly assembly)
        {
            foreach (var applyType in assembly.TypesWithAttribute<CudaIncludeAttribute>(false))
            {
                foreach (var attribute in applyType.Item2)
                {
                    var info = this.HeaderInfoFromAttribute(applyType.Item1, attribute);
                    this.RegisterHeader(info.Item1, info.Item2);
                }
            }
        }

        private Tuple<string, string> HeaderInfoFromAttribute(Type containingType, CudaIncludeAttribute attribute)
        {
            var field = containingType.GetField(attribute.FieldName, BindingFlags.Public | BindingFlags.Static);
            var content = (string)field.GetValue(null);
            return Tuple.Create(attribute.IncludeName, content);
        }
    }
}
