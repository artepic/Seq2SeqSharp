using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    public class DeviceKernelTemplate
    {
        private readonly string templateCode;
        private readonly List<string> requiredHeaders;
        private readonly HashSet<string> requiredConfigArgs = new();
        private readonly Dictionary<KernelConfig, byte[]> ptxCache = new();


        public DeviceKernelTemplate(string templateCode, params string[] requiredHeaders)
        {
            this.templateCode = templateCode;
            this.requiredHeaders = new List<string>(requiredHeaders);
        }

        public void AddConfigArgs(params string[] args)
        {
            foreach (var item in args)
            {
                this.requiredConfigArgs.Add(item);
            }
        }

        public void AddHeaders(params string[] headers)
        {
            this.requiredHeaders.AddRange(headers);
        }

        public byte[] PtxForConfig(CudaCompiler compiler, KernelConfig config)
        {
            if (this.ptxCache.TryGetValue(config, out var cachedResult))
            {
                return cachedResult;
            }

            if (!this.requiredConfigArgs.All(config.ContainsKey))
            {
                var allRequired = string.Join(", ", this.requiredConfigArgs);
                throw new InvalidOperationException("All config arguments must be provided. Required: " + allRequired);
            }

            // Checking this ensures that there is only one config argument that can evaluate to the same code,
            // which ensures that the ptx cacheing does not generate unnecessary combinations. Also, a mismatch
            // occurring here probably indicates a bug somewhere else.
            if (!config.Keys.All(this.requiredConfigArgs.Contains))
            {
                var allRequired = string.Join(", ", this.requiredConfigArgs);
                throw new InvalidOperationException("Config provides some unnecessary arguments. Required: " + allRequired);
            }

            //return new DeviceKernelCode(config.ApplyToTemplate(templateCode), requiredHeaders.ToArray());
            var finalCode = config.ApplyToTemplate(this.templateCode);

            var result = compiler.CompileToPtx(finalCode, this.requiredHeaders.ToArray());
            this.ptxCache.Add(config, result);
            return result;
        }
    }
}
