using System.Text;

namespace TensorSharp.CUDA.DeviceCode
{
    public class PermutationGenerator
    {
        public readonly StringBuilder sb = new();

        public override string ToString()
        {
            return this.sb.ToString();
        }

        public void AddApplyT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* v) const {{ {operatorCode} }} }};");
                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> src, __int64 totalElements)");
                this.sb.AppendLine("   {");

                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, src);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName}()(&src.data[aOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");
            }
        }

        public void AddApplyTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};");
                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, __int64 totalElements)");
                this.sb.AppendLine("   {");

                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                this.sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");


            }
        }

        public void AddApplyTTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(3))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c) const {{ {operatorCode} }} }};");
                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, __int64 totalElements)");
                this.sb.AppendLine("   {");

                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                this.sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                this.sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");

            }
        }

        public void AddApplyTTTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(4))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                var dimsD = spec.TensorDims[3].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c, float *d) const {{ {operatorCode} }} }};");
                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, TensorInfo<{indexType}> tensorD, __int64 totalElements)");
                this.sb.AppendLine("   {");

                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                this.sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                this.sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                this.sb.AppendLine($"         const {indexType} dOffset = IndexToOffset < {indexType}, {dimsD}>::get(linearIndex, tensorD);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset], &tensorD.data[dOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");
            }
        }

        public void AddApplyTTTTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(5))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                var dimsD = spec.TensorDims[3].ToString();
                var dimsE = spec.TensorDims[4].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c, float *d, float *e) const {{ {operatorCode} }} }};");
                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, TensorInfo<{indexType}> tensorD, TensorInfo<{indexType}> tensorE, __int64 totalElements)");
                this.sb.AppendLine("   {");

                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                this.sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                this.sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                this.sb.AppendLine($"         const {indexType} dOffset = IndexToOffset < {indexType}, {dimsD}>::get(linearIndex, tensorD);");
                this.sb.AppendLine($"         const {indexType} eOffset = IndexToOffset < {indexType}, {dimsE}>::get(linearIndex, tensorE);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset], &tensorD.data[dOffset], &tensorE.data[eOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");
            }
        }
        public void AddApplyTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                this.sb.AppendLine("float b;");
                this.sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float bVal) {{ this->b = bVal; }}");
                this.sb.AppendLine($"__device__ __forceinline__ void operator()(float* a) const {{ {operatorCode} }}");
                this.sb.AppendLine("};");

                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, __int64 totalElements)");
                this.sb.AppendLine("   {");

                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, a);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b);");
                this.sb.AppendLine($"         op(&a.data[aOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");
            }
        }

        public void AddApplyTSS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                this.sb.AppendLine("float b;");
                this.sb.AppendLine("float c;");
                this.sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float bVal, float cVal) {{ this->b = bVal; this->c = cVal; }}");
                this.sb.AppendLine($"__device__ __forceinline__ void operator()(float* a) const {{ {operatorCode} }}");
                this.sb.AppendLine("};");

                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, float c, __int64 totalElements)");
                this.sb.AppendLine("   {");
                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, a);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b, c);");
                this.sb.AppendLine($"         op(&a.data[aOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");
            }
        }

        public void AddApplyTTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                this.sb.AppendLine("float c;");
                this.sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float cVal) {{ this->c = cVal; }}");
                this.sb.AppendLine($"__device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};");

                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, float c, __int64 totalElements)");
                this.sb.AppendLine("   {");
                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                this.sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(c);");
                this.sb.AppendLine($"         op(&tensorA.data[aOffset], &tensorB.data[bOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");

            }
        }

        public void AddApplyTTSS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                this.sb.AppendLine("float c;");
                this.sb.AppendLine("float d;");
                this.sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float cVal, float dVal) {{ this->c = cVal; this->d = dVal; }}");
                this.sb.AppendLine($"__device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};");

                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, float c, float d, __int64 totalElements)");
                this.sb.AppendLine("   {");
                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                this.sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(c, d);");
                this.sb.AppendLine($"         op(&tensorA.data[aOffset], &tensorB.data[bOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");
            }
        }

        public void AddApplyTTTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(3))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();

                this.sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                this.sb.AppendLine("float d;");
                this.sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float dVal) {{ this->d = dVal; }}");
                this.sb.AppendLine($"__device__ __forceinline__ void operator()(float* a, float *b, float *c) const {{ {operatorCode} }} }};");

                this.sb.AppendLine("extern \"C\" {");
                this.sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, float d, __int64 totalElements)");
                this.sb.AppendLine("   {");
                this.sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                this.sb.AppendLine("      {");
                this.sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                this.sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                this.sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                this.sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(d);");
                this.sb.AppendLine($"         op(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset]);");
                this.sb.AppendLine("      }");
                this.sb.AppendLine("   }");
                this.sb.AppendLine("}");
            }
        }

        public void AddReduce(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                this.sb.AppendFormat("REDUCE_KERNELS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        public void AddReduceNorm(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                this.sb.AppendFormat("REDUCE_NORM_KERNELS({0}, {1}, {2}, {3})\n", indexType, dimsA, dimsB, kernelName);
            }
        }

        public void AddReduceAll(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                this.sb.AppendFormat("REDUCE_ALL_KERNELS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        public void AddReduceAllNorm(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                this.sb.AppendFormat("REDUCE_ALL_NORM_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }

        public void AddReduceAllSubSquare(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                this.sb.AppendFormat("REDUCE_ALL_SUB_SQUARE_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }


        // TODO make member of ApplySpecialization
        public static string GetMangledName(string baseName, ApplySpecialization spec)
        {
            var sb = new StringBuilder();

            sb.Append(baseName);
            sb.Append(spec.Use32BitIndices ? "__int32" : "__int64");
            foreach (var dimSize in spec.TensorDims)
            {
                sb.Append("_").Append(dimSize.ToString().Replace('-', 'M'));
            }
            return sb.ToString();
        }
    }
}
