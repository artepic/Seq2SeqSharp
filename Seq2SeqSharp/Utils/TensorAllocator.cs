using AdvUtils;
using System;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace Seq2SeqSharp
{
    public static class TensorAllocator
    {
        private static IAllocator[] allocator;

        private static TSCudaContext context;

        /// <summary>
        /// 
        /// </summary>
        private static int[] deviceIds;

        /// <summary>
        /// The architecture type
        /// </summary>
        private static ProcessorTypeEnums architectureType;


        public static void InitDevices(ProcessorTypeEnums archType, int[] ids, float memoryUsageRatio = 0.9f, string[] compilerOptions = null)
        {
            architectureType = archType;

            if (architectureType == ProcessorTypeEnums.GPU)
            {
                deviceIds = ids;

                foreach (var id in deviceIds)
                {
                    Logger.WriteLine($"Initialize device '{id}'");
                }

                context = new TSCudaContext(deviceIds, memoryUsageRatio, compilerOptions);
                context.Precompile(Console.Write);
                context.CleanUnusedPTX();

                allocator = new IAllocator[deviceIds.Length];
            }
            else
            {
                allocator = new IAllocator[1];
            }
        }

        public static IAllocator Allocator(int deviceId)
        {
            if (architectureType == ProcessorTypeEnums.GPU)
            {
                var index = GetDeviceIdIndex(deviceId);
                return allocator[index] ?? (allocator[index] = new CudaAllocator(context, deviceId));
            }

            return allocator[0] ?? (allocator[0] = new CpuAllocator());
        }

        private static int GetDeviceIdIndex(int id)
        {
            for (var i = 0; i < deviceIds.Length; i++)
            {
                if (deviceIds[i] == id)
                {
                    return i;
                }
            }

            var strIds = string.Empty;
            foreach (var item in deviceIds)
            {
                strIds += $"{strIds} {item}";
            }

            throw new ArgumentException($"Failed to get deviceId '{id}', deviceId List = {strIds}");
        }
    }
}
