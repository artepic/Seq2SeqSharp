using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using TensorSharp.CUDA.ContextState;
using TensorSharp.CUDA.Util;

namespace TensorSharp.CUDA
{
    public struct ScratchSpace
    {
        public int size;
        public CUdeviceptr buffer;
    }

    [Serializable]
    public class TSCudaContext : IDisposable
    {
        public const int MaxDims = 25;
        private const string CacheDir = @"cuda_cache\general";


        //  private readonly int deviceCount;
        private readonly DeviceState[] devices;
        private readonly bool[,] p2pAccess;
        private readonly int[] deviceIds;

        private readonly RuntimeCompiler.KernelDiskCache diskCache;

        private readonly RuntimeCompiler.CudaCompiler compiler;
        private readonly CudaKernelCache kernelCache = new CudaKernelCache();


        public TSCudaContext(int[] deviceIds, float memoryUsageRatio = 0.9f, string[] compilerOptions = null)
        {
            this.deviceIds = deviceIds;

            this.devices = new DeviceState[deviceIds.Length];
            for (var i = 0; i < deviceIds.Length; i++)
            {
                this.devices[i] = new DeviceState(deviceIds[i], memoryUsageRatio);
            }

            this.p2pAccess = EnablePeerAccess(this.devices.Select(x => x.CudaContext).ToArray(), this.devices[0].CudaContext);

            this.diskCache = new RuntimeCompiler.KernelDiskCache(Path.Combine(Environment.CurrentDirectory, CacheDir));
            this.compiler = new RuntimeCompiler.CudaCompiler(this.diskCache, compilerOptions);

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        private int GetDeviceIdIndex(int id)
        {
            for (var i = 0; i < this.deviceIds.Length; i++)
            {
                if (this.deviceIds[i] == id)
                {
                    return i;
                }
            }

            return -1;
        }


        public RuntimeCompiler.CudaCompiler Compiler => this.compiler;
        public CudaKernelCache KernelCache => this.kernelCache;
        //  public int DeviceCount { get { return deviceCount; } }

        public void Dispose()
        {
            this.kernelCache.Dispose();

            foreach (var device in this.devices)
            {
                device.Dispose();
            }
        }

        public void Synchronize(int deviceId)
        {
            var idx = this.GetDeviceIdIndex(deviceId);
            this.devices[idx].CudaContext.Synchronize();
        }

        public void SynchronizeAll()
        {
            foreach (var device in this.devices)
            {
                device.CudaContext.Synchronize();
            }
        }

        public CudaContext CudaContextForDevice(int deviceId)
        {
            var idx = this.GetDeviceIdIndex(deviceId);
            return this.devices[idx].CudaContext;
        }

        public IDeviceAllocator AllocatorForDevice(int deviceId)
        {
            var idx = this.GetDeviceIdIndex(deviceId);
            return this.devices[idx].MemoryAllocator;
        }

        public CudaContext CudaContextForTensor(Tensor tensor)
        {
            return this.CudaContextForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        public ScratchSpace ScratchSpaceForDevice(int deviceId)
        {
            var idx = this.GetDeviceIdIndex(deviceId);
            return this.devices[idx].ScratchSpace;
        }

        public PooledObject<CudaBlas> BlasForDevice(int deviceId)
        {
            var idx = this.GetDeviceIdIndex(deviceId);
            return this.devices[idx].BlasHandles.Get();
        }

        public PooledObject<CudaBlas> BlasForTensor(Tensor tensor)
        {
            return this.BlasForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        public bool CanAccessPeer(int srcDevice, int peerDevice)
        {
            var srcDeviceIdx = this.GetDeviceIdIndex(srcDevice);
            var peerDeviceIdx = this.GetDeviceIdIndex(peerDevice);
            return this.p2pAccess[srcDeviceIdx, peerDeviceIdx];
        }

        public CudaDeviceProperties DeviceInfoForContext(CudaContext cudaContext)
        {
            var idx = this.GetDeviceIdIndex(cudaContext.DeviceId);
            return this.devices[idx].DeviceInfo;
        }



        // Returns a matrix of [i, j] values where [i, j] is true iff device i can access device j
        private static bool[,] EnablePeerAccess(CudaContext[] cudaContexts, CudaContext restoreCurrent)
        {
            var result = new bool[cudaContexts.Length, cudaContexts.Length];

            for (var i = 0; i < cudaContexts.Length; ++i)
            {
                for (var j = 0; j < cudaContexts.Length; ++j)
                {
                    if (i == j)
                    {
                        result[i, j] = true;
                    }
                    else
                    {
                        result[i, j] = EnablePeers(cudaContexts[i], cudaContexts[j]);
                    }
                }
            }

            restoreCurrent.SetCurrent();
            return result;
        }

        private static bool EnablePeers(CudaContext src, CudaContext target)
        {
            if (!src.DeviceCanAccessPeer(target))
            {
                return false;
            }

            src.SetCurrent();

            try
            {
                CudaContext.EnablePeerAccess(target);
                return true;
            }
            catch
            {
                return false;
            }
        }


        public void Precompile()
        {
            this.Precompile(Console.Write);
        }

        public void Precompile(Action<string> precompileProgressWriter)
        {
            var assembly = Assembly.GetExecutingAssembly();
            foreach (var applyType in assembly.TypesWithAttribute<PrecompileAttribute>(true).Where(x => !x.Item1.IsAbstract))
            {
                precompileProgressWriter("Precompiling " + applyType.Item1.Name + "\n");

                var instance = (IPrecompilable)Activator.CreateInstance(applyType.Item1);
                instance.Precompile(this.Compiler);
            }
        }

        public void CleanUnusedPTX()
        {
            this.diskCache.CleanUnused();
        }
    }
}
