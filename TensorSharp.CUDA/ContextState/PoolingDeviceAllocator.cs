using AdvUtils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;

namespace TensorSharp.CUDA.ContextState
{
    public class PoolingDeviceAllocator : IDeviceAllocator
    {
        private const long MemoryAlignment = 256;

        private readonly CudaContext m_context;
        private readonly object locker = new();

        private readonly ulong m_ulAvailMemByteInTotal;
        private CUdeviceptr m_memPoolPtr;
        private readonly SizeT m_startMemAddr;
        private readonly SizeT m_endMemAddr;

        private SortedDictionary<ulong, ulong> m_usedAddr2Size;

        public PoolingDeviceAllocator(CudaContext context, float memoryUsageRatio = 0.9f)
        {
            this.m_context = context;
            context.SetCurrent();

            this.m_ulAvailMemByteInTotal = (ulong)((ulong)context.GetFreeDeviceMemorySize() * memoryUsageRatio);

            this.m_memPoolPtr = context.AllocateMemory(this.m_ulAvailMemByteInTotal);

            this.m_startMemAddr = this.m_memPoolPtr.Pointer;
            this.m_endMemAddr = this.m_startMemAddr + this.m_ulAvailMemByteInTotal;

            this.m_usedAddr2Size = new SortedDictionary<ulong, ulong>();

            Logger.WriteLine($"Allocated Cuda memory: {this.m_ulAvailMemByteInTotal}, address from '{this.m_startMemAddr}' to '{this.m_endMemAddr}'");
        }

        public float GetAllocatedMemoryRatio()
        {
            lock (this.locker)
            {
                ulong allocatedMemByte = 0;
                foreach (var pair in this.m_usedAddr2Size)
                {
                    allocatedMemByte += pair.Value;
                }

                return (float)((float)allocatedMemByte / (float)this.m_ulAvailMemByteInTotal);
            }
        }

        private CUdeviceptr AllocateMemory(ulong size)
        {
            lock (this.locker)
            {
                var currMemAddr = this.m_startMemAddr;
                SizeT currMemAddrEnd;

                foreach (var kv in this.m_usedAddr2Size)
                {
                    currMemAddrEnd = currMemAddr + size;

                    if (currMemAddrEnd > this.m_endMemAddr)
                    {
                        throw new OutOfMemoryException($"Out of GPU memory. Current memory usage = '{(this.GetAllocatedMemoryRatio() * 100.0f).ToString("F")}%'");
                    }

                    if (currMemAddrEnd < kv.Key)
                    {
                        this.m_usedAddr2Size.Add(currMemAddr, size);
                        return new CUdeviceptr(currMemAddr);
                    }
                    else
                    {
                        currMemAddr = kv.Key + kv.Value;
                    }
                }

                currMemAddrEnd = currMemAddr + size;
                if (currMemAddrEnd > this.m_endMemAddr)
                {
                    throw new OutOfMemoryException($"Out of GPU memory. Current memory usage = '{(this.GetAllocatedMemoryRatio() * 100.0f).ToString("F")}%'");
                }

                this.m_usedAddr2Size.Add(currMemAddr, size);
                return new CUdeviceptr(currMemAddr);
            }
        }

        public IDeviceMemory Allocate(long byteCount)
        {
            var size = PadToAlignment(byteCount, MemoryAlignment);

            lock (this.locker)
            {            
                var buffer = this.AllocateMemory(size);

                BasicDeviceMemory devMemory = null;
                devMemory = new BasicDeviceMemory(buffer, () =>
                {
                    lock (this.locker)
                    {
                        this.m_usedAddr2Size.Remove(devMemory.Pointer.Pointer);
                    }
                });

                return devMemory;
            }
        }

        public void Dispose()
        {
            this.m_context.SetCurrent();
            this.m_context.FreeMemory(this.m_memPoolPtr);
        }

        private static ulong PadToAlignment(long size, long alignment)
        {
            // ReSharper disable once ArrangeRedundantParentheses
            return (ulong)(((size + alignment - 1) / alignment) * alignment);
        }
    }
}
