using TensorSharp.CUDA.DeviceCode;

namespace TensorSharp.CUDA
{
    [OpsClass]
    public class CudaIndexingOps
    {
        private readonly IndexSelectKernels indexSelect = new();
        private readonly GatherScatterKernels gather = new();


        public CudaIndexingOps()
        {
        }

        [RegisterOpStorageType("index_select", typeof(CudaStorage))]
        public Tensor IndexSelect(Tensor result, Tensor src, int dimension, Tensor indices) { return this.indexSelect.IndexSelect(result, src, dimension, indices); }

        [RegisterOpStorageType("gather", typeof(CudaStorage))]
        public Tensor Gather(Tensor result, Tensor src, int dimension, Tensor indices) { return this.gather.Gather(result, src, dimension, indices); }

        [RegisterOpStorageType("scatter", typeof(CudaStorage))]
        public Tensor Scatter(Tensor result, Tensor src, int dimension, Tensor indices) { return this.gather.Scatter(result, src, dimension, indices); }

        [RegisterOpStorageType("scatter_fill", typeof(CudaStorage))]
        public Tensor ScatterFill(Tensor result, float value, int dimension, Tensor indices) { return this.gather.ScatterFill(result, value, dimension, indices); }
    }
}
