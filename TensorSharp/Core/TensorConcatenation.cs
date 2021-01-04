using System;
using System.Linq;

namespace TensorSharp.Core
{
    public static class TensorConcatenation
    {
        // Requires an implementation of *copy* for the given tensor types
        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new ArgumentException("Concat: at least two tensors required", nameof(inputs));
            }

            for (var i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] == null)
                {
                    throw new ArgumentException($"Concat: input[{i}] is null.");
                }
            }


            var ndim = Math.Max(dimension, inputs.Max(x => x.DimensionCount));
            var size = ConcatTensorSize(ndim, dimension, inputs);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, inputs[0], false, size);


            // Select each region of the result corresponding to each input tensor,
            // and copy into the result
            long offset = 0;
            foreach (var t in inputs)
            {
                var dimSize = GetDimSize(t, dimension);
                using (var nt = writeTarget.Narrow(dimension, offset, dimSize))
                {
                    Ops.Copy(nt, t);
                }
                offset += dimSize;
            }

            return writeTarget;
        }



        private static long GetDimSize(Tensor tensor, int dim)
        {
            return dim < tensor.DimensionCount ? tensor.Sizes[dim] : 1;
        }

        private static long[] ConcatTensorSize(int ndim, int dimension, Tensor[] tensors)
        {
            var result = new long[ndim];
            for (var i = 0; i < ndim; ++i)
            {
                var dimSize = GetDimSize(tensors[0], i);
                if (i == dimension)
                {
                    for (var j = 1; j < tensors.Length; ++j)
                    {
                        dimSize += GetDimSize(tensors[j], i);
                    }
                }
                else
                {
                    for (var j = 1; j < tensors.Length; ++j)
                    {
                        if (dimSize != GetDimSize(tensors[j], i))
                        {
                            var message = "";
                            for (var k = 0; k < tensors.Length; k++)
                            {
                                message += $"{k}: ({tensors[k].Sizes[0]}, {tensors[k].Sizes[1]}) ";
                            }
                            throw new InvalidOperationException($"Inconsistent tensor sizes. {message}");
                        }
                    }
                }
                result[i] = dimSize;
            }
            return result;
        }

    }
}
