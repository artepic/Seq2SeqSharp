using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Core;

namespace TensorSharp
{
    [Serializable]
    public class Tensor : IDisposable
    {
        private long[] sizes;
        private long[] strides;
        private readonly Storage storage;
        private readonly long storageOffset;

        private bool isDisposed = false;


        /// <summary>
        /// Construct a new tensor, using the given allocator to construct a storage. The new tensor
        /// will be contiguous in memory. The tensor's elements will not be initialized.
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="elementType"></param>
        /// <param name="sizes"></param>
        public Tensor(IAllocator allocator, DType elementType, params long[] sizes)
            : this(allocator, elementType, sizes, TensorDimensionHelpers.GetContiguousStride(sizes))
        {
        }

        public Tensor(IAllocator allocator, DType elementType, long[] sizes, long[] strides)
        {
            this.sizes = sizes;
            this.strides = strides;
            this.storageOffset = 0;
            this.storage = allocator.Allocate(elementType, TensorDimensionHelpers.GetStorageSize(sizes, strides));
        }

        public Tensor(long[] sizes, long[] strides, Storage storage, long storageOffset)
        {
            this.sizes = sizes;
            this.strides = strides;
            this.storage = storage;
            this.storageOffset = storageOffset;

            this.storage.AddRef();
        }

        //~Tensor()
        //{
        //    if (!isDisposed)
        //    {
        //        Dispose();
        //    }
        //}

        public override string ToString()
        {
            return TensorFormatting.FormatTensorTypeAndSize(this);
        }

        public void Dispose()
        {
            if (!this.isDisposed)
            {
                this.isDisposed = true;
                this.storage.Release();
            }
            else
            {
                throw new ObjectDisposedException("Tensor");
            }
        }

        public override bool Equals(object obj)
        {
            var o = obj as Tensor;
            if (o == null)
            {
                return false;
            }

            return
                ReferenceEquals(this.storage, o.storage) &&
                this.storageOffset == o.storageOffset &&
                TensorResultBuilder.ArrayEqual(this.sizes, o.sizes) &&
                TensorResultBuilder.ArrayEqual(this.strides, o.strides);
        }

        public override int GetHashCode()
        {
            return this.storage.GetHashCode() ^
                   this.storageOffset.GetHashCode() ^
                   this.sizes.Aggregate(0, (acc, item) => acc ^ item.GetHashCode()) ^
                   this.strides.Aggregate(0, (acc, item) => acc ^ item.GetHashCode());
        }

        public DType ElementType => this.storage.ElementType;
        public long[] Sizes => this.sizes;
        public long[] Strides => this.strides;
        public Storage Storage => this.storage;
        public long StorageOffset => this.storageOffset;
        public IAllocator Allocator => this.storage.Allocator;

        public int DimensionCount => this.sizes.Length;


        /// <summary>
        /// Returns a new Tensor object which points to the same storage as this,
        /// incrementing the refcount of the storage object.
        /// </summary>
        public Tensor CopyRef()
        {
            return new Tensor(this.sizes, this.strides, this.storage, this.storageOffset);
        }

        public string Format()
        {
            return TensorFormatting.Format(this);
        }

        private long? elementCount = null;
        public long ElementCount()
        {
            if (this.elementCount.HasValue)
            {
                return this.elementCount.Value;
            }

            this.elementCount = TensorDimensionHelpers.ElementCount(this.sizes);
            return this.elementCount.Value;
        }

        public bool IsContiguous()
        {
            long z = 1;
            for (var d = this.sizes.Length - 1; d >= 0; d--)
            {
                if (this.sizes[d] != 1)
                {
                    if (this.strides[d] == z)
                    {
                        z *= this.sizes[d];
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            return true;
        }


        public bool IsSameSizeAs(Tensor other)
        {
            return TensorResultBuilder.ArrayEqual(this.sizes, other.sizes);
        }

        /// <summary>
        /// Note: this does not check whether indices are in range
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public float GetElementAsFloat(params long[] indices)
        {
            if (indices.Length != this.DimensionCount)
            {
                throw new ArgumentException("Number of indices must equal number of tensor dimensions");
            }

            for (var i = 0; i < indices.Length; ++i)
            {
                if (indices[i] < 0 || indices[i] >= this.Sizes[i])
                {
                    throw new ArgumentException("Index " + i + " with value " + indices[i] + " is out of range");
                }
            }

            long offset = 0;
            for (var i = 0; i < indices.Length; ++i)
            {
                offset += indices[i] * this.strides[i];
            }

            return this.storage.GetElementAsFloat(this.storageOffset + offset);
        }

        public float[] GetElementsAsFloat(int length)
        {
            return this.storage.GetElementsAsFloat(this.storageOffset, length);
        }

        /// <summary>
        /// Note: this does not check whether indices are in range
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public void SetElementAsFloat(float value, params long[] indices)
        {
            if (indices.Length != this.DimensionCount)
            {
                throw new ArgumentException("Number of indices must equal number of tensor dimensions");
            }

            for (var i = 0; i < indices.Length; ++i)
            {
                if (indices[i] < 0 || indices[i] >= this.Sizes[i])
                {
                    throw new ArgumentException("Index " + i + " with value " + indices[i] + " is out of range");
                }
            }

            long offset = 0;
            for (var i = 0; i < indices.Length; ++i)
            {
                offset += indices[i] * this.strides[i];
            }

            this.storage.SetElementAsFloat(this.storageOffset + offset, value);
        }

        public void SetElementsAsFloat(float[] value)
        {
            this.storage.SetElementsAsFloat(this.storageOffset, value);
        }

        public void SetElementsAsFloat(float[] value, params long[] indices)
        {
            if (indices.Length != this.DimensionCount)
            {
                throw new ArgumentException("Number of indices must equal number of tensor dimensions");
            }

            for (var i = 0; i < indices.Length; ++i)
            {
                if (indices[i] < 0 || indices[i] >= this.Sizes[i])
                {
                    throw new ArgumentException("Index " + i + " with value " + indices[i] + " is out of range");
                }
            }

            long offset = 0;
            for (var i = 0; i < indices.Length; ++i)
            {
                offset += indices[i] * this.strides[i];
            }

            this.storage.SetElementsAsFloat(this.storageOffset + offset, value);
        }


        private string PrintSizes(long[] sizes)
        {
            var sb = new StringBuilder();
            foreach (var size in sizes)
            {
                sb.Append(size);
                sb.Append(" ");
            }

            return sb.ToString();
        }

        public int[] GetElementsAsInt(int length)
        {
            return this.storage.GetElementsAsInt(this.storageOffset, length);
        }

        public void SetElementsAsInt(int[] value)
        {
            this.storage.SetElementsAsInt(this.storageOffset, value);
        }

        public Tensor View(params long[] sizes)
        {
            if (!this.IsContiguous())
            {
                throw new InvalidOperationException("Cannot use View on a non-contiguous tensor");
            }

            if (this.ElementCount() != TensorDimensionHelpers.ElementCount(sizes))
            {
                throw new InvalidOperationException($"Output tensor must have the same number of elements as the input. Size = {this.PrintSizes(this.sizes)}, New Size = {this.PrintSizes(sizes)}");
            }

            return new Tensor(sizes, TensorDimensionHelpers.GetContiguousStride(sizes), this.storage, this.storageOffset);
        }

        public Tensor Narrow(int dimension, long startIndex, long size)
        {
            if (dimension < 0 || dimension >= this.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension");
            }

            if (startIndex < 0 || startIndex >= this.sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("startIndex", $"startIndex = '{startIndex}', sizes[dimension] = '{this.sizes[dimension]}', dimension = '{dimension}', size = '{size}'");
            }

            if (size <= 0 || startIndex + size > this.sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("size", $"startIndex = '{startIndex}', sizes[dimension] = '{this.sizes[dimension]}', dimension = '{dimension}', size = '{size}'");
            }

            var newOffset = this.storageOffset + startIndex * this.strides[dimension];
            var newSizes = (long[])this.sizes.Clone();
            newSizes[dimension] = size;

            return new Tensor(newSizes, this.strides, this.storage, newOffset);
        }

        public Tensor Select(int dimension, long index)
        {
            if (this.DimensionCount == 1)
            {
                throw new InvalidOperationException("Select requires 2 or more dimensions");
            }

            if (dimension < 0 || dimension >= this.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension");
            }

            if (index < 0 || index >= this.sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("index");
            }

            var result = this.Narrow(dimension, index, 1);
            result.sizes = ArrayRemove(this.sizes, dimension);
            result.strides = ArrayRemove(this.strides, dimension);

            return result;
        }


        public Tensor Transpose()
        {
            if (this.DimensionCount != 2)
            {
                throw new InvalidOperationException("Parameterless Transpose is only valid on 2d tensors");
            }

            return this.Transpose(0, 1);
        }

        public Tensor Transpose(int dimension1, int dimension2)
        {
            if (dimension1 < 0 || dimension1 >= this.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension1");
            }

            if (dimension2 < 0 || dimension2 >= this.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension2");
            }

            if (dimension1 == dimension2)
            {
                this.storage.AddRef();
                return this;
            }

            var newSizes = (long[])this.sizes.Clone();
            var newStrides = (long[])this.strides.Clone();
            ArraySwap(newSizes, dimension1, dimension2);
            ArraySwap(newStrides, dimension1, dimension2);
            return new Tensor(newSizes, newStrides, this.storage, this.storageOffset);
        }

        public Tensor Permute(params int[] dims)
        {
            if (dims.Length != this.DimensionCount)
            {
                throw new InvalidOperationException("The number of permutation indices must equal the number of tensor dimensions");
            }

            var result = this.CopyRef();
            foreach (var swap in SwapsForPermutation(dims))
            {
                var resultOld = result;
                result = result.Transpose(swap.Item1, swap.Item2);
                resultOld.Dispose();
            }

            return result;
        }

        /// <summary>
        /// Expand one or more singleton dimensions (dimensions with size 1) by using a stride of 0
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="sizes"></param>
        /// <returns></returns>
        public Tensor Expand(params long[] newSizes)
        {
            if (newSizes.Length != this.DimensionCount)
            {
                throw new InvalidOperationException("number of elements of newSizes must match the dimension count of tensor");
            }

            var newStrides = (long[])this.strides.Clone();
            for (var i = 0; i < newSizes.Length; ++i)
            {
                if (newSizes[i] != this.Sizes[i])
                {
                    if (this.Sizes[i] != 1)
                    {
                        throw new InvalidOperationException("Can only expand singleton dimensions (dimensions of size 1)");
                    }

                    newStrides[i] = 0;
                }
            }

            return new Tensor(newSizes, newStrides, this.storage, this.storageOffset);
        }


        /// <summary>
        /// Return a new tensor where **all** singleton dimensions have been removed
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public Tensor Squeeze()
        {
            var newSizeStrides = this.sizes.Zip(this.strides, Tuple.Create)
                                     .Where(x => x.Item1 != 1)
                                     .ToArray();

            var newSizes = newSizeStrides.Select(x => x.Item1).ToArray();
            var newStrides = newSizeStrides.Select(x => x.Item2).ToArray();

            return new Tensor(newSizes, newStrides, this.storage, this.storageOffset);
        }


        /// <summary>
        /// Return a new tensor where the given singleton dimension has been removed
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns></returns>
        public Tensor Squeeze(int dimension)
        {
            if (this.DimensionCount == 1)
            {
                throw new InvalidOperationException("Squeeze requires 2 or more dimensions");
            }

            if (dimension < 0 || dimension >= this.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension");
            }

            var newSizes = ArrayRemove(this.sizes, dimension);
            var newStrides = ArrayRemove(this.strides, dimension);

            return new Tensor(newSizes, newStrides, this.storage, this.storageOffset);
        }




        /// <summary>
        /// Returns a tensor which contains all slices of size size in the given dimension. The step between two slices is given by step.
        /// The result tensor has an additional dimension of size size.
        /// </summary>
        /// <param name="dimension"></param>
        /// <param name="size"></param>
        /// <param name="step"></param>
        /// <returns></returns>
        public Tensor Unfold(int dimension, long size, long step)
        {
            if (this.DimensionCount == 0)
            {
                throw new InvalidOperationException("Cannot unfold an empty tensor");
            }

            if (dimension < 0 || dimension >= this.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dimension is out of range", "dimension");
            }

            if (size > this.sizes[dimension])
            {
                throw new ArgumentOutOfRangeException("size cannot be larger than the size of dimension", "size");
            }

            if (step <= 0)
            {
                throw new ArgumentOutOfRangeException("step must be at least 1", "step");
            }

            var newSize = new long[this.DimensionCount + 1];
            var newStrides = new long[this.DimensionCount + 1];
            Array.Copy(this.sizes, newSize, this.DimensionCount);
            Array.Copy(this.strides, newStrides, this.DimensionCount);

            newSize[this.DimensionCount] = size;
            newStrides[this.DimensionCount] = this.strides[dimension];

            newSize[dimension] = (this.sizes[dimension] - size) / step + 1;
            newStrides[dimension] = step * this.strides[dimension];

            return new Tensor(newSize, newStrides, this.Storage, this.StorageOffset);
        }


        // Pad array by prepending with 1 until its length equals newSize
        private static long[] Pad1Prepend(long[] array, int newSize)
        {
            var result = new long[newSize];

            // Fill new extra elements with 1
            for (var i = 0; i < newSize - array.Length; ++i)
            {
                result[i] = 1;
            }

            // Copy array to the last array.Length elements of result
            Array.Copy(array, 0, result, newSize - array.Length, array.Length);

            return result;
        }

        // Prepend singleton dimensions until DimensionCount equals newDimCount
        private Tensor PadToDimCount(int newDimCount)
        {
            var newSizes = Pad1Prepend(this.sizes, newDimCount);

            var newStrides = TensorDimensionHelpers.GetContiguousStride(newSizes);
            Array.Copy(this.strides, 0, newStrides, newStrides.Length - this.strides.Length, this.strides.Length);

            return new Tensor(newSizes, newStrides, this.storage, this.storageOffset);
        }

        public Tensor RepeatTensor(params long[] repetitions)
        {
            if (repetitions.Length < this.DimensionCount)
            {
                throw new InvalidOperationException("repetitions must be at least the same length as the number of tensor dimensions");
            }

            if (repetitions.Any(x => x < 1))
            {
                throw new InvalidOperationException("All dimensions must be repeated at least once");
            }

            var paddedSrc = this.PadToDimCount(repetitions.Length);
            var resultSize = paddedSrc.Sizes.Zip(repetitions, (s, r) => s * r).ToArray();

            var result = new Tensor(this.Allocator, this.ElementType, resultSize);

            var urTensor = result.CopyRef();
            for (var i = 0; i < paddedSrc.DimensionCount; ++i)
            {
                var oldUrTensor = urTensor;
                urTensor = urTensor.Unfold(i, paddedSrc.Sizes[i], paddedSrc.Sizes[i]);
                oldUrTensor.Dispose();
            }

            var paddedSrc2 = paddedSrc.PadToDimCount(urTensor.DimensionCount);
            var expandedSrc = paddedSrc2.Expand(urTensor.Sizes);
            Ops.Copy(urTensor, expandedSrc);

            paddedSrc.Dispose();
            paddedSrc2.Dispose();
            urTensor.Dispose();
            expandedSrc.Dispose();

            /*
            var sizesWritten = (long[])this.sizes.Clone();
            using (var subResult = result.GetRegion(Enumerable.Repeat((long)0, DimensionCount).ToArray(), sizesWritten))
            {
                Ops.Copy(subResult, this);
            }

            for (int i = 0; i < repetitions.Length; ++i)
            {
                if (repetitions[i] == 1) continue;

                sizesWritten[i] *= repetitions[i];
                using (var subResultSrc = result.GetRegion(Enumerable.Repeat((long)0, DimensionCount).ToArray(), this.sizes))
                using (var subResultTgt = result.GetRegion(Enumerable.Repeat((long)0, DimensionCount).ToArray(), this.sizes))
                {
                    Ops.Copy(subResultTgt, subResultSrc);
                }
            }*/

            return result;
        }

        /*
            private Tensor GetRegion(long[] dimensionStarts, long[] dimensionSizes)
        {
            var result = this.CopyRef();
            for (int i = 0; i < dimensionStarts.Length; ++i)
            {
                var resultOld = result;
                result.Narrow(i, dimensionStarts[i], dimensionSizes[i]);
                resultOld.Dispose();
            }
            return result;
        }*/


        public void CopyFrom(Array array)
        {
            var elementType = DTypeBuilder.FromCLRType(array.GetType().GetElementType());

            if (!this.IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous to copy from CLR array");
            }

            if (this.ElementCount() != array.LongLength)
            {
                throw new InvalidOperationException("Tensor and array must have the same number of elements");
            }

            if (this.ElementType != elementType)
            {
                throw new InvalidOperationException("Tensor and array must have the same element types");
            }

            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            try
            {
                var length = Buffer.ByteLength(array);
                this.Storage.CopyToStorage(this.StorageOffset, handle.AddrOfPinnedObject(), length);
            }
            finally
            {
                handle.Free();
            }
        }


        public void CopyToArray(Array array)
        {
            var elementType = DTypeBuilder.FromCLRType(array.GetType().GetElementType());

            if (!this.IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous to copy from CLR array");
            }

            if (this.ElementCount() != array.LongLength)
            {
                throw new InvalidOperationException("Tensor and array must have the same number of elements");
            }

            if (this.ElementType != elementType)
            {
                throw new InvalidOperationException("Tensor and array must have the same element types");
            }

            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            try
            {
                var length = Buffer.ByteLength(array);
                this.Storage.CopyFromStorage(handle.AddrOfPinnedObject(), this.StorageOffset, length);
            }
            finally
            {
                handle.Free();
            }
        }


        public static Tensor FromArray(IAllocator allocator, Array array)
        {
            // From the CLI spec(section 8.9.1):
            // Array elements shall be laid out within the array object in row - major order
            // (i.e., the elements associated with the rightmost array dimension shall be laid out contiguously from lowest to highest index).
            // The actual storage allocated for each array element can include platform - specific padding.

            // This is already in the order we want - and here we will (potentially incorrectly) assume that there is no
            // 'platform-specific padding'. This appears to be a reasonable assumption on both CLR and Mono.
            // Assuming no platform-specific padding allows us to use memcpy instead of iterating and copying each element

            var elementType = DTypeBuilder.FromCLRType(array.GetType().GetElementType());

            var dimSizes =
                Enumerable.Range(0, array.Rank)
                .Select(x => (long)array.GetLength(x))
                .ToArray();

            var result = new Tensor(allocator, elementType, dimSizes);
            result.CopyFrom(array);
            return result;
        }


        private static void ArraySwap<T>(T[] array, int index1, int index2)
        {
            var temp = array[index1];
            array[index1] = array[index2];
            array[index2] = temp;
        }

        // Return a copy of an array, but with the item at index removed
        private static T[] ArrayRemove<T>(T[] source, long index)
        {
            var result = new T[source.Length - 1];
            for (var i = 0; i < result.Length; ++i)
            {
                if (i < index)
                {
                    result[i] = source[i];
                }
                else
                {
                    result[i] = source[i + 1];
                }
            }
            return result;
        }

        // Convert a permutation into a sequence of swap operations.
        // perm must contain a permuation of the indices [0, perm.Length)
        // The returned tuples indicate pairs of indices that should be swapped. The swaps
        // must be performed in the given order.
        private static IEnumerable<Tuple<int, int>> SwapsForPermutation(int[] perm)
        {
            int j;
            for (var i = 0; i < perm.Length; ++i)
            {
                var p = perm[i];
                if (p != i && p != -1)
                {
                    j = i;
                    do
                    {
                        if (perm[j] < 0 || perm[j] >= perm.Length)
                        {
                            throw new InvalidOperationException("Invalid permutation");
                        }

                        yield return Tuple.Create(j, perm[j]);


                        var jOld = j;
                        j = perm[j];
                        perm[jOld] = -1;
                    } while (perm[j] != i);
                    perm[j] = j;
                }
            }
        }


        public void Serialize(System.IO.Stream stream)
        {
            TensorSerialization.Serialize(this, stream);
        }

        public static Tensor Deserialize(IAllocator allocator, System.IO.Stream stream)
        {
            return TensorSerialization.Deserialize(allocator, stream);
        }
    }
}
