using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp.Utils
{
    public class MaskUtils
    {

        public static IWeightTensor BuildPadSelfMask(IComputeGraph g, int paddedLength, List<int> originalLengths, int deviceId)
        {
            var buf = new float[originalLengths.Count * paddedLength * paddedLength];
            for (var i = 0; i < buf.Length; i++)
            {
                buf[i] = -1e30f;
            }

            for (var k = 0; k < originalLengths.Count; k++)
            {
                for (var i = 0; i < originalLengths[k]; i++)
                {
                    for (var j = 0; j < originalLengths[k]; j++)
                    {
                        // ReSharper disable once ArrangeRedundantParentheses
                        buf[k * (paddedLength * paddedLength) + i * paddedLength + j] = 0.0f;
                    }
                }
            }

            var tensor = new WeightTensor(new long[] { originalLengths.Count, paddedLength, paddedLength }, 0.0f, deviceId, $"TriMask_{deviceId}", false);
            tensor.SetWeightArray(buf);

            return tensor;
        }

        public static IWeightTensor BuildPadSelfTriMask(IComputeGraph g, int paddedLength, List<int> originalLengths, int deviceId)
        {
            var buf = new float[originalLengths.Count * paddedLength * paddedLength];
            Array.Fill(buf, -1e30f);

            for (var k = 0; k < originalLengths.Count; k++)
            {
                // ReSharper disable once ArrangeRedundantParentheses
                var offset_k = k * (paddedLength * paddedLength);
                for (var i = 0; i < originalLengths[k]; i++)
                {
                    var offset_k_i = offset_k + i * paddedLength;
                    for (var j = 0; j < originalLengths[k]; j++)
                    {
                        if (i >= j)
                        {
                            buf[offset_k_i + j] = 0.0f;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }

            var tensor = new WeightTensor(new long[] { originalLengths.Count, paddedLength, paddedLength }, deviceId, $"TriMask_{deviceId}", false);
            tensor.SetWeightArray(buf);


            //WeightTensor tensor = new WeightTensor(new long[] { originalLengths.Count, paddedLength, paddedLength }, deviceId, $"TriMask_{deviceId}", isTrainable: false);
            //using (Tensor originalLengthsTensor = new Tensor(TensorAllocator.Allocator(deviceId), DType.Int32, 1, originalLengths.Count))
            //{
            //    originalLengthsTensor.SetElementsAsInt(originalLengths.ToArray());
            //    tensor.TWeight = Ops.BuildPadSelfTriMask(originalLengthsTensor, originalLengths.Count, paddedLength);
            //}

            return tensor;
        }



        public static IWeightTensor BuildSrcTgtMask(IComputeGraph g, int srcPaddedLength, int tgtPaddedLength, List<int> tgtOriginalLengths, List<int> srcOriginalLengths, int deviceId)
        {
            var buf = new float[tgtOriginalLengths.Count * tgtPaddedLength * srcPaddedLength];
            Array.Fill(buf, -1e30f);

            for (var k = 0; k < tgtOriginalLengths.Count; k++) // batch size
            {
                // ReSharper disable once ArrangeRedundantParentheses
                var offset_k = k * (tgtPaddedLength * srcPaddedLength);
                for (var i = 0; i < tgtOriginalLengths[k]; i++)
                {
                    var offset_k_i = offset_k + i * srcPaddedLength;
                    for (var j = 0; j < srcOriginalLengths[k]; j++)
                    {
                        buf[offset_k_i + j] = 0.0f;
                    }
                }
            }

            var tensor = new WeightTensor(new long[] { tgtOriginalLengths.Count, tgtPaddedLength, srcPaddedLength }, deviceId, $"SrcTgtMask_{deviceId}", false);
            tensor.SetWeightArray(buf);

            return tensor;

            //WeightTensor tensor = new WeightTensor(new long[] { tgtOriginalLengths.Count, tgtPaddedLength, srcPaddedLength }, deviceId, $"SrcTgtMask_{deviceId}", isTrainable: false);

            //using (Tensor srcOriginalLengthsTensor = new Tensor(TensorAllocator.Allocator(deviceId), DType.Int32, 1, srcOriginalLengths.Count))
            //{
            //    using (Tensor tgtOriginalLengthsTensor = new Tensor(TensorAllocator.Allocator(deviceId), DType.Int32, 1, tgtOriginalLengths.Count))
            //    {
            //        srcOriginalLengthsTensor.SetElementsAsInt(srcOriginalLengths.ToArray());
            //        tgtOriginalLengthsTensor.SetElementsAsInt(tgtOriginalLengths.ToArray());

            //        tensor.TWeight = Ops.BuildSrcTgtMask(srcOriginalLengthsTensor, tgtOriginalLengthsTensor, srcOriginalLengths.Count, srcPaddedLength, tgtPaddedLength);

            //        return tensor;
            //    }
            //}


        }
    }
}
