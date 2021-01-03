using AdvUtils;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;

using System;
using System.Collections.Generic;
using System.Linq;

namespace Seq2SeqSharp
{
    public class SequenceLabel : BaseSeq2SeqFramework
    {
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper<FeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after LSTM layers in decoder
                                                                                 //  private CRFDecoder m_crfDecoder;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;

        private readonly float m_dropoutRatio;
        private readonly SeqLabelModelMetaData m_modelMetaData;
        private readonly int m_maxSntSize;

        public SequenceLabel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType,
            float dropoutRatio, Vocab vocab, int[] deviceIds, ProcessorTypeEnums processorType, string modelFilePath, int maxSntSize = 128) :
            base(deviceIds, processorType, modelFilePath)
        {
            this.m_modelMetaData = new SeqLabelModelMetaData(hiddenDim, embeddingDim, encoderLayerDepth, multiHeadNum, encoderType, vocab);
            this.m_dropoutRatio = dropoutRatio;
            this.m_maxSntSize = maxSntSize;

            //Initializng weights in encoders and decoders
            this.CreateTrainableParameters(this.m_modelMetaData);
        }

        public SequenceLabel(string modelFilePath, ProcessorTypeEnums processorType, int[] deviceIds, float dropoutRatio = 0.0f, int maxSntSize = 128)
            : base(deviceIds, processorType, modelFilePath)
        {
            this.m_dropoutRatio = dropoutRatio;
            this.m_modelMetaData = this.LoadModel(this.CreateTrainableParameters) as SeqLabelModelMetaData;
            this.m_maxSntSize = maxSntSize;
        }


        private bool CreateTrainableParameters(IModelMetaData mmd)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            var modelMetaData = mmd as SeqLabelModelMetaData;
            var raDeviceIds = new RoundArray<int>(this.DeviceIds);

            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                this.m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                                                                            new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: true), this.DeviceIds);
                this.m_decoderFFLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", modelMetaData.HiddenDim * 2, modelMetaData.Vocab.TargetWordSize, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), this.DeviceIds);
            }
            else
            {
                this.m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                                                                            new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, this.m_dropoutRatio, raDeviceIds.GetNextItem(), isTrainable: true), this.DeviceIds);
                this.m_decoderFFLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", modelMetaData.HiddenDim, modelMetaData.Vocab.TargetWordSize, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), this.DeviceIds);
            }

            this.m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normal: NormType.Normal, name: "SrcEmbeddings", isTrainable: true), this.DeviceIds);
            //      m_crfDecoder = new CRFDecoder(modelMetaData.Vocab.TargetWordSize);


            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                this.m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(this.BuildPositionWeightTensor(Math.Max(this.m_maxSntSize, this.m_maxSntSize) + 2, modelMetaData.EmbeddingDim, raDeviceIds.GetNextItem(), "PosEmbedding", false), this.DeviceIds, true);
            }
            else
            {
                this.m_posEmbedding = null;
            }

            return true;
        }

        private double CalAngle(double position, double hid_idx, int d_hid)
        {
            return position / Math.Pow(10000, 2 * (hid_idx / 2) / d_hid);
        }

        private WeightTensor BuildPositionWeightTensor(int row, int column, int deviceId, string name = "", bool isTrainable = false)
        {
            var t = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable);
            var posWeights = new float[row * column];

            for (var p = 0; p < row; ++p)
            {
                for (var i = 0; i < column; i += 2)
                {
                    posWeights[p * column + i] = (float)Math.Sin(this.CalAngle(p, i, column));
                    posWeights[p * column + i + 1] = (float)Math.Cos(this.CalAngle(p, i, column));
                }
            }

            t.TWeight.CopyFrom(posWeights);

            return t;
        }


        public void Train(int maxTrainingEpoch, IEnumerable<SntPairBatch> trainCorpus, IEnumerable<SntPairBatch> validCorpus, ILearningRate learningRate, List<IMetric> metrics, AdamOptimizer optimizer)
        {
            Logger.WriteLine("Start to train...");
            for (var i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                this.TrainOneEpoch(i, trainCorpus, validCorpus, learningRate, optimizer, metrics, this.m_modelMetaData, this.RunForwardOnSingleDevice);
            }
        }

        public void Valid(IEnumerable<SntPairBatch> validCorpus, List<IMetric> metrics)
        {
            this.RunValid(validCorpus, this.RunForwardOnSingleDevice, metrics, true);
        }

        public List<List<string>> Test(List<List<string>> inputTokens)
        {
            return this.RunTest(inputTokens, this.RunForwardOnSingleDevice);
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IWeightTensor, IWeightTensor, FeedForwardLayer) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (this.m_encoder.GetNetworkOnDevice(deviceIdIdx), this.m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx), this.m_posEmbedding == null ? null : this.m_posEmbedding.GetNetworkOnDevice(deviceIdIdx), this.m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="g">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side. In training mode, it inputs target tokens, otherwise, it outputs target tokens generated by decoder</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private float RunForwardOnSingleDevice(IComputeGraph g, List<List<string>> srcSnts, List<List<string>> tgtSnts, int deviceIdIdx, bool isTraining)
        {

            var (encoder, srcEmbedding, posEmbedding, decoderFFLayer) = this.GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(g.GetWeightFactory(), srcSnts.Count);


            var originalSrcLengths = ParallelCorpus.PadSentences(srcSnts);
            var seqLen = srcSnts[0].Count;
            var batchSize = srcSnts.Count;

            // Encoding input source sentences
            var encOutput = this.Encode(g, srcSnts, encoder, srcEmbedding, null, posEmbedding, originalSrcLengths);
            var ffLayer = decoderFFLayer.Process(encOutput, batchSize, g);
            var ffLayerBatch = g.TransposeBatch(ffLayer, batchSize);

            var cost = 0.0f;
            using (var probs = g.Softmax(ffLayerBatch, runGradients: false, inPlace: true))
            {
                if (isTraining)
                {
                    //Calculate loss for each word in the batch
                    for (var k = 0; k < batchSize; k++)
                    {
                        for (var j = 0; j < seqLen; j++)
                        {
                            using (var probs_k_j = g.PeekRow(probs, k * seqLen + j, runGradients: false))
                            {
                                var ix_targets_k_j = this.m_modelMetaData.Vocab.GetTargetWordIndex(tgtSnts[k][j]);
                                var score_k = probs_k_j.GetWeightAt(ix_targets_k_j);
                                cost += (float)-Math.Log(score_k);

                                probs_k_j.SetWeightAt(score_k - 1, ix_targets_k_j);
                            }

                        }

                        ////CRF part
                        //using (var probs_k = g.PeekRow(probs, k * seqLen, seqLen, runGradients: false))
                        //{
                        //    var weights_k = probs_k.ToWeightArray();
                        //    var crfOutput_k = m_crfDecoder.ForwardBackward(seqLen, weights_k);

                        //    int[] trueTags = new int[seqLen];
                        //    for (int j = 0; j < seqLen; j++)
                        //    {
                        //        trueTags[j] = m_modelMetaData.Vocab.GetTargetWordIndex(tgtSnts[k][j]);
                        //    }
                        //    m_crfDecoder.UpdateBigramTransition(seqLen, crfOutput_k, trueTags);
                        //}

                    }

                    ffLayerBatch.CopyWeightsToGradients(probs);
                }
                else
                {
                    // CRF decoder
                    //for (int k = 0; k < batchSize; k++)
                    //{
                    //    //CRF part
                    //    using (var probs_k = g.PeekRow(probs, k * seqLen, seqLen, runGradients: false))
                    //    {
                    //        var weights_k = probs_k.ToWeightArray();

                    //        var crfOutput_k = m_crfDecoder.DecodeNBestCRF(weights_k, seqLen, 1);
                    //        var targetWords = m_modelMetaData.Vocab.ConvertTargetIdsToString(crfOutput_k[0].ToList());
                    //        tgtSnts.Add(targetWords);
                    //    }
                    //}


                    // Output "i"th target word
                    var targetIdx = g.Argmax(probs, 1);
                    var targetWords = this.m_modelMetaData.Vocab.ConvertTargetIdsToString(targetIdx.ToList());

                    for (var k = 0; k < batchSize; k++)
                    {
                        tgtSnts[k] = targetWords.GetRange(k * seqLen, seqLen);
                    }
                }

            }

            return cost;
        }

        /// <summary>
        /// Encode source sentences and output encoded weights
        /// </summary>
        /// <param name="g"></param>
        /// <param name="srcSnts"></param>
        /// <param name="encoder"></param>
        /// <param name="reversEncoder"></param>
        /// <param name="Embedding"></param>
        /// <returns></returns>
        private IWeightTensor Encode(IComputeGraph g, List<List<string>> srcSnts, IEncoder encoder, IWeightTensor Embedding, IWeightTensor srcSelfMask, IWeightTensor posEmbedding, List<int> originalSrcLengths)
        {
            var seqLen = srcSnts[0].Count;
            var batchSize = srcSnts.Count;

            var inputs = new List<IWeightTensor>();

            // Generate batch-first based input embeddings
            for (var j = 0; j < batchSize; j++)
            {
                var originalLength = originalSrcLengths[j];
                for (var i = 0; i < seqLen; i++)
                {
                    var ix_source = this.m_modelMetaData.Vocab.GetSourceWordIndex(srcSnts[j][i], logUnk: true);

                    var emb = g.PeekRow(Embedding, ix_source, runGradients: i < originalLength ? true : false);

                    inputs.Add(emb);
                }
            }

            var inputEmbs = g.ConcatRows(inputs);

            if (this.m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                inputEmbs = this.AddPositionEmbedding(g, posEmbedding, batchSize, seqLen, inputEmbs);
            }


            return encoder.Encode(inputEmbs, batchSize, g, srcSelfMask);
        }


        private IWeightTensor AddPositionEmbedding(IComputeGraph g, IWeightTensor posEmbedding, int batchSize, int seqLen, IWeightTensor inputEmbs)
        {
            using (var posEmbeddingPeek = g.PeekRow(posEmbedding, 0, seqLen, false))
            {
                using (var posEmbeddingPeekView = g.View(posEmbeddingPeek, runGradient: false, dims: new long[] { 1, seqLen, this.m_modelMetaData.EmbeddingDim }))
                {
                    using (var posEmbeddingPeekViewExp = g.Expand(posEmbeddingPeekView, runGradient: false, dims: new long[] { batchSize, seqLen, this.m_modelMetaData.EmbeddingDim }))
                    {
                        inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize, seqLen, this.m_modelMetaData.EmbeddingDim });
                        inputEmbs = g.Add(inputEmbs, posEmbeddingPeekViewExp, true, false);
                        inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize * seqLen, this.m_modelMetaData.EmbeddingDim });
                    }
                }
            }

            inputEmbs = g.Dropout(inputEmbs, batchSize, this.m_dropoutRatio, inPlace: true);

            return inputEmbs;
        }


        //internal override void SaveParameters(Stream stream)
        //{
        //    Logger.WriteLine($"Saving parameters for sequence labeling model.");
        //    base.SaveParameters(stream);
        //    m_crfDecoder.Save(stream);
        //}

        //internal override void LoadParameters(Stream stream)
        //{
        //    Logger.WriteLine($"Loading parameters for sequence labeling model.");
        //    base.LoadParameters(stream);
        //    m_crfDecoder.Load(stream);
        //}
    }
}
