using System;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int EmbeddingDim;
        public int EncoderLayerDepth;
        public int DecoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public DecoderTypeEnums DecoderType;
        public Vocab Vocab;
        public bool EnableCoverageModel = true;

        public Seq2SeqModelMetaData()
        {

        }

        public Seq2SeqModelMetaData(int hiddenDim, int embeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab vocab, bool enableCoverageModel)
        {
            this.HiddenDim = hiddenDim;
            this.EmbeddingDim = embeddingDim;
            this.EncoderLayerDepth = encoderLayerDepth;
            this.DecoderLayerDepth = decoderLayerDepth;
            this.MultiHeadNum = multiHeadNum;
            this.EncoderType = encoderType;
            this.DecoderType = decoderType;
            this.Vocab = vocab;
            this.EnableCoverageModel = enableCoverageModel;
        }
    }
}
