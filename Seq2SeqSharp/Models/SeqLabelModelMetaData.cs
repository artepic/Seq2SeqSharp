﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    [Serializable]
    public class SeqLabelModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int EmbeddingDim;
        public int EncoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public Vocab Vocab;

        public SeqLabelModelMetaData()
        {

        }

        public SeqLabelModelMetaData(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab vocab)
        {
            this.HiddenDim = hiddenDim;
            this.EmbeddingDim = embeddingDim;
            this.EncoderLayerDepth = encoderLayerDepth;
            this.MultiHeadNum = multiHeadNum;
            this.EncoderType = encoderType;
            this.Vocab = vocab;
        }
    }
}
