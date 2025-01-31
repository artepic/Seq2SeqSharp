﻿using AdvUtils;
using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using System.Linq;

namespace Seq2SeqSharp
{
    public class BeamSearchStatus
    {
        public List<int> OutputIds;
        public float Score;

        public List<IWeightTensor> HTs;
        public List<IWeightTensor> CTs;

        public BeamSearchStatus()
        {
            this.OutputIds = new List<int>();
            this.HTs = new List<IWeightTensor>();
            this.CTs = new List<IWeightTensor>();

            this.Score = 1.0f;
        }
    }

    public class BeamSearch
    {
        public static List<BeamSearchStatus> GetTopNBSS(List<BeamSearchStatus> bssList, int topN)
        {
            var q = new FixedSizePriorityQueue<ComparableItem<BeamSearchStatus>>(topN, new ComparableItemComparer<BeamSearchStatus>(false));

            foreach (var status in bssList)
            {
                q.Enqueue(new ComparableItem<BeamSearchStatus>(status.Score, status));
            }

            return q.Select(x => x.Value).ToList();
        }
    }
}
