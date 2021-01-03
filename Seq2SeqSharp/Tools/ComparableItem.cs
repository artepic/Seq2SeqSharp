using System;
using System.Collections.Generic;

namespace Seq2SeqSharp.Tools
{

    public class ComparableItem<T>
    {
        public float Score { get; }
        public T Value { get; }

        public ComparableItem(float score, T value)
        {
            this.Score = score;
            this.Value = value;
        }
    }

    public class ComparableItemComparer<T> : IComparer<ComparableItem<T>>
    {
        public ComparableItemComparer(bool fAscending)
        {
            this.m_fAscending = fAscending;
        }

        public int Compare(ComparableItem<T> x, ComparableItem<T> y)
        {
            var iSign = Math.Sign(x.Score - y.Score);
            if (!this.m_fAscending)
            {
                iSign = -iSign;
            }

            return iSign;
        }

        protected bool m_fAscending;
    }
}
