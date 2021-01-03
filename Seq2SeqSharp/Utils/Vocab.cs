using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Seq2SeqSharp
{
    public enum SENTTAGS
    {
        END = 0,
        START,
        UNK
    }

    [Serializable]
    public class Vocab
    {
        public Dictionary<string, int> SrcWordToIndex;
        public Dictionary<string, int> TgtWordToIndex;

        private Dictionary<int, string> m_srcIndexToWord;
        private List<string> m_srcVocab = new List<string>();
        private Dictionary<int, string> m_tgtIndexToWord;
        private List<string> m_tgtVocab = new List<string>();


        public int SourceWordSize => this.m_srcIndexToWord.Count;
        public int TargetWordSize => this.m_tgtIndexToWord.Count;

        public List<string> SrcVocab => this.m_srcVocab.GetRange(3, this.m_srcVocab.Count - 3);

        public List<string> TgtVocab => this.m_tgtVocab.GetRange(3, this.m_tgtVocab.Count - 3);

        private object locker = new object();

        public Vocab()
        {
            this.CreateIndex();
        }

        private void CreateIndex()
        {
            this.SrcWordToIndex = new Dictionary<string, int>();
            this.m_srcIndexToWord = new Dictionary<int, string>();
            this.m_srcVocab = new List<string>();

            this.TgtWordToIndex = new Dictionary<string, int>();
            this.m_tgtIndexToWord = new Dictionary<int, string>();
            this.m_tgtVocab = new List<string>();

            this.m_srcVocab.Add(ParallelCorpus.EOS);
            this.m_srcVocab.Add(ParallelCorpus.BOS);
            this.m_srcVocab.Add(ParallelCorpus.UNK);

            this.SrcWordToIndex[ParallelCorpus.EOS] = (int)SENTTAGS.END;
            this.SrcWordToIndex[ParallelCorpus.BOS] = (int)SENTTAGS.START;
            this.SrcWordToIndex[ParallelCorpus.UNK] = (int)SENTTAGS.UNK;

            this.m_srcIndexToWord[(int)SENTTAGS.END] = ParallelCorpus.EOS;
            this.m_srcIndexToWord[(int)SENTTAGS.START] = ParallelCorpus.BOS;
            this.m_srcIndexToWord[(int)SENTTAGS.UNK] = ParallelCorpus.UNK;

            this.m_tgtVocab.Add(ParallelCorpus.EOS);
            this.m_tgtVocab.Add(ParallelCorpus.BOS);
            this.m_tgtVocab.Add(ParallelCorpus.UNK);

            this.TgtWordToIndex[ParallelCorpus.EOS] = (int)SENTTAGS.END;
            this.TgtWordToIndex[ParallelCorpus.BOS] = (int)SENTTAGS.START;
            this.TgtWordToIndex[ParallelCorpus.UNK] = (int)SENTTAGS.UNK;

            this.m_tgtIndexToWord[(int)SENTTAGS.END] = ParallelCorpus.EOS;
            this.m_tgtIndexToWord[(int)SENTTAGS.START] = ParallelCorpus.BOS;
            this.m_tgtIndexToWord[(int)SENTTAGS.UNK] = ParallelCorpus.UNK;
        }

        /// <summary>
        /// Load vocabulary from given files
        /// </summary>
        /// <param name="srcVocabFilePath"></param>
        /// <param name="tgtVocabFilePath"></param>
        public Vocab(string srcVocabFilePath, string tgtVocabFilePath)
        {
            Logger.WriteLine("Loading vocabulary files...");
            var srcVocab = File.ReadAllLines(srcVocabFilePath);
            var tgtVocab = File.ReadAllLines(tgtVocabFilePath);

            this.CreateIndex();

            //Build word index for both source and target sides
            var q = 3;
            foreach (var line in srcVocab)
            {
                var items = line.Split('\t');
                var word = items[0];

                if (ParallelCorpus.IsPreDefinedToken(word) == false)
                {
                    this.m_srcVocab.Add(word);
                    this.SrcWordToIndex[word] = q;
                    this.m_srcIndexToWord[q] = word;
                    q++;
                }
            }

            q = 3;
            foreach (var line in tgtVocab)
            {
                var items = line.Split('\t');
                var word = items[0];

                if (ParallelCorpus.IsPreDefinedToken(word) == false)
                {
                    this.m_tgtVocab.Add(word);
                    this.TgtWordToIndex[word] = q;
                    this.m_tgtIndexToWord[q] = word;
                    q++;
                }
            }

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="trainCorpus"></param>
        /// <param name="minFreq"></param>
        public Vocab(IEnumerable<SntPairBatch> trainCorpus, int minFreq = 1)
        {
            Logger.WriteLine($"Building vocabulary from given training corpus.");
            // count up all words
            var s_d = new Dictionary<string, int>();
            var t_d = new Dictionary<string, int>();

            this.CreateIndex();

            foreach (var sntPairBatch in trainCorpus)
            {
                foreach (var sntPair in sntPairBatch.SntPairs)
                {
                    var item = sntPair.SrcSnt;
                    for (int i = 0, n = item.Length; i < n; i++)
                    {
                        var txti = item[i];
                        if (s_d.Keys.Contains(txti)) { s_d[txti] += 1; }
                        else { s_d.Add(txti, 1); }
                    }

                    var item2 = sntPair.TgtSnt;
                    for (int i = 0, n = item2.Length; i < n; i++)
                    {
                        var txti = item2[i];
                        if (t_d.Keys.Contains(txti)) { t_d[txti] += 1; }
                        else { t_d.Add(txti, 1); }
                    }
                }
            }


            var q = 3;
            foreach (var ch in s_d)
            {
                if (ch.Value >= minFreq && ParallelCorpus.IsPreDefinedToken(ch.Key) == false)
                {
                    // add word to vocab
                    this.SrcWordToIndex[ch.Key] = q;
                    this.m_srcIndexToWord[q] = ch.Key;
                    this.m_srcVocab.Add(ch.Key);
                    q++;
                }

            }
            Logger.WriteLine($"Source language Max term id = '{q}'");


            q = 3;
            foreach (var ch in t_d)
            {
                if (ch.Value >= minFreq && ParallelCorpus.IsPreDefinedToken(ch.Key) == false)
                {
                    // add word to vocab
                    this.TgtWordToIndex[ch.Key] = q;
                    this.m_tgtIndexToWord[q] = ch.Key;
                    this.m_tgtVocab.Add(ch.Key);
                    q++;
                }

            }

            Logger.WriteLine($"Target language Max term id = '{q}'");
        }


        public void DumpTargetVocab(string fileName)
        {
            var lines = new List<string>();
            foreach (var pair in this.m_tgtIndexToWord)
            {
                lines.Add($"{pair.Value}\t{pair.Key}");
            }

            File.WriteAllLines(fileName, lines);
        }


        public void DumpSourceVocab(string fileName)
        {
            var lines = new List<string>();
            foreach (var pair in this.m_srcIndexToWord)
            {
                lines.Add($"{pair.Value}\t{pair.Key}");
            }

            File.WriteAllLines(fileName, lines);
        }


        public List<string> ConvertTargetIdsToString(List<int> idxs)
        {
            lock (this.locker)
            {
                var result = new List<string>();
                foreach (var idx in idxs)
                {
                    var letter = ParallelCorpus.UNK;
                    if (this.m_tgtIndexToWord.ContainsKey(idx))
                    {
                        letter = this.m_tgtIndexToWord[idx];
                    }
                    result.Add(letter);
                }

                return result;
            }
        }

        public int GetSourceWordIndex(string word, bool logUnk = false)
        {
            lock (this.locker)
            {
                if (!this.SrcWordToIndex.TryGetValue(word, out var id))
                {
                    id = (int)SENTTAGS.UNK;
                    if (logUnk)
                    {
                        Logger.WriteLine($"Source word '{word}' is UNK");
                    }
                }
                return id;
            }
        }

        public int GetTargetWordIndex(string word, bool logUnk = false)
        {
            lock (this.locker)
            {
                if (!this.TgtWordToIndex.TryGetValue(word, out var id))
                {
                    id = (int)SENTTAGS.UNK;

                    if (logUnk)
                    {
                        Logger.WriteLine($"Target word '{word}' is UNK");
                    }
                }
                return id;
            }
        }
    }
}
