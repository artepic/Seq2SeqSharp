﻿using AdvUtils;

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public class SequenceLabelingCorpus : IEnumerable<SntPairBatch>
    {
        private readonly int m_maxSentLength = 32;

        private readonly int m_blockSize = 1000000;
        private readonly int m_batchSize = 1;
        private readonly bool m_addBOSEOS = true;
        private readonly List<string> m_srcFileList;
        private readonly List<string> m_tgtFileList;

        public int CorpusSize;

        public int BatchSize => this.m_batchSize;

        public const string EOS = "<END>";
        public const string BOS = "<START>";
        public const string UNK = "<UNK>";

        private bool m_showTokenDist = true;

        public static bool IsPreDefinedToken(string str)
        {
            return str == EOS || str == BOS || str == UNK;
        }

        private readonly Random rnd = new(DateTime.Now.Millisecond);

        private void Shuffle(List<RawSntPair> rawSntPairs)
        {

            //Put sentence pair with same source length into the bucket
            var dict = new Dictionary<int, List<RawSntPair>>(); //<source sentence length, sentence pair set>
            foreach (var item in rawSntPairs)
            {
                var length = item.SrcLength;

                if (dict.ContainsKey(length) == false)
                {
                    dict.Add(length, new List<RawSntPair>());
                }
                dict[length].Add(item);
            }

            //Randomized the order of sentence pairs with same length in source side
            Parallel.ForEach(dict, pair =>
            //foreach (KeyValuePair<int, List<SntPair>> pair in dict)
            {
                var rnd2 = new Random(DateTime.Now.Millisecond + pair.Key);

                var sntPairList = pair.Value;
                for (var i = 0; i < sntPairList.Count; i++)
                {
                    var idx = rnd2.Next(0, sntPairList.Count);
                    var tmp = sntPairList[i];
                    sntPairList[i] = sntPairList[idx];
                    sntPairList[idx] = tmp;
                }
            });

            var sdict = new SortedDictionary<int, List<RawSntPair>>(); //<The bucket size, sentence pair set>
            foreach (var pair in dict)
            {
                if (sdict.ContainsKey(pair.Value.Count) == false)
                {
                    sdict.Add(pair.Value.Count, new List<RawSntPair>());
                }
                sdict[pair.Value.Count].AddRange(pair.Value);
            }

            rawSntPairs.Clear();

            var keys = sdict.Keys.ToArray();
            for (var i = 0; i < keys.Length; i++)
            {
                var idx = this.rnd.Next(0, keys.Length);
                var tmp = keys[i];
                keys[i] = keys[idx];
                keys[idx] = tmp;
            }

            foreach (var key in keys)
            {
                rawSntPairs.AddRange(sdict[key]);
            }

        }

        private (string, string) ShuffleAll()
        {
            var dictSrcLenDist = new SortedDictionary<int, int>();
            var dictTgtLenDist = new SortedDictionary<int, int>();

            var srcShuffledFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + ".tmp");
            var tgtShuffledFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + ".tmp");

            Logger.WriteLine($"Shuffling corpus for '{this.m_srcFileList.Count}' files.");

            var swSrc = new StreamWriter(srcShuffledFilePath, false);
            var swTgt = new StreamWriter(tgtShuffledFilePath, false);

            var sntPairs = new List<RawSntPair>();
            this.CorpusSize = 0;
            var tooLongSrcSntCnt = 0;

            for (var i = 0; i < this.m_srcFileList.Count; i++)
            {
                if (this.m_showTokenDist)
                {
                    Logger.WriteLine($"Process file '{this.m_srcFileList[i]}' and '{this.m_tgtFileList[i]}'");
                }

                var srSrc = new StreamReader(this.m_srcFileList[i]);
                var srTgt = new StreamReader(this.m_tgtFileList[i]);

                while (true)
                {
                    if (srSrc.EndOfStream && srTgt.EndOfStream)
                    {
                        break;
                    }

                    var rawSntPair = new RawSntPair(srSrc.ReadLine(), srTgt.ReadLine());
                    if (rawSntPair.IsEmptyPair())
                    {
                        break;
                    }

                    if (dictSrcLenDist.ContainsKey(rawSntPair.SrcLength / 100) == false)
                    {
                        dictSrcLenDist.Add(rawSntPair.SrcLength / 100, 0);
                    }
                    dictSrcLenDist[rawSntPair.SrcLength / 100]++;

                    if (dictTgtLenDist.ContainsKey(rawSntPair.TgtLength / 100) == false)
                    {
                        dictTgtLenDist.Add(rawSntPair.TgtLength / 100, 0);
                    }
                    dictTgtLenDist[rawSntPair.TgtLength / 100]++;


                    var hasTooLongSent = false;
                    if (rawSntPair.SrcLength > this.m_maxSentLength)
                    {
                        tooLongSrcSntCnt++;
                        hasTooLongSent = true;
                    }

                    if (hasTooLongSent)
                    {
                        continue;
                    }

                    sntPairs.Add(rawSntPair);
                    this.CorpusSize++;
                    if (this.m_blockSize > 0 && sntPairs.Count >= this.m_blockSize)
                    {
                        this.Shuffle(sntPairs);
                        foreach (var item in sntPairs)
                        {
                            swSrc.WriteLine(item.SrcSnt);
                            swTgt.WriteLine(item.TgtSnt);
                        }
                        sntPairs.Clear();
                    }
                }

                srSrc.Close();
                srTgt.Close();
            }

            if (sntPairs.Count > 0)
            {
                this.Shuffle(sntPairs);
                foreach (var item in sntPairs)
                {
                    swSrc.WriteLine(item.SrcSnt);
                    swTgt.WriteLine(item.TgtSnt);
                }

                sntPairs.Clear();
            }


            swSrc.Close();
            swTgt.Close();

            Logger.WriteLine($"Shuffled '{this.CorpusSize}' sentence pairs to file '{srcShuffledFilePath}' and '{tgtShuffledFilePath}'.");

            if (tooLongSrcSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongSrcSntCnt} source sentences are longer than '{this.m_maxSentLength}' tokens, ignore them.");
            }

            if (this.m_showTokenDist)
            {
                Logger.WriteLine($"Src token length distribution");

                var srcTotalNum = 0;
                foreach (var pair in dictSrcLenDist)
                {
                    srcTotalNum += pair.Value;
                }

                var srcAccNum = 0;
                foreach (var pair in dictSrcLenDist)
                {
                    srcAccNum += pair.Value;

                    Logger.WriteLine($"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value} (acc: {(100.0f * (float)srcAccNum / (float)srcTotalNum).ToString("F")}%)");
                }

                Logger.WriteLine($"Tgt token length distribution");

                var tgtTotalNum = 0;
                foreach (var pair in dictTgtLenDist)
                {
                    tgtTotalNum += pair.Value;
                }

                var tgtAccNum = 0;

                foreach (var pair in dictTgtLenDist)
                {
                    tgtAccNum += pair.Value;

                    Logger.WriteLine($"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value}  (acc: {(100.0f * (float)tgtAccNum / (float)tgtTotalNum).ToString("F")}%)");
                }

                this.m_showTokenDist = false;
            }


            return (srcShuffledFilePath, tgtShuffledFilePath);
        }

        public IEnumerator<SntPairBatch> GetEnumerator()
        {
            var (srcShuffledFilePath, tgtShuffledFilePath) = this.ShuffleAll();

            using (var srSrc = new StreamReader(srcShuffledFilePath))
            {
                using (var srTgt = new StreamReader(tgtShuffledFilePath))
                {
                    var lastSrcSntLen = -1;
                    var lastTgtSntLen = -1;
                    var maxOutputsSize = this.m_batchSize * 10000;
                    var outputs = new List<SntPair>();

                    while (true)
                    {
                        string line;
                        var sntPair = new SntPair();
                        if ((line = srSrc.ReadLine()) == null)
                        {
                            break;
                        }

                        line = line.ToLower().Trim();
                        if (this.m_addBOSEOS)
                        {
                            line = $"{BOS} {line} {EOS}";
                        }
                        sntPair.SrcSnt = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

                        line = srTgt.ReadLine().ToLower().Trim();
                        if (this.m_addBOSEOS)
                        {
                            line = $"{BOS} {line} {EOS}";
                        }
                        sntPair.TgtSnt = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);


                        if (
                            // ReSharper disable ArrangeRedundantParentheses
                            (lastSrcSntLen > 0 && lastSrcSntLen != sntPair.SrcSnt.Length) ||
                            // ReSharper restore ArrangeRedundantParentheses
                            outputs.Count > maxOutputsSize)
                        {
                            // InnerShuffle(outputs);
                            for (var i = 0; i < outputs.Count; i += this.m_batchSize)
                            {
                                var size = Math.Min(this.m_batchSize, outputs.Count - i);
                                yield return new SntPairBatch(outputs.GetRange(i, size));
                            }

                            outputs.Clear();
                        }

                        outputs.Add(sntPair);

                        lastSrcSntLen = sntPair.SrcSnt.Length;
                        lastTgtSntLen = sntPair.TgtSnt.Length;
                    }

                    // InnerShuffle(outputs);
                    for (var i = 0; i < outputs.Count; i += this.m_batchSize)
                    {
                        var size = Math.Min(this.m_batchSize, outputs.Count - i);
                        yield return new SntPairBatch(outputs.GetRange(i, size));
                    }
                }
            }

            File.Delete(srcShuffledFilePath);
            File.Delete(tgtShuffledFilePath);
        }

        /// <summary>
        /// Pad given sentences to the same length and return their original length
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static List<int> PadSentences(List<List<string>> s, int maxLen = -1)
        {
            var originalLengths = new List<int>();

            if (maxLen <= 0)
            {
                foreach (var item in s)
                {
                    if (item.Count > maxLen)
                    {
                        maxLen = item.Count;
                    }
                }
            }

            for (var i = 0; i < s.Count; i++)
            {
                var count = s[i].Count;
                originalLengths.Add(count);

                for (var j = 0; j < maxLen - count; j++)
                {
                    s[i].Add(ParallelCorpus.EOS);
                }
            }

            return originalLengths;
        }


        public static List<List<string>> LeftShiftSnts(List<List<string>> input, string lastTokenToPad)
        {
            var r = new List<List<string>>();

            foreach (var seq in input)
            {
                var rseq = new List<string>();

                rseq.AddRange(seq);
                rseq.RemoveAt(0);
                rseq.Add(lastTokenToPad);

                r.Add(rseq);
            }

            return r;
        }


        /// <summary>
        /// Shuffle given sentence pairs and return the length of the longgest source sentence
        /// </summary>
        /// <param name="sntPairs"></param>
        /// <returns></returns>
        //private int InnerShuffle(List<SntPair> sntPairs)
        //{
        //    int maxSrcLen = 0;
        //    for (int i = 0; i < sntPairs.Count; i++)
        //    {
        //        if (sntPairs[i].SrcSnt.Length > maxSrcLen)
        //        {
        //            maxSrcLen = sntPairs[i].SrcSnt.Length;
        //        }

        //        int idx = rnd.Next(0, sntPairs.Count);
        //        SntPair tmp = sntPairs[i];
        //        sntPairs[i] = sntPairs[idx];
        //        sntPairs[idx] = tmp;
        //    }

        //    return maxSrcLen;
        //}

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        private (string, string) ConvertSequenceLabelingFormatToParallel(string filePath)
        {
            var srcLines = new List<string>();
            var tgtLines = new List<string>();

            var currSrcLine = new List<string>();
            var currTgtLine = new List<string>();
            foreach (var line in File.ReadAllLines(filePath))
            {
                if (string.IsNullOrEmpty(line) == true)
                {
                    //This is a new record

                    srcLines.Add(string.Join(" ", currSrcLine));
                    tgtLines.Add(string.Join(" ", currTgtLine));

                    currSrcLine = new List<string>();
                    currTgtLine = new List<string>();
                }
                else
                {
                    var items = line.Split(new char[] { ' ', '\t' });
                    var srcItem = items[0];
                    var tgtItem = items[1];

                    currSrcLine.Add(srcItem);
                    currTgtLine.Add(tgtItem);
                }
            }

            srcLines.Add(string.Join(" ", currSrcLine));
            tgtLines.Add(string.Join(" ", currTgtLine));

            var srcFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_src.tmp");
            var tgtFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_tgt.tmp");

            File.WriteAllLines(srcFilePath, srcLines);
            File.WriteAllLines(tgtFilePath, tgtLines);

            Logger.WriteLine($"Convert sequence labeling corpus file '{filePath}' to parallel corpus files '{srcFilePath}' and '{tgtFilePath}'");

            return (srcFilePath, tgtFilePath);
        }


        public SequenceLabelingCorpus(string corpusFilePath, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 128)
        {
            Logger.WriteLine($"Loading sequence labeling corpus from '{corpusFilePath}' MaxSrcSentLength = '{maxSentLength}'");
            this.m_batchSize = batchSize;
            this.m_blockSize = shuffleBlockSize;
            this.m_maxSentLength = maxSentLength;

            this.m_srcFileList = new List<string>();
            this.m_tgtFileList = new List<string>();


            var (srcFilePath, tgtFilePath) = this.ConvertSequenceLabelingFormatToParallel(corpusFilePath);

            this.m_srcFileList.Add(srcFilePath);
            this.m_tgtFileList.Add(tgtFilePath);
        }
    }
}