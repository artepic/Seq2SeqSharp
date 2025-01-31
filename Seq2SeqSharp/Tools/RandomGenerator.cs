﻿using System;

namespace Seq2SeqSharp
{
    [Serializable]
    public static class RandomGenerator
    {

        public static bool Return_V { get; set; }
        public static float V_Val { get; set; }

        private static readonly Random random = new(DateTime.Now.Millisecond);
        public static float GaussRandom()
        {
            if (Return_V)
            {
                Return_V = false;
                return V_Val;
            }
            var u = 2 * random.NextDouble() - 1;
            var v = 2 * random.NextDouble() - 1;
            // ReSharper disable ArrangeRedundantParentheses
            var r = (u * u) + (v * v);
            // ReSharper restore ArrangeRedundantParentheses

            if (r == 0 || r > 1)
            {
                return GaussRandom();
            }

            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            V_Val = (float)(v * c);
            Return_V = true;
            return (float)(u * c);
        }

        public static float floatRandom(float a, float b)
        {

            return (float)(random.NextDouble() * (b - a) + a);

        }

        public static float IntegarRandom(float a, float b)
        {
            // ReSharper disable once ArrangeRedundantParentheses
            return (float)(Math.Floor(random.NextDouble() * (b - a) + a));
        }

        ////public static float NormalRandom(float mu, float std)
        ////{
        ////    return mu + GaussRandom() * std;
        ////}

        public static float NormalRandom(float mu, float std)
        {
            return mu + floatRandom(-1.0f, 1.0f) * std;
        }


    }

}
