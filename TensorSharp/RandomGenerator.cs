﻿using System;
using System.Linq;

namespace TensorSharp
{
    public class RandomGenerator
    {
        static readonly Random rnd = new(DateTime.Now.Millisecond);

        public int NextSeed()
        {
            return rnd.Next();
        }

        public static float[] BuildRandomUniformWeight(long[] sizes, float min, float max)
        {
            long size = 1;
            foreach (var s in sizes)
            {
                size *= s;
            }

            var w = new float[size];

            for (var i = 0; i < size; i++)
            {
                w[i] = (float)rnd.NextDouble() * (max - min) + min;
            }

            return w;
        }


        public static float[] BuildRandomBernoulliWeight(long[] sizes, float p)
        {
            var size = sizes.Aggregate<long, long>(1, (current, s) => current * s);

            var w = new float[size];
            
            for (var i = 0; i < size; i++)
            {
                w[i] = rnd.NextDouble() <= p ? 1.0f : 0.0f;
            }

            return w;
        }
    }
}
