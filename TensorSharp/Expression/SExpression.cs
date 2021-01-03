﻿using System;

namespace TensorSharp.Expression
{
    public abstract class SExpression
    {
        public abstract float Evaluate();
    }


    public class ConstScalarExpression : SExpression
    {
        private readonly float value;

        public ConstScalarExpression(float value)
        {
            this.value = value;
        }

        public override float Evaluate()
        {
            return this.value;
        }
    }

    public class DelegateScalarExpression : SExpression
    {
        private readonly Func<float> evaluate;

        public DelegateScalarExpression(Func<float> evaluate)
        {
            this.evaluate = evaluate;
        }

        public override float Evaluate()
        {
            return this.evaluate();
        }
    }

    public class UnaryScalarExpression : SExpression
    {
        private readonly SExpression src;
        private readonly Func<float, float> evaluate;


        public UnaryScalarExpression(SExpression src, Func<float, float> evaluate)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        public override float Evaluate()
        {
            return this.evaluate(this.src.Evaluate());
        }
    }

    public class BinaryScalarExpression : SExpression
    {
        private readonly SExpression left;
        private readonly SExpression right;
        private readonly Func<float, float, float> evaluate;


        public BinaryScalarExpression(SExpression left, SExpression right, Func<float, float, float> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        public override float Evaluate()
        {
            return this.evaluate(this.left.Evaluate(), this.right.Evaluate());
        }
    }
}
