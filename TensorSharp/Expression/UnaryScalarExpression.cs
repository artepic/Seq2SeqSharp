using System;
using System.Collections.Generic;
using System.Text;

namespace TensorSharp.Expression
{
    public class UnaryScalarExpression : ScalarExpression
    {
        private readonly ScalarExpression src;
        private readonly Func<float, float> evaluate;


        public UnaryScalarExpression(ScalarExpression src, Func<float, float> evaluate)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        public override float Evaluate()
        {
            return this.evaluate(this.src.Evaluate());
        }
    }
}
