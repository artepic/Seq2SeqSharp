using System;

namespace TensorSharp.Expression
{
    public class DelegateScalarExpression : ScalarExpression
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
}