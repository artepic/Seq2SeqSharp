using System;

namespace TensorSharp.Expression
{
    public class BinaryScalarExpression : ScalarExpression
    {
        private readonly Func<float, float, float> evaluate;
        private readonly ScalarExpression left;
        private readonly ScalarExpression right;


        public BinaryScalarExpression(ScalarExpression left, ScalarExpression right, Func<float, float, float> evaluate)
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