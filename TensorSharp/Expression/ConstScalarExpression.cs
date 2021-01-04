namespace TensorSharp.Expression
{
    public class ConstScalarExpression : ScalarExpression
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
}