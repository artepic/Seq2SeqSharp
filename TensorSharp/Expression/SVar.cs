using System;

namespace TensorSharp.Expression
{
    public class SVar
    {
        private readonly ScalarExpression expression;


        public SVar(ScalarExpression expression)
        {
            this.expression = expression;
        }


        public float Evaluate()
        {
            return this.expression.Evaluate();
        }

        public static implicit operator SVar(float value) { return new(new ConstScalarExpression(value)); }

        public static SVar operator -(SVar src) { return new(new UnaryScalarExpression(src.expression, val => -val)); }

        public static SVar operator +(SVar lhs, SVar rhs) { return new(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l + r)); }
        public static SVar operator -(SVar lhs, SVar rhs) { return new(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l - r)); }
        public static SVar operator *(SVar lhs, SVar rhs) { return new(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l * r)); }
        public static SVar operator /(SVar lhs, SVar rhs) { return new(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l / r)); }
        public static SVar operator %(SVar lhs, SVar rhs) { return new(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l % r)); }


        public SVar Abs() { return new(new UnaryScalarExpression(this.expression, Math.Abs)); }
        public SVar Sign() { return new(new UnaryScalarExpression(this.expression, val => Math.Sign(val))); }

        public SVar Sqrt() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Sqrt(val))); }
        public SVar Exp() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Exp(val))); }
        public SVar Log() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Log(val))); }
        public SVar Floor() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Floor(val))); }
        public SVar Ceil() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Ceiling(val))); }
        public SVar Round() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Round(val))); }
        public SVar Trunc() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Truncate(val))); }


        public SVar Sin() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Sin(val))); }
        public SVar Cos() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Cos(val))); }
        public SVar Tan() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Tan(val))); }

        public SVar Asin() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Asin(val))); }
        public SVar Acos() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Acos(val))); }
        public SVar Atan() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Atan(val))); }

        public SVar Sinh() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Sinh(val))); }
        public SVar Cosh() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Cosh(val))); }
        public SVar Tanh() { return new(new UnaryScalarExpression(this.expression, val => (float)Math.Tanh(val))); }


        public SVar Pow(SVar y) { return new(new BinaryScalarExpression(this.expression, y.expression, (xVal, yVal) => (float)Math.Pow(xVal, yVal))); }
        public SVar Clamp(SVar min, SVar max) { return new(new DelegateScalarExpression(() => ClampFloat(this.expression.Evaluate(), min.expression.Evaluate(), max.expression.Evaluate()))); }

        // public TVar Pow(TVar y) { return new TVar(new BinaryScalarTensorExpression(this.Expression, y.Expression, Ops.Tpow)); }


        public static SVar Atan2(SVar y, SVar x) { return new(new DelegateScalarExpression(() => (float)Math.Atan2(y.Evaluate(), x.Evaluate()))); }
        public static SVar Lerp(SVar a, SVar b, SVar weight) { return new(new DelegateScalarExpression(() => LerpFloat(a.Evaluate(), b.Evaluate(), weight.Evaluate()))); }


        private static float LerpFloat(float a, float b, float weight)
        {
            return a + weight * (b - a);
        }

        private static float ClampFloat(float value, float min, float max)
        {
            if (value < min)
            {
                return min;
            }

            return value > max ? max : value;
        }
    }
}
