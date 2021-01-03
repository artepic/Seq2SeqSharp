using System;

namespace TensorSharp
{
    public abstract class OpConstraint
    {
        public abstract bool SatisfiedFor(object[] args);
    }

    public class ArgCountConstraint : OpConstraint
    {
        private readonly int argCount;

        public ArgCountConstraint(int argCount) { this.argCount = argCount; }

        public override bool SatisfiedFor(object[] args)
        {
            return args.Length == this.argCount;
        }
    }

    public class ArgTypeConstraint : OpConstraint
    {
        private readonly int argIndex;
        private readonly Type requiredType;

        public ArgTypeConstraint(int argIndex, Type requiredType)
        {
            this.argIndex = argIndex;
            this.requiredType = requiredType;
        }

        public override bool SatisfiedFor(object[] args)
        {
            return this.requiredType.IsAssignableFrom(args[this.argIndex].GetType());
        }
    }

    public class ArgStorageTypeConstraint : OpConstraint
    {
        private readonly int argIndex;
        private readonly Type requiredType;
        private readonly bool allowNull;

        public ArgStorageTypeConstraint(int argIndex, Type requiredType, bool allowNull = true)
        {
            this.argIndex = argIndex;
            this.requiredType = requiredType;
            this.allowNull = allowNull;
        }

        public override bool SatisfiedFor(object[] args)
        {
            if (this.allowNull && args[this.argIndex] == null)
            {
                return true;
            }
            else if (!this.allowNull && args[this.argIndex] == null)
            {
                return false;
            }

            var argStorage = ((Tensor)args[this.argIndex]).Storage;
            return this.requiredType.IsAssignableFrom(argStorage.GetType());
        }
    }
}
