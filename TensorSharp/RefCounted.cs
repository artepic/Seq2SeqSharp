using System;
using System.Threading;

namespace TensorSharp
{
    /// <summary>
    /// Provides a thread safe reference counting implementation. Inheritors need only implement the Destroy() method,
    /// which will be called when the reference count reaches zero. The reference count automatically starts at 1.
    /// </summary>

    [Serializable]
    public abstract class RefCounted
    {
        private int refCount = 1;

        ~RefCounted()
        {
            if (this.refCount > 0)
            {
                this.Destroy();
                this.refCount = 0;
            }
        }

        /// <summary>
        /// This method is called when the reference count reaches zero. It will be called at most once to allow subclasses to release resources.
        /// </summary>
        protected abstract void Destroy();

        /// <summary>
        /// Returns true if the object has already been destroyed; false otherwise.
        /// </summary>
        /// <returns>true if the object is destroyed; false otherwise.</returns>
        protected bool IsDestroyed()
        {
            return this.refCount == 0;
        }

        /// <summary>
        /// Throws an exception if the object has been destroyed, otherwise does nothing.
        /// </summary>
        protected void ThrowIfDestroyed()
        {
            if (this.IsDestroyed())
            {
                throw new InvalidOperationException("Reference counted object has been destroyed");
            }
        }

        protected int GetCurrentRefCount()
        {
            return this.refCount;
        }

        /// <summary>
        /// Increments the reference count. If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void AddRef()
        {
            var spin = new SpinWait();
            while (true)
            {
                var curRefCount = this.refCount;
                if (curRefCount == 0)
                {
                    throw new InvalidOperationException("Cannot AddRef - object has already been destroyed");
                }

                var desiredRefCount = curRefCount + 1;
                var original = Interlocked.CompareExchange(ref this.refCount, desiredRefCount, curRefCount);
                if (original == curRefCount)
                {
                    break;
                }

                spin.SpinOnce();
            }
        }

        /// <summary>
        /// Decrements the reference count. If the reference count reaches zero, the object is destroyed.
        /// If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void Release()
        {
            var spin = new SpinWait();
            while (true)
            {
                var curRefCount = this.refCount;
                if (curRefCount == 0)
                {
                    throw new InvalidOperationException("Cannot release object - object has already been destroyed");
                }

                var desiredRefCount = this.refCount - 1;
                var original = Interlocked.CompareExchange(ref this.refCount, desiredRefCount, curRefCount);
                if (original == curRefCount)
                {
                    break;
                }

                spin.SpinOnce();
            }

            if (this.refCount <= 0)
            {
                this.Destroy();
            }
        }
    }
}
