using System;
using System.Collections.Generic;

namespace TensorSharp.CUDA.Util
{
    public class PooledObject<T> : IDisposable
    {
        private readonly Action<PooledObject<T>> onDispose;
        private readonly T value;

        private bool disposed = false;

        public PooledObject(T value, Action<PooledObject<T>> onDispose)
        {
            this.onDispose = onDispose ?? throw new ArgumentNullException(nameof(onDispose));
            this.value = value;
        }

        public T Value
        {
            get
            {
                if (this.disposed)
                {
                    throw new ObjectDisposedException(this.ToString());
                }

                return this.value;
            }
        }

        public void Dispose()
        {
            if (!this.disposed)
            {
                this.onDispose(this);
                this.disposed = true;
            }
            else
            {
                throw new ObjectDisposedException(this.ToString());
            }
        }
    }

    public class ObjectPool<T> : IDisposable
    {
        private readonly Func<T> constructor;
        private readonly Action<T> destructor;
        private readonly Stack<T> freeList = new();
        private bool disposed = false;


        public ObjectPool(int initialSize, Func<T> constructor, Action<T> destructor)
        {
            this.constructor = constructor ?? throw new ArgumentNullException(nameof(constructor));
            this.destructor = destructor ?? throw new ArgumentNullException(nameof(destructor));

            for (var i = 0; i < initialSize; ++i)
            {
                this.freeList.Push(constructor());
            }
        }

        public void Dispose()
        {
            if (!this.disposed)
            {
                this.disposed = true;
                foreach (var item in this.freeList)
                {
                    this.destructor(item);
                }

                this.freeList.Clear();
            }
        }

        public PooledObject<T> Get()
        {
            var value = this.freeList.Count > 0 ? this.freeList.Pop() : this.constructor();
            return new PooledObject<T>(value, this.Release);
        }

        private void Release(PooledObject<T> handle)
        {
            this.freeList.Push(handle.Value);
        }
    }
}
