﻿using System;
using System.Reflection;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public class PrecompileAttribute : Attribute
    {
    }

    public interface IPrecompilable
    {
        void Precompile(CudaCompiler compiler);
    }

    public static class PrecompileHelper
    {
        public static void PrecompileAllFields(object instance, CudaCompiler compiler)
        {
            var type = instance.GetType();

            foreach (var field in type.GetFields())
            {
                if (typeof(IPrecompilable).IsAssignableFrom(field.FieldType))
                {
                    var precompilableField = (IPrecompilable)field.GetValue(instance);
                    Console.WriteLine("Compiling field " + field.Name);
                    precompilableField.Precompile(compiler);
                }
            }
        }
    }
}
