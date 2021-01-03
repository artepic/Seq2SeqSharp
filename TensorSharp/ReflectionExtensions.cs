using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace TensorSharp
{
    public static class AssemblyExtensions
    {
        public static IEnumerable<Tuple<Type, IEnumerable<T>>> TypesWithAttribute<T>(this Assembly assembly, bool inherit)
        {
            return from type in assembly.GetTypes() let attributes = type.GetCustomAttributes(typeof(T), inherit) where attributes.Any() select Tuple.Create(type, attributes.Cast<T>());
        }
    }

    public static class TypeExtensions
    {
        public static IEnumerable<Tuple<MethodInfo, IEnumerable<T>>> MethodsWithAttribute<T>(this Type type, bool inherit)
        {
            return from method in type.GetMethods() let attributes = method.GetCustomAttributes(typeof(T), inherit) where attributes.Any() select Tuple.Create(method, attributes.Cast<T>());
        }
    }

    public static class MethodExtensions
    {
        public static IEnumerable<Tuple<ParameterInfo, IEnumerable<T>>> ParametersWithAttribute<T>(this MethodInfo method, bool inherit)
        {
            return from parameter in method.GetParameters() let attributes = parameter.GetCustomAttributes(typeof(T), inherit) where attributes.Any() select Tuple.Create(parameter, attributes.Cast<T>());
        }
    }
}
