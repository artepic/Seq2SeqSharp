using System.Collections.Generic;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    public class KernelConfig
    {
        private readonly SortedDictionary<string, string> values = new SortedDictionary<string, string>();


        public KernelConfig()
        {
        }

        public IEnumerable<string> Keys => this.values.Keys;

        public IEnumerable<KeyValuePair<string, string>> AllValues()
        {
            return this.values;
        }

        public override bool Equals(object obj)
        {
            var o = obj as KernelConfig;
            if (o == null)
            {
                return false;
            }

            if (this.values.Count != o.values.Count)
            {
                return false;
            }

            foreach (var kvp in this.values)
            {
                if (this.values.TryGetValue(kvp.Key, out var oValue))
                {
                    if (!kvp.Value.Equals(oValue))
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

        public override int GetHashCode()
        {
            var result = 0;
            foreach (var kvp in this.values)
            {
                result ^= kvp.Key.GetHashCode();
                result ^= kvp.Value.GetHashCode();
            }
            return result;
        }

        public bool ContainsKey(string name)
        {
            return this.values.ContainsKey(name);
        }

        public void Set(string name, string value)
        {
            this.values[name] = value;
        }

        public string ApplyToTemplate(string templateCode)
        {
            var fullCode = new StringBuilder();
            foreach (var item in this.values)
            {
                fullCode.AppendFormat("#define {0} {1}\n", item.Key, item.Value);
            }
            fullCode.AppendLine(templateCode);
            return fullCode.ToString();
        }
    }
}
