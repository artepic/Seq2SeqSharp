using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{

    [Serializable]
    public class KernelDiskCache
    {
        private readonly string cacheDir;
        private readonly Dictionary<string, byte[]> memoryCachedKernels = new();


        public KernelDiskCache(string cacheDir)
        {
            this.cacheDir = cacheDir;
            if (!Directory.Exists(cacheDir))
            {
                Directory.CreateDirectory(cacheDir);
            }
        }

        /// <summary>
        /// Deletes all kernels from disk if they are not currently loaded into memory. Calling this after
        /// calling TSCudaContext.Precompile() will delete any cached .ptx files that are no longer needed
        /// </summary>
        public void CleanUnused()
        {
            foreach (var file in Directory.GetFiles(this.cacheDir))
            {
                var key = this.KeyFromFilePath(file);
                if (!this.memoryCachedKernels.ContainsKey(key))
                {
                    File.Delete(file);
                }
            }
        }

        public byte[] Get(string fullSourceCode, Func<string, byte[]> compile)
        {
            var key = KeyFromSource(fullSourceCode);
            if (this.memoryCachedKernels.TryGetValue(key, out var ptx))
            {
                return ptx;
            }
            else if (this.TryGetFromFile(key, out ptx))
            {
                this.memoryCachedKernels.Add(key, ptx);
                return ptx;
            }
            else
            {
                this.WriteCudaCppToFile(key, fullSourceCode);

                ptx = compile(fullSourceCode);
                this.memoryCachedKernels.Add(key, ptx);
                this.WriteToFile(key, ptx);

                return ptx;
            }
        }


        private void WriteToFile(string key, byte[] ptx)
        {
            var filePath = this.FilePathFromKey(key);

            Logger.WriteLine($"Writing PTX code to '{filePath}'");
            File.WriteAllBytes(filePath, ptx);
        }

        private void WriteCudaCppToFile(string key, string sourceCode)
        {           
            var filePath = this.FilePathFromKey(key) + ".cu";

            Logger.WriteLine($"Writing cuda source code to '{filePath}'");
            File.WriteAllText(filePath, sourceCode);
        }

        private bool TryGetFromFile(string key, out byte[] ptx)
        {
            var filePath = this.FilePathFromKey(key);
            if (!File.Exists(filePath))
            {
                ptx = null;
                return false;
            }

            ptx = File.ReadAllBytes(filePath);
            return true;
        }

        private string FilePathFromKey(string key)
        {
            return Path.Combine(this.cacheDir, key + ".ptx");
        }

        private string KeyFromFilePath(string filepath)
        {
            var fileExts = new string[] { ".ptx", ".cu" };

            foreach (var ext in fileExts)
            {
                if (filepath.EndsWith(ext, StringComparison.InvariantCultureIgnoreCase))
                {
                    filepath = filepath.Substring(0, filepath.Length - ext.Length);
                }
            }

     //       return filepath;
           

            return Path.GetFileNameWithoutExtension(filepath);
        }

        private static string KeyFromSource(string fullSource)
        {
            var fullKey = fullSource.Length.ToString() + fullSource;

            using var sha1 = new SHA1Managed();
            return BitConverter.ToString(sha1.ComputeHash(Encoding.UTF8.GetBytes(fullKey)))
                               .Replace("-", "");
        }
    }
}
