using System.IO;
using System.Threading.Tasks;

using Seq2SeqSharp.Tools;

namespace Seq2SeqSharp
{
    public class MultiProcessorNetworkWrapper<T> : IMultiProcessorNetworkWrapper where T : INeuralUnit
    {
        private readonly T[] m_networks;
        private readonly int m_defaultDeviceId;
        private readonly int[] m_deviceIds;
        private readonly T m_networkOnDefaultDevice;
        private readonly bool m_isStaticWeights;
        private bool m_weightsSynced;

        public MultiProcessorNetworkWrapper(T networkOnDefaultDevice, int[] deviceIds, bool isStaticWeights = false)
        {
            this.m_networks = new T[deviceIds.Length];
            this.m_defaultDeviceId = networkOnDefaultDevice.GetDeviceId();
            this.m_deviceIds = deviceIds;
            this.m_networkOnDefaultDevice = networkOnDefaultDevice;
            this.m_isStaticWeights = isStaticWeights;
            this.m_weightsSynced = false;

            for (var i = 0; i < deviceIds.Length; i++)
            {
                if (deviceIds[i] == this.m_defaultDeviceId)
                {
                    this.m_networks[i] = networkOnDefaultDevice;
                }
                else
                {
                    this.m_networks[i] = (T)networkOnDefaultDevice.CloneToDeviceAt(deviceIds[i]);
                }
            }
        }

        /// <summary>
        /// Copy weights from tensors on the default device to all other devices
        /// </summary>
        public void SyncWeights()
        {
            if (this.m_isStaticWeights &&
                this.m_weightsSynced)
            {
                return;
            }

            var tensorsOnDefaultDevice = this.m_networkOnDefaultDevice.GetParams();
            Parallel.ForEach(this.m_networks, network =>
            {
                if (network.Equals(this.m_networkOnDefaultDevice) == false)
                {
                    var tensors = network.GetParams();

                    for (var j = 0; j < tensors.Count; j++)
                    {
                        tensors[j].CopyWeightsFrom(tensorsOnDefaultDevice[j]);
                    }
                }

            });

            this.m_weightsSynced = true;
        }

        /// <summary>
        /// Collect gradients from other devices and sum it up to the default device
        /// </summary>
        public void SumGradientsToNetworkOnDefaultDevice()
        {
            if (this.m_isStaticWeights)
            {
                return;
            }

            var tensorsOnDefaultDevice = this.m_networkOnDefaultDevice.GetParams();
            Parallel.ForEach(this.m_networks, network =>
            {
                if (network.Equals(this.m_networkOnDefaultDevice) == false)
                {
                    var tensors = network.GetParams();

                    for (var j = 0; j < tensors.Count; j++)
                    {
                        tensorsOnDefaultDevice[j].AddGradientFrom(tensors[j]);
                    }
                }

            });

        }

        /// <summary>
        /// Fill zero to all gradients on all devices
        /// </summary>
        public void ZeroGradientsOnAllDevices()
        {
            if (this.m_isStaticWeights)
            {
                return;
            }

            Parallel.ForEach(this.m_networks, network =>
            {
                var tensors = network.GetParams();
                foreach (var t in tensors)
                {
                    t.ZeroGradient();
                }
            });
        }

        /// <summary>
        /// Save weights of the network on default device to given stream
        /// </summary>
        /// <param name="stream"></param>
        public void Save(Stream stream)
        {
            if (this.m_isStaticWeights == false)
            {
                this.m_networkOnDefaultDevice.Save(stream);
            }
        }

        /// <summary>
        /// Load weights from given stream to the network on default device
        /// </summary>
        /// <param name="stream"></param>
        public void Load(Stream stream)
        {
            if (this.m_isStaticWeights == false)
            {
                this.m_networkOnDefaultDevice.Load(stream);
            }
        }

        public T GetNetworkOnDefaultDevice()
        {
            return this.m_networkOnDefaultDevice;
        }

        public INeuralUnit GetNeuralUnitOnDefaultDevice()
        {
            return this.GetNetworkOnDefaultDevice();
        }

        /// <summary>
        /// Return the network on specific device
        /// </summary>
        /// <param name="deviceIdIdx">The device id index. -1 is default device</param>
        /// <returns></returns>
        public T GetNetworkOnDevice(int deviceIdIdx)
        {
            return deviceIdIdx == -1 ? this.m_networkOnDefaultDevice : this.m_networks[deviceIdIdx];
        }
    }
}
