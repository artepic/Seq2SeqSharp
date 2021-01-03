using AdvUtils;
using Seq2SeqSharp.Metrics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    /// <summary>
    /// This is a framework for neural network training. It includes many core parts, such as backward propagation, parameters updates, 
    /// memory management, computing graph managment, corpus shuffle & batching, I/O for model, logging & monitoring, checkpoints.
    /// You need to create your network inherited from this class, implmenet forward part only and pass it to TrainOneEpoch method for training
    /// </summary>
    public abstract class BaseSeq2SeqFramework
    {
        public event EventHandler IterationDone;

        private readonly int[] m_deviceIds;
        public int[] DeviceIds => this.m_deviceIds;

        private readonly string m_modelFilePath;
        private readonly float m_regc = 1e-10f; // L2 regularization strength
        private int m_weightsUpdateCount = 0;
        private double m_avgCostPerWordInTotalInLastEpoch = 10000.0;
        private double m_bestPrimaryScore = 0.0f;
        private readonly object locker = new object();
        private SortedList<string, IMultiProcessorNetworkWrapper> m_name2network;
        DateTime m_lastCheckPointDateTime = DateTime.Now;

        public BaseSeq2SeqFramework(int[] deviceIds, ProcessorTypeEnums processorType, string modelFilePath, float memoryUsageRatio = 0.9f, string[] compilerOptions = null)
        {
            this.m_deviceIds = deviceIds;
            this.m_modelFilePath = modelFilePath;
            TensorAllocator.InitDevices(processorType, this.m_deviceIds, memoryUsageRatio, compilerOptions);
        }

        public IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true)
        {
            if (deviceIdIdx < 0 || deviceIdIdx >= this.DeviceIds.Length)
            {
                throw new ArgumentOutOfRangeException($"Index '{deviceIdIdx}' is out of deviceId range. DeviceId length is '{this.DeviceIds.Length}'");
            }

            // Create computing graph instance and return it
            return new ComputeGraphTensor(new WeightTensorFactory(), this.DeviceIds[deviceIdIdx], needBack);
        }

        public bool SaveModel(IModelMetaData modelMetaData)
        {
            try
            {
                Logger.WriteLine($"Saving model to '{this.m_modelFilePath}'");

                if (File.Exists(this.m_modelFilePath))
                {
                    File.Copy(this.m_modelFilePath, $"{this.m_modelFilePath}.bak", true);
                }

                var bf = new BinaryFormatter();
                using (var fs = new FileStream(this.m_modelFilePath, FileMode.Create, FileAccess.Write))
                {
                    // Save model meta data to the stream
                    bf.Serialize(fs, modelMetaData);
                    // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be saved to given stream
                    this.SaveParameters(fs);
                }

                return true;
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Failed to save model to file. Exception = '{err.Message}'");
                return false;
            }
        }

        /// <summary>
        /// Load model from given file
        /// </summary>
        /// <param name="InitializeParameters"></param>
        /// <returns></returns>
        public IModelMetaData LoadModel(Func<IModelMetaData, bool> InitializeParameters)
        {
            Logger.WriteLine($"Loading model from '{this.m_modelFilePath}'...");
            IModelMetaData modelMetaData = null;
            var bf = new BinaryFormatter();
            using (var fs = new FileStream(this.m_modelFilePath, FileMode.Open, FileAccess.Read))
            {
                modelMetaData = bf.Deserialize(fs) as IModelMetaData;

                //Initialize parameters on devices
                InitializeParameters(modelMetaData);

                // Load embedding and weights from given model
                // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be loaded from given stream
                this.LoadParameters(fs);
            }

            return modelMetaData;
        }

        internal void TrainOneEpoch(int ep, IEnumerable<SntPairBatch> trainCorpus, IEnumerable<SntPairBatch> validCorpus, ILearningRate learningRate, AdamOptimizer solver, List<IMetric> metrics, IModelMetaData modelMetaData,
            Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice)
        {
            var processedLineInTotal = 0;
            var startDateTime = DateTime.Now;
            var costInTotal = 0.0;
            long srcWordCntsInTotal = 0;
            long tgtWordCntsInTotal = 0;
            var avgCostPerWordInTotal = 0.0;

            Logger.WriteLine($"Start to process training corpus.");
            var sntPairBatchs = new List<SntPairBatch>();

            foreach (var sntPairBatch in trainCorpus)
            {
                sntPairBatchs.Add(sntPairBatch);
                if (sntPairBatchs.Count == this.m_deviceIds.Length)
                {
                    // Copy weights from weights kept in default device to all other devices
                    this.CopyWeightsFromDefaultDeviceToAllOtherDevices();

                    var batchSplitFactor = 1;
                    var runNetwordSuccssed = false;

                    while (runNetwordSuccssed == false)
                    {
                        try
                        {
                            (var cost, var sWordCnt, var tWordCnt, var processedLine) = this.RunNetwork(ForwardOnSingleDevice, sntPairBatchs, batchSplitFactor);
                            processedLineInTotal += processedLine;
                            srcWordCntsInTotal += sWordCnt;
                            tgtWordCntsInTotal += tWordCnt;

                            //Sum up gradients in all devices, and kept it in default device for parameters optmization
                            this.SumGradientsToTensorsInDefaultDevice();

                            //Optmize parameters
                            var lr = learningRate.GetCurrentLearningRate();
                            var models = this.GetParametersFromDefaultDevice();
                            solver.UpdateWeights(models, processedLine, lr, this.m_regc, this.m_weightsUpdateCount + 1);


                            costInTotal += cost;
                            avgCostPerWordInTotal = costInTotal / tgtWordCntsInTotal;
                            this.m_weightsUpdateCount++;
                            if (this.IterationDone != null &&
                                this.m_weightsUpdateCount % 100 == 0)
                            {
                                this.IterationDone(this, new CostEventArg()
                                {
                                    LearningRate = lr,
                                    CostPerWord = cost / tWordCnt,
                                    AvgCostInTotal = avgCostPerWordInTotal,
                                    Epoch = ep,
                                    Update = this.m_weightsUpdateCount,
                                    ProcessedSentencesInTotal = processedLineInTotal,
                                    ProcessedWordsInTotal = srcWordCntsInTotal + tgtWordCntsInTotal,
                                    StartDateTime = startDateTime
                                });
                            }

                            runNetwordSuccssed = true;
                        }
                        catch (AggregateException err)
                        {
                            if (err.InnerExceptions != null)
                            {
                                var oomMessage = String.Empty;
                                var isOutOfMemException = false;
                                var isArithmeticException = false;
                                foreach (var excep in err.InnerExceptions)
                                {
                                    if (excep is OutOfMemoryException)
                                    {
                                        isOutOfMemException = true;
                                        oomMessage = excep.Message;
                                        break;
                                    }
                                    else if (excep is ArithmeticException)
                                    {
                                        isArithmeticException = true;
                                        oomMessage = excep.Message;
                                        break;
                                    }
                                }

                                if (isOutOfMemException)
                                {
                                    batchSplitFactor *= 2;
                                    Logger.WriteLine($"Got an exception ('{oomMessage}'), so we increase batch split factor to {batchSplitFactor}, and retry it.");

                                    if (batchSplitFactor >= sntPairBatchs[0].BatchSize)
                                    {
                                        Logger.WriteLine($"Batch split factor is larger than batch size, so ignore current mini-batch.");
                                        break;
                                    }
                                }
                                else if (isArithmeticException)
                                {
                                    Logger.WriteLine($"Arithmetic exception: '{err.Message}'");
                                    break;
                                }
                                else
                                {
                                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}, Call stack: {err.StackTrace}");
                                    throw err;
                                }
                            }
                            else
                            {
                                Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}, Call stack: {err.StackTrace}");
                                throw err;
                            }

                        }
                        catch (OutOfMemoryException err)
                        {
                            batchSplitFactor *= 2;
                            Logger.WriteLine($"Got an exception ('{err.Message}'), so we increase batch split factor to {batchSplitFactor}, and retry it.");

                            if (batchSplitFactor >= sntPairBatchs[0].BatchSize)
                            {
                                Logger.WriteLine($"Batch split factor is larger than batch size, so ignore current mini-batch.");
                                break;
                            }
                        }
                        catch (ArithmeticException err)
                        {
                            Logger.WriteLine($"Arithmetic exception: '{err.Message}'");
                            break;
                        }
                        catch (Exception err)
                        {
                            Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}, Call stack: {err.StackTrace}");
                            throw err;
                        }
                    }

                    // Evaluate model every hour and save it if we could get a better one.
                    var ts = DateTime.Now - this.m_lastCheckPointDateTime;
                    if (ts.TotalMinutes > 1.0)
                    {
                        this.CreateCheckPoint(validCorpus, metrics, modelMetaData, ForwardOnSingleDevice, avgCostPerWordInTotal);
                        this.m_lastCheckPointDateTime = DateTime.Now;
                    }

                    sntPairBatchs.Clear();
                }
            }

            Logger.WriteLine(Logger.Level.info, ConsoleColor.Green, $"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal.ToString("F6")}, AvgCostInLastEpoch = {this.m_avgCostPerWordInTotalInLastEpoch.ToString("F6")}");

            //  CreateCheckPoint(validCorpus, metrics, modelMetaData, ForwardOnSingleDevice, avgCostPerWordInTotal);
            this.m_avgCostPerWordInTotalInLastEpoch = avgCostPerWordInTotal;
        }

        private (float, int, int, int) RunNetwork(Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice, List<SntPairBatch> sntPairBatchs, int batchSplitFactor)
        {
            var cost = 0.0f;
            var processedLine = 0;
            var srcWordCnts = 0;
            var tgtWordCnts = 0;

            //Clear gradient over all devices
            this.ZeroGradientOnAllDevices();

            // Run forward and backward on all available processors
            Parallel.For(0, this.m_deviceIds.Length, i =>
            {
                try
                {
                    var sntPairBatch_i = sntPairBatchs[i];
                    var batchSegSize = sntPairBatch_i.BatchSize / batchSplitFactor;

                    for (var k = 0; k < batchSplitFactor; k++)
                    {
                        // Construct sentences for encoding and decoding
                        var srcTkns = new List<List<string>>();
                        var tgtTkns = new List<List<string>>();
                        var sLenInBatch = 0;
                        var tLenInBatch = 0;
                        for (var j = k * batchSegSize; j < (k + 1) * batchSegSize; j++)
                        {
                            srcTkns.Add(sntPairBatch_i.SntPairs[j].SrcSnt.ToList());
                            sLenInBatch += sntPairBatch_i.SntPairs[j].SrcSnt.Length;

                            tgtTkns.Add(sntPairBatch_i.SntPairs[j].TgtSnt.ToList());
                            tLenInBatch += sntPairBatch_i.SntPairs[j].TgtSnt.Length;
                        }

                        var lcost = 0.0f;
                        // Create a new computing graph instance
                        using (var computeGraph_i = this.CreateComputGraph(i))
                        {
                            // Run forward part
                            lcost = ForwardOnSingleDevice(computeGraph_i, srcTkns, tgtTkns, i, true);
                            // Run backward part and compute gradients
                            computeGraph_i.Backward();
                        }

                        lock (this.locker)
                        {
                            cost += lcost;
                            srcWordCnts += sLenInBatch;
                            tgtWordCnts += tLenInBatch;
                            processedLine += batchSegSize;
                        }
                    }
                }
                catch (OutOfMemoryException err)
                {                    
                    throw err;
                }
                catch (Exception err)
                {
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: '{err.Message}'");
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Call stack: '{err.StackTrace}'");

                    throw err;
                }
            });

            return (cost, srcWordCnts, tgtWordCnts, processedLine);
        }

        private void CreateCheckPoint(IEnumerable<SntPairBatch> validCorpus, List<IMetric> metrics, IModelMetaData modelMetaData, Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice, double avgCostPerWordInTotal)
        {
            if (validCorpus != null)
            {
                // The valid corpus is provided, so evaluate the model.
                if (this.RunValid(validCorpus, ForwardOnSingleDevice, metrics) == true)
                {
                    this.SaveModel(modelMetaData);
                }
            }
            else if (this.m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
            {
                // We don't have valid corpus, so if we could have lower cost, save the model
                this.SaveModel(modelMetaData);
            }
        }

        internal List<List<string>> RunTest(List<List<string>> inputTokens, Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice)
        {
            var hypTkns = new List<List<string>>();
            hypTkns.Add(new List<string>());
            hypTkns[0].Add(ParallelCorpus.BOS);

            try
            {
                // Create a new computing graph instance
                using (var computeGraph = this.CreateComputGraph(this.DeviceIds[0], needBack: false))
                {
                    // Run forward part
                    ForwardOnSingleDevice(computeGraph, inputTokens, hypTkns, this.DeviceIds[0], false);
                }
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}', Call Stack = '{err.StackTrace}'");
            }
       
            return hypTkns;
        }

        /// <summary>
        /// Evaluate the quality of model on valid corpus.
        /// </summary>
        /// <param name="validCorpus">valid corpus to measure the quality of model</param>
        /// <param name="RunNetwork">The network to run on specific device</param>
        /// <param name="metrics">A set of metrics. The first one is the primary metric</param>
        /// <param name="outputToFile">It indicates if valid corpus and results should be dumped to files</param>
        /// <returns>true if we get a better result on primary metric, otherwise, false</returns>
        internal bool RunValid(IEnumerable<SntPairBatch> validCorpus, Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> RunNetwork, List<IMetric> metrics, bool outputToFile = false)
        {
            var srcSents = new List<string>();
            var refSents = new List<string>();
            var hypSents = new List<string>();


            // Clear inner status of each metrics
            foreach (var metric in metrics)
            {
                metric.ClearStatus();
            }

            var sntPairBatchs = new List<SntPairBatch>();
            foreach (var item in validCorpus)
            {
                sntPairBatchs.Add(item);
                if (sntPairBatchs.Count == this.DeviceIds.Length)
                {

                    // Run forward on all available processors
                    Parallel.For(0, this.m_deviceIds.Length, i =>
                    {
                        var sntPairBatch = sntPairBatchs[i];

                        // Construct sentences for encoding and decoding
                        var srcTkns = new List<List<string>>();
                        var refTkns = new List<List<string>>();
                        var hypTkns = new List<List<string>>();
                        for (var j = 0; j < sntPairBatch.BatchSize; j++)
                        {
                            srcTkns.Add(sntPairBatch.SntPairs[j].SrcSnt.ToList());
                            refTkns.Add(sntPairBatch.SntPairs[j].TgtSnt.ToList());
                            hypTkns.Add(new List<string>() { ParallelCorpus.BOS });
                        }

                        // Create a new computing graph instance
                        using (var computeGraph = this.CreateComputGraph(i, needBack: false))
                        {
                            // Run forward part
                            RunNetwork(computeGraph, srcTkns, hypTkns, i, false);
                        }

                        lock (this.locker)
                        {

                            for (var j = 0; j < hypTkns.Count; j++)
                            {
                                foreach (var metric in metrics)
                                {
                                    if (j < 0 || j >= refTkns.Count)
                                    {
                                        throw new InvalidDataException($"Ref token only has '{refTkns.Count}' batch, however, it try to access batch '{j}'. Hyp token has '{hypTkns.Count}' tokens, Batch Size = '{sntPairBatch.BatchSize}'");
                                    }

                                    if (j < 0 || j >= hypTkns.Count)
                                    {
                                        throw new InvalidDataException($"Hyp token only has '{hypTkns.Count}' batch, however, it try to access batch '{j}'. Ref token has '{refTkns.Count}' tokens, Batch Size = '{sntPairBatch.BatchSize}'");
                                    }

                                    metric.Evaluate(new List<List<string>>() { refTkns[j] }, hypTkns[j]);
                                }
                            }

                            if (outputToFile)
                            {
                                for (var j = 0; j < srcTkns.Count; j++)
                                {
                                    srcSents.Add(string.Join(" ", srcTkns[j]));
                                    refSents.Add(string.Join(" ", refTkns[j]));
                                    hypSents.Add(string.Join(" ", hypTkns[j]));
                                }
                            }
                        }


                    });

                    sntPairBatchs.Clear();
                }
            }



            Logger.WriteLine($"Metrics result:");
            foreach (var metric in metrics)
            {
                Logger.WriteLine(Logger.Level.info, ConsoleColor.DarkGreen, $"{metric.Name} = {metric.GetScoreStr()}");
            }

            if (outputToFile)
            {
                File.WriteAllLines("valid_src.txt", srcSents);
                File.WriteAllLines("valid_ref.txt", refSents);
                File.WriteAllLines("valid_hyp.txt", hypSents);
            }

            if (metrics.Count > 0)
            {
                if (metrics[0].GetPrimaryScore() > this.m_bestPrimaryScore)
                {
                    Logger.WriteLine(Logger.Level.info, ConsoleColor.Green, $"We got a better score '{metrics[0].GetPrimaryScore().ToString("F")}' on primary metric '{metrics[0].Name}'. The previous score is '{this.m_bestPrimaryScore.ToString("F")}'");
                    //We have a better primary score on valid set
                    this.m_bestPrimaryScore = metrics[0].GetPrimaryScore();
                    return true;
                }
            }

            return false;
        }

        internal virtual void SaveParameters(Stream stream)
        {
            this.RegisterTrainableParameters(this);
            foreach (var pair in this.m_name2network)
            {
                pair.Value.Save(stream);
            }
        }

        internal virtual void LoadParameters(Stream stream)
        {
            this.RegisterTrainableParameters(this);
            foreach (var pair in this.m_name2network)
            {
                Logger.WriteLine($"Loading parameter '{pair.Key}'");
                pair.Value.Load(stream);
            }
        }

        /// <summary>
        /// Copy weights from default device to all other devices
        /// </summary>
        internal void CopyWeightsFromDefaultDeviceToAllOtherDevices()
        {
            this.RegisterTrainableParameters(this);
            foreach (var pair in this.m_name2network)
            {
                pair.Value.SyncWeights();
            }
        }


        /// <summary>
        /// Sum up gradients in all devices and keep them in the default device
        /// </summary>
        internal void SumGradientsToTensorsInDefaultDevice()
        {
            this.RegisterTrainableParameters(this);
            foreach (var pair in this.m_name2network)
            {
                pair.Value.SumGradientsToNetworkOnDefaultDevice();
            }
        }

        internal List<IWeightTensor> GetParametersFromDefaultDevice()
        {
            this.RegisterTrainableParameters(this);
            var result = new List<IWeightTensor>();
            foreach (var pair in this.m_name2network)
            {
                result.AddRange(pair.Value.GetNeuralUnitOnDefaultDevice().GetParams());
            }

            return result;
        }

        internal void ZeroGradientOnAllDevices()
        {
            this.RegisterTrainableParameters(this);
            foreach (var pair in this.m_name2network)
            {
                pair.Value.ZeroGradientsOnAllDevices();
            }
        }

        internal void RegisterTrainableParameters(object obj)
        {
            if (this.m_name2network != null)
            {
                return;
            }
            Logger.WriteLine($"Registering trainable parameters.");
            this.m_name2network = new SortedList<string, IMultiProcessorNetworkWrapper>();

            foreach (var childFieldInfo in obj.GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Instance))
            {
                var childValue = childFieldInfo.GetValue(obj);
                var name = childFieldInfo.Name;
                this.Register(childValue, name);
            }
            foreach (var childPropertyInfo in obj.GetType().GetProperties(BindingFlags.NonPublic | BindingFlags.Instance))
            {
                var childValue = childPropertyInfo.GetValue(obj);
                var name = childPropertyInfo.Name;
                this.Register(childValue, name);
            }
        }

        private void Register(object childValue, string name)
        {
            var networks = childValue as IMultiProcessorNetworkWrapper;
            if (networks != null)
            {
                this.m_name2network.Add(name, networks);
                Logger.WriteLine($"Register network '{name}'");
            }
        }
    }
}
