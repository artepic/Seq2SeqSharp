﻿using AdvUtils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using TensorSharp.Core;

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    internal class AdvFuncKernels : CudaCode
    {
        private static readonly string Code = @"
extern ""C""
{

__global__ void gLNormalization(float* out,
                                const float* in,
                                const float* alpha,
                                const float* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      float* _sum = _share;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share;

      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = sp[id] - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float t = alpha[id] * ((sp[id] - mean) / sigma);
          if(beta)
            t += beta[id];
          so[id] = t;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void gLayerNormalizationGrad(float* gradX,
                                        float* gradGamma,
                                        float* gradBeta,
                                        float* adj,
                                        float* y,
                                        float* x,
                                        float* gamma,
                                        float* beta,
                                        int rows,
                                        int cols,
                                        float eps = 1e-9) {
  extern __shared__ float shared[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* sum_adj = shared;
      float* sum_adj_x = shared + blockDim.x;
      float* sum_x = shared + 2 * blockDim.x;
      float* sum_sqr = shared + 3 * blockDim.x;

      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      sum_x[threadIdx.x] = 0.0f;
      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += xRow[id];
          sum_adj_x[threadIdx.x]
              += adjRow[id] * (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          sum_adj[threadIdx.x] += adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x] += sum_x[threadIdx.x + skip];
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = sum_x[0] / cols;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = xRow[id] - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (sum_sqr[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float grad_x = 0.0f;
          float x_hat = (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          grad_x += cols * adjRow[id];
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          float valX = gamma[id] * grad_x;
          float sign = (0.f < valX) - (valX < 0.f);
          valX = fabs(valX) > 1000 ? sign * 1000 : valX;

          gradXRow[id] += valX;
          atomicAdd(gradGamma + id, adjRow[id] * x_hat);
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
    __syncthreads();
  }
}


__global__ void gAddLNormalization(float* out,
                                const float* in1,
                                const float* in2,
                                const float* alpha,
                                const float* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp1 = in1 + j * cols;
      const float* sp2 = in2 + j * cols;

      float* _sum = _share;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (sp1[id] + sp2[id]);
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share;

      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = (sp1[id] + sp2[id]) - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float t = alpha[id] * (((sp1[id] + sp2[id]) - mean) / sigma);
          if(beta)
            t += beta[id];
          so[id] = t;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void UpdateCost(float* weights, int* ids, float* costs, unsigned rows, unsigned cols)
{
    for(int bid = 0; bid < rows; bid += gridDim.x) 
    {
      int j = bid + blockIdx.x;
      if(j < rows && threadIdx.x == 0) 
      {
        float* sw = weights + j * cols;
        int sid = ids[j];

        if (sid >= 0)
        {
          costs[j] = -logf(sw[sid]);
          sw[sid] -= 1.0f;
        }
        else
        {
          costs[j] = 0.0f;
          sw[sid] = 0.0f;
        }
      }
    }
}


__global__ void BuildPadSelfTriMask(float* weights, int* originalLengths, int batchSize, unsigned rows, unsigned cols)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      int paddedLength = cols;
      float* weights_j = weights + j * cols;

      int originalLen = originalLengths[j / paddedLength];

      int row_i = j % paddedLength;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {

            if (row_i >= id && id < originalLen)
            {
                  weights_j[id] = 0.0f;
            }
            else
            {
                  weights_j[id] = -1e30f;
            }

        }
      }

    }

  }
}


__global__ void BuildSrcTgtMask(float* weights, int* originalSrcLengths, int* originalTgtLengths, int batchSize, unsigned rows, unsigned cols)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      //int paddedSrcLength = cols;
      int paddedTgtLength = rows / batchSize;

      float* weights_j = weights + j * cols;

      int originalSrcLen = originalSrcLengths[j / paddedTgtLength];
      int originalTgtLen = originalTgtLengths[j / paddedTgtLength];

      int row_i = j % paddedTgtLength;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {

           if (row_i < originalTgtLen && id < originalSrcLen)
           {
              weights_j[id] = 0.0f;
           }
           else
           {
              weights_j[id] = -1e30f;
           }
        }
      }
    }

  }
}




__global__ void gAddLayerNormalizationGrad(float* gradX1,
                                        float* gradX2,
                                        float* gradGamma,
                                        float* gradBeta,
                                        float* adj,
                                        float* y,
                                        float* x1,
                                        float* x2,
                                        float* gamma,
                                        float* beta,
                                        int rows,
                                        int cols,
                                        float eps = 1e-9) {
  extern __shared__ float shared[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* sum_adj = shared;
      float* sum_adj_x = shared + blockDim.x;
      float* sum_x = shared + 2 * blockDim.x;
      float* sum_sqr = shared + 3 * blockDim.x;

      const float* x1Row = x1 + j * cols;
      const float* x2Row = x2 + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradX1Row = gradX1 + j * cols;
      float* gradX2Row = gradX2 + j * cols;

      sum_x[threadIdx.x] = 0.0f;
      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += (x1Row[id] + x2Row[id]);
          sum_adj_x[threadIdx.x]
              += adjRow[id] * (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          sum_adj[threadIdx.x] += adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x] += sum_x[threadIdx.x + skip];
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = sum_x[0] / cols;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = (x1Row[id] + x2Row[id]) - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (sum_sqr[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float grad_x = 0.0f;
          float x_hat = (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          grad_x += cols * adjRow[id];
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          float valX = gamma[id] * grad_x;
          float sign = (0.f < valX) - (valX < 0.f);
          valX = fabs(valX) > 1000 ? sign * 1000 : valX;

          gradX1Row[id] += valX;
          gradX2Row[id] += valX;
          atomicAdd(gradGamma + id, adjRow[id] * x_hat);
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
         }
        }
      }
    }
    __syncthreads();
  }
}



__global__ void Adam(float* __restrict__ w, float* __restrict__ g, float* __restrict__ v, float* __restrict__ m, unsigned rows, unsigned cols, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) 
  {
    int j = bid + blockIdx.x;
    if(j < rows) 
    {
      float* sw = w + j * cols;
      float* sg = g + j * cols;
      float* sv = v + j * cols;
      float* sm = m + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) 
      {        
        int i = tid + threadIdx.x;
        if(i < cols)
        {
          // float g = sg[i] / batchSize;
           
           //if (g > clipval)
           //{
           //    g = clipval;
           //}
           //if (g < -clipval)
           //{
           //    g = -clipval;
           //}

           float g = sg[i];

           sm[i] = sm[i] * decay_rate_m + (1.0 - decay_rate_m) * g;
           sv[i] = sv[i] * decay_rate_v + (1.0 - decay_rate_v) * g * g;

           //float m_cap = sm[i] / (1.0 - powf(decay_rate_m, iter));
           //float v_cap = sv[i] / (1.0 - powf(decay_rate_v, iter));


           float bias_correction1 = 1.0 / (1.0 - powf(decay_rate_m, iter));
           float bias_correction2 = 1.0 / (1.0 - powf(decay_rate_v, iter));
           float adapted_learning_rate = step_size * bias_correction1 * rsqrtf(bias_correction2);


           sw[i] -= adapted_learning_rate * sm[i] / (sqrtf(sv[i]) + eps);

           sg[i] = 0;
        }
      }
    }
  }
}




__global__ void RMSProp(float* __restrict__ w, float* __restrict__ g, float* __restrict__ c, unsigned rows, unsigned cols, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) 
  {
    int j = bid + blockIdx.x;
    if(j < rows) 
    {
      float* sw = w + j * cols;
      float* sg = g + j * cols;
      float* sc = c + j * cols;
      
      for(int tid = 0; tid < cols; tid += blockDim.x) 
      {        
        int i = tid + threadIdx.x;
        if(i < cols && sg[i] != 0.0) 
        {
           float g = sg[i] / batchSize;
           
           if (g > clipval)
           {
               g = clipval;
           }
           if (g < -clipval)
           {
               g = -clipval;
           }

           sc[i] = sc[i] * decay_rate + (1.0 - decay_rate) * g * g;

           g = g * rsqrtf(sc[i] + eps);

           sw[i] -= g * step_size + sw[i] * regc;

           sg[i] = 0;
        }
      }
    }
  }
}


  __global__ void gSoftmaxGrad(float* grad, float* adj, float* val, int rows, int cols, int addGrad)
  {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share;

      float* gradRow = grad + j * cols;
      const float* adjRow = adj + j * cols;
      const float* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float v = valRow[id] * adjRow[id];

          if (addGrad == 0)
          {
             gradRow[id] = v;
          }
          else
          {
             gradRow[id] += v;
          }
          _sum[threadIdx.x] += v;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sum = _sum[0];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
         gradRow[id] -= sum * valRow[id];
        }
      }
    }
    __syncthreads();
  }
}

  __global__ void gSoftmax(float* out, float* in, unsigned rows, unsigned cols)
  {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      extern __shared__ float _share[];     
      float* _max = _share;
      _max[threadIdx.x] = -1.70141e+38;
      
      for(int tid = 0; tid < cols; tid += blockDim.x) {        
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(sp[i] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[i];
        }
      }
      __syncthreads();      
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();
    
      float* _sum = _share;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {         
          float ex = expf(sp[i] - max);
          so[i] = ex;
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();     
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
    
      float sum = _sum[0];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          so[i] = so[i] / sum;
        }
      }
    }
    __syncthreads();
  }
}



  __global__ void gSoftmaxMask(float* out, float* in, float *mask, unsigned rows, unsigned cols, unsigned maskRows)
  {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;
      const float* mp = mask + (j % maskRows) * cols;

      extern __shared__ float _share[];     
      float* _max = _share;
      _max[threadIdx.x] = -1.70141e+38;
      
      for(int tid = 0; tid < cols; tid += blockDim.x) {        
        int i = tid + threadIdx.x;
        if(i < cols && mp[i] == 0.0f) {
          if(sp[i] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[i];
        }
      }
      __syncthreads();      
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();
    
      float* _sum = _share;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols && mp[i] == 0.0f) {         
          float ex = expf(sp[i] - max);
          so[i] = ex;
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();     
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
    
      float sum = _sum[0];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {

          if (mp[i] == 0.0f)
          {
             so[i] = so[i] / sum;
          }
          else
          {
            so[i] = 0.0f;
          }
        }
      }
    }
    __syncthreads();
  }
}
}

";

        public AdvFuncKernels()
            : base(GetFullCode())
        {
        }

        private static string GetFullCode()
        {
            return Code;
        }

 
        private void BuildPadSelfTriMask(TSCudaContext context, Tensor mask, Tensor originalLengths, int batchSize)
        {
            var cudaContext = context.CudaContextForTensor(mask);

            cudaContext.SetCurrent();

            var ndim = mask.DimensionCount;
            long rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                rows *= mask.Sizes[dim];
            }

            var cols = mask.Sizes[ndim - 1];

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));


            var maskPtr = CudaHelpers.GetBufferStart(mask);
            var originalLengthsPtr = CudaHelpers.GetBufferStart(originalLengths);

            this.Invoke(context, cudaContext, "BuildPadSelfTriMask", grid, threads, 0, CUstream.NullStream, maskPtr, originalLengthsPtr, batchSize, rows, cols);
        }

        public Tensor BuildPadSelfTriMask(Tensor originalLengths, int batchSize, int paddedLength)
        {
            var context = CudaHelpers.TSContextForTensor(originalLengths);
            var writeTarget = TensorResultBuilder.GetWriteTarget(null, originalLengths.Allocator, DType.Float32, true, new long[] { batchSize, paddedLength, paddedLength });

            this.BuildPadSelfTriMask(context, writeTarget, originalLengths, batchSize);

            return writeTarget;
        }



        //BuildSrcTgtMask(float* weights, int* originalSrcLengths, int* originalTgtLengths, int batchSize, unsigned rows, unsigned cols)

        private void BuildSrcTgtMask(TSCudaContext context, Tensor mask, Tensor originalSrcLengths, Tensor originalTgtLengths, int batchSize)
        {
            var cudaContext = context.CudaContextForTensor(mask);

            cudaContext.SetCurrent();

            var ndim = mask.DimensionCount;
            long rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                rows *= mask.Sizes[dim];
            }

            var cols = mask.Sizes[ndim - 1];

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));


            var maskPtr = CudaHelpers.GetBufferStart(mask);
            var originalSrcLengthsPtr = CudaHelpers.GetBufferStart(originalSrcLengths);
            var originalTgtLengthsPtr = CudaHelpers.GetBufferStart(originalTgtLengths);

            this.Invoke(context, cudaContext, "BuildSrcTgtMask", grid, threads, 0, CUstream.NullStream, maskPtr, originalSrcLengthsPtr, originalTgtLengthsPtr, batchSize, rows, cols);
        }

        public Tensor BuildSrcTgtMask(Tensor originalSrcLengths, Tensor originalTgtLengths, int batchSize, int paddedSrcLength, int paddedTgtLength)
        {
            var context = CudaHelpers.TSContextForTensor(originalSrcLengths);
            var writeTarget = TensorResultBuilder.GetWriteTarget(null, originalSrcLengths.Allocator, DType.Float32, true, new long[] { batchSize, paddedTgtLength, paddedSrcLength });

            this.BuildSrcTgtMask(context, writeTarget, originalSrcLengths, originalTgtLengths, batchSize);

            return writeTarget;
        }



        private void UpdateCost(TSCudaContext context, Tensor weight, Tensor ids, Tensor costs)
        {
            var cudaContext = context.CudaContextForTensor(weight);

            cudaContext.SetCurrent();

            var ndim = weight.DimensionCount;
            long rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                rows *= weight.Sizes[dim];
            }

            var cols = weight.Sizes[ndim - 1];

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var weightPtr = CudaHelpers.GetBufferStart(weight);
            var idsPtr = CudaHelpers.GetBufferStart(ids);
            var costsPtr = CudaHelpers.GetBufferStart(costs);

            this.Invoke(context, cudaContext, "UpdateCost", grid, threads, 0, CUstream.NullStream, weightPtr, idsPtr, costsPtr, rows, cols);
        }


        public Tensor UpdateCost(Tensor costs, Tensor weight, Tensor ids)
        {
            var context = CudaHelpers.TSContextForTensor(weight);
            var writeTarget = TensorResultBuilder.GetWriteTarget(costs, weight, true, ids.Sizes);

            Ops.Fill(writeTarget, 0.0f);

            this.UpdateCost(context, weight, ids, writeTarget);

            return writeTarget;
        }


        public Tensor LayerNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(inGrad);
            var writeTarget = TensorResultBuilder.GetWriteTarget(outGrad, inGrad, false, inGrad.Sizes);
            this.LayerNormGrad(context, writeTarget, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps);

            return writeTarget;
        }


        private void LayerNormGrad(TSCudaContext context, Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(inGrad);

            cudaContext.SetCurrent();

            var ndim = inGrad.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(inGrad.Sizes, inGrad.Strides);
            var cols = inGrad.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var outGradPtr = CudaHelpers.GetBufferStart(outGrad);
            var alphaGradPtr = CudaHelpers.GetBufferStart(alphaGrad);
            var betaGradPtr = CudaHelpers.GetBufferStart(betaGrad);
            var inGradPtr = CudaHelpers.GetBufferStart(inGrad);
            var yPtr = CudaHelpers.GetBufferStart(y);
            var xPtr = CudaHelpers.GetBufferStart(x);
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);

            this.Invoke(context, cudaContext, "gLayerNormalizationGrad", grid, threads, threads.x * sizeof(float) * 4, CUstream.NullStream, outGradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, xPtr, alphaPtr, betaPtr, rows, cols, eps);

        }


        public void AddLayerNormGrad(Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(inGrad);
            var writeTarget1 = TensorResultBuilder.GetWriteTarget(out1Grad, inGrad, false, inGrad.Sizes);
            var writeTarget2 = TensorResultBuilder.GetWriteTarget(out2Grad, inGrad, false, inGrad.Sizes);
            this.AddLayerNormGrad(context, writeTarget1, writeTarget2, alphaGrad, betaGrad, inGrad, y, x1, x2, alpha, beta, eps);
        }

        private void AddLayerNormGrad(TSCudaContext context, Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(inGrad);

            cudaContext.SetCurrent();
            var ndim = inGrad.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(inGrad.Sizes, inGrad.Strides);
            var cols = inGrad.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var out1GradPtr = CudaHelpers.GetBufferStart(out1Grad);
            var out2GradPtr = CudaHelpers.GetBufferStart(out2Grad);
            var alphaGradPtr = CudaHelpers.GetBufferStart(alphaGrad);
            var betaGradPtr = CudaHelpers.GetBufferStart(betaGrad);
            var inGradPtr = CudaHelpers.GetBufferStart(inGrad);
            var yPtr = CudaHelpers.GetBufferStart(y);
            var x1Ptr = CudaHelpers.GetBufferStart(x1);
            var x2Ptr = CudaHelpers.GetBufferStart(x2);
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);

            this.Invoke(context, cudaContext, "gAddLayerNormalizationGrad", grid, threads, threads.x * sizeof(float) * 4, CUstream.NullStream, out1GradPtr, out2GradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, x1Ptr, x2Ptr, alphaPtr, betaPtr, rows, cols, eps);

        }


        public Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            this.LayerNorm(context, writeTarget, src, alpha, beta, eps);

            return writeTarget;
        }


        private void LayerNorm(TSCudaContext context, Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var ndim = src.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            var cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;


            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);

            this.Invoke(context, cudaContext, "gLNormalization", grid, threads, threads.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, alphaPtr, betaPtr, rows, cols, eps);

        }

        public Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(src1);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src1, false, src1.Sizes);
            this.AddLayerNorm(context, writeTarget, src1, src2, alpha, beta, eps);

            return writeTarget;
        }

        private void AddLayerNorm(TSCudaContext context, Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(src1);

            cudaContext.SetCurrent();

            var ndim = src1.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(src1.Sizes, src1.Strides);
            var cols = src1.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;


            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var src1Ptr = CudaHelpers.GetBufferStart(src1);
            var src2Ptr = CudaHelpers.GetBufferStart(src2);
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);

            this.Invoke(context, cudaContext, "gAddLNormalization", grid, threads, threads.x * sizeof(float), CUstream.NullStream, resultPtr, src1Ptr, src2Ptr, alphaPtr, betaPtr, rows, cols, eps);

        }


        private void Softmax(TSCudaContext context, Tensor result, Tensor src)
        {
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var ndim = src.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            var cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;


            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);

            this.Invoke(context, cudaContext, "gSoftmax", grid, threads, threads.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, rows, cols);
        }


        private void SoftmaxMask(TSCudaContext context, Tensor result, Tensor src, Tensor mask)
        {
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var ndim = src.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            var cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;




            var maskNdim = mask.DimensionCount;
            var maskStorageSize = TensorDimensionHelpers.GetStorageSize(mask.Sizes, mask.Strides);
            var maskCols = mask.Sizes[maskNdim - 1];

            if (maskStorageSize % maskCols != 0)
            {
                throw new Exception($"Invalid mask tensor storage size = '{maskStorageSize}', and cols = '{maskCols}'");
            }

            var maskRows = maskStorageSize / maskCols;

            if (rows % maskRows != 0)
            {
                throw new Exception($"Invalid tensor rows = '{rows}' and mask tensor rows = '{maskRows}'");
            }

            if (cols != maskCols)
            {
                throw new Exception($"Tensor cols = '{cols}', mask tensor cols = '{maskCols}'. They should be equal.");
            }


            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);
            var maskPtr = CudaHelpers.GetBufferStart(mask);

            this.Invoke(context, cudaContext, "gSoftmaxMask", grid, threads, threads.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, maskPtr, rows, cols, maskRows);
        }


        private void Adam(TSCudaContext context, Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            var cudaContext = context.CudaContextForTensor(weight);

            cudaContext.SetCurrent();

            var ndim = weight.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(weight.Sizes, weight.Strides);
            var cols = weight.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var weightPtr = CudaHelpers.GetBufferStart(weight);
            var gradientPtr = CudaHelpers.GetBufferStart(gradient);
            var vPtr = CudaHelpers.GetBufferStart(v);
            var mPtr = CudaHelpers.GetBufferStart(m);

            this.Invoke(context, cudaContext, "Adam", grid, threads, 0, CUstream.NullStream, weightPtr, gradientPtr, vPtr, mPtr, rows, cols, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
        }

        public Tensor Adam(Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            var context = CudaHelpers.TSContextForTensor(weight);
            this.Adam(context, weight, gradient, v, m, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);

            return weight;
        }

        private void RMSProp(TSCudaContext context, Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            var cudaContext = context.CudaContextForTensor(weight);

            cudaContext.SetCurrent();

            var ndim = weight.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(weight.Sizes, weight.Strides);
            var cols = weight.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var weightPtr = CudaHelpers.GetBufferStart(weight);
            var gradientPtr = CudaHelpers.GetBufferStart(gradient);
            var cachePtr = CudaHelpers.GetBufferStart(cache);

            this.Invoke(context, cudaContext, "RMSProp", grid, threads, 0, CUstream.NullStream, weightPtr, gradientPtr, cachePtr, rows, cols, batchSize, step_size, clipval, regc, decay_rate, eps);
        }

        public Tensor RMSProp(Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            var context = CudaHelpers.TSContextForTensor(weight);
            this.RMSProp(context, weight, gradient, cache, batchSize, step_size, clipval, regc, decay_rate, eps);

            return weight;
        }

        private void SoftmaxGrad(TSCudaContext context, Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            var cudaContext = context.CudaContextForTensor(grad);

            cudaContext.SetCurrent();

            var ndim = grad.DimensionCount;
            var storageSize = TensorDimensionHelpers.GetStorageSize(grad.Sizes, grad.Strides);
            var cols = grad.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            var rows = storageSize / cols;

            var iAddGrad = addGrad ? 1 : 0;

            var threads = new dim3((uint)Math.Min(512, rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, threads.y)));

            var gradPtr = CudaHelpers.GetBufferStart(grad);
            var adjPtr = CudaHelpers.GetBufferStart(adj);
            var valPtr = CudaHelpers.GetBufferStart(val);

            this.Invoke(context, cudaContext, "gSoftmaxGrad", grid, threads, threads.x * sizeof(float), CUstream.NullStream, gradPtr, adjPtr, valPtr, rows, cols, iAddGrad);
        }


        public Tensor Softmax(Tensor result, Tensor src)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            this.Softmax(context, writeTarget, src);

            return writeTarget;
        }


        public Tensor SoftmaxMask(Tensor result, Tensor src, Tensor mask)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            this.SoftmaxMask(context, writeTarget, src, mask);

            return writeTarget;
        }


        public Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            var context = CudaHelpers.TSContextForTensor(grad);
            this.SoftmaxGrad(context, grad, adj, val, addGrad);

            return grad;
        }


        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
        {
            var ptx = this.GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
