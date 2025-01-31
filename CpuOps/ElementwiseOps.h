#pragma once

#include "General.h"
#include "TensorRef.h"


OPS_API int TS_Abs(TensorRef* result, TensorRef* src);
OPS_API int TS_Neg(TensorRef* result, TensorRef* src);
OPS_API int TS_Sign(TensorRef* result, TensorRef* src);

OPS_API int TS_Sqrt(TensorRef* result, TensorRef* src);
OPS_API int TS_Exp(TensorRef* result, TensorRef* src);
OPS_API int TS_Log(TensorRef* result, TensorRef* src);
OPS_API int TS_Log1p(TensorRef* result, TensorRef* src);
OPS_API int TS_Floor(TensorRef* result, TensorRef* src);
OPS_API int TS_Ceil(TensorRef* result, TensorRef* src);
OPS_API int TS_Round(TensorRef* result, TensorRef* src);
OPS_API int TS_Trunc(TensorRef* result, TensorRef* src);
OPS_API int TS_Frac(TensorRef* result, TensorRef* src);

OPS_API int TS_Relu(TensorRef* result, TensorRef* src);
OPS_API int TS_ReluD(TensorRef* result, TensorRef* srcW, TensorRef* srcG);
OPS_API int TS_Sin(TensorRef* result, TensorRef* src);
OPS_API int TS_Cos(TensorRef* result, TensorRef* src);
OPS_API int TS_Tan(TensorRef* result, TensorRef* src);
OPS_API int TS_Asin(TensorRef* result, TensorRef* src);
OPS_API int TS_Acos(TensorRef* result, TensorRef* src);
OPS_API int TS_Atan(TensorRef* result, TensorRef* src);
OPS_API int TS_Sinh(TensorRef* result, TensorRef* src);
OPS_API int TS_Cosh(TensorRef* result, TensorRef* src);
OPS_API int TS_Tanh(TensorRef* result, TensorRef* src);

OPS_API int TS_Sigmoid(TensorRef* result, TensorRef* src);
//OPS_API int TS_Threshold(TensorRef* result, TensorRef* src, float threshold);
//OPS_API int TS_ThresholdGradInput(TensorRef* input, TensorRef* gradOutput, TensorRef* gradInput, float threshold);

OPS_API int TS_TanhD(TensorRef* result, TensorRef* resW, TensorRef* resG);
OPS_API int TS_SigmoidD(TensorRef* result, TensorRef* resW, TensorRef* resG);

OPS_API int TS_Atan2(TensorRef* result, TensorRef* srcY, TensorRef* srcX);
OPS_API int TS_Pow(TensorRef* result, TensorRef* src, float value);
OPS_API int TS_Tpow(TensorRef* result, float value, TensorRef* src);
OPS_API int TS_Lerp(TensorRef* result, TensorRef* srcA, TensorRef* srcB, float weight);
OPS_API int TS_Clamp(TensorRef* result, TensorRef* src, float min, float max);

OPS_API int TS_MulMulAdd(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ, TensorRef* srcW);
OPS_API int TS_AddTanh(TensorRef* result, TensorRef* srcX, TensorRef* srcY);
OPS_API int TS_AddTanh3(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ);
OPS_API int TS_AddTanhD(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ);
OPS_API int TS_AddReluD(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ);

OPS_API int TS_AddMul(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ);
OPS_API int TS_AddMulV(TensorRef* result, TensorRef* srcX, TensorRef* srcY, float v);

OPS_API int TS_MaskFill(TensorRef* result, TensorRef* t, TensorRef* mask, float defValue);

OPS_API int TS_Add(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_Sub(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_Rsub(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_Mul(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_Div(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_Rdiv(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_Mod(TensorRef* result, TensorRef* lhs, float rhs);

OPS_API int TS_gtValue(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_ltValue(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_geValue(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_leValue(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_eqValue(TensorRef* result, TensorRef* lhs, float rhs);
OPS_API int TS_neValue(TensorRef* result, TensorRef* lhs, float rhs);


OPS_API int TS_CAdd(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_CSub(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_CMul(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_CDiv(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_CMod(TensorRef* result, TensorRef* lhs, TensorRef* rhs);

OPS_API int TS_gtTensor(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_ltTensor(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_geTensor(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_leTensor(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_eqTensor(TensorRef* result, TensorRef* lhs, TensorRef* rhs);
OPS_API int TS_neTensor(TensorRef* result, TensorRef* lhs, TensorRef* rhs);

