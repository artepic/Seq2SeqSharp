
#include "ElementwiseOps.h"
#include "TensorApply-inl.h"
#include <cmath>

#define DECLARE_UNARY_FLOAT_TYPES(EXPORTNAME, FUNCNAME)\
template<typename T>\
INLINE_FUNC void EXPORTNAME##_Apply(TensorRef* result, TensorRef* src)\
{\
	auto func = [](T *r, T *s) { *r = FUNCNAME(*s); };\
	Apply2<T, T>(result, src, func);\
}\
int EXPORTNAME(TensorRef* result, TensorRef* src)\
{\
	API_BEGIN()\
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, EXPORTNAME##_Apply, result, src)\
	API_END()\
}

#define DECLARE_UNARY_ALL_CPU_TYPES(EXPORTNAME, FUNCNAME)\
template<typename T>\
INLINE_FUNC void EXPORTNAME##_Apply(TensorRef* result, TensorRef* src)\
{\
	auto func = [](T *r, T *s) { *r = FUNCNAME(*s); };\
	Apply2<T, T>(result, src, func);\
}\
int EXPORTNAME(TensorRef* result, TensorRef* src)\
{\
	API_BEGIN()\
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, EXPORTNAME##_Apply, result, src)\
	API_END()\
}

INLINE_FUNC uint8 Mod_op(uint8 x, uint8 y) { return x % y; }
INLINE_FUNC __int32 Mod_op(__int32 x, __int32 y) { return x % y; }
INLINE_FUNC float Mod_op(float x, float y) { return fmod(x, y); }
INLINE_FUNC double Mod_op(double x, double y) { return fmod(x, y); }


template<typename T> INLINE_FUNC T rsub_op(T x, T y) { return (T)(y - x); }
template<typename T> INLINE_FUNC T rdiv_op(T x, T y) { return (T)(y / x); }

#define INFIX_TO_FUNC(OPNAME, OPERATOR) template<typename T> INLINE_FUNC T OPNAME(T x, T y) { return (T)(x OPERATOR y); }
INFIX_TO_FUNC(add_op, +)
INFIX_TO_FUNC(sub_op, -)
INFIX_TO_FUNC(mul_op, *)
INFIX_TO_FUNC(div_op, /)

INFIX_TO_FUNC(gt_op, >)
INFIX_TO_FUNC(lt_op, <)
INFIX_TO_FUNC(ge_op, >=)
INFIX_TO_FUNC(le_op, <=)
INFIX_TO_FUNC(eq_op, ==)
INFIX_TO_FUNC(ne_op, !=)


template<typename T> INLINE_FUNC T Neg(T x) {
	return -x;
}

template<typename T> INLINE_FUNC T Frac(T x) {
	return x - trunc(x);
}

template<typename T> INLINE_FUNC T Lerp(T a, T b, T weight) {
	return a + weight * (b - a);
}

template<typename T> INLINE_FUNC T Sigmoid(T x) {
	return T(1) / (T(1) + exp(-x));
}

template<typename T> INLINE_FUNC T TanhD(T resW, T resG) {
	return (T(1) - resW * resW) * resG;
}

template<typename T> INLINE_FUNC T SigmoidD(T resW, T resG) {
	return resW * (T(1) - resW) * resG;
}

template<typename T> INLINE_FUNC T AddTanhD(T t, T resW, T resG) {
	return t + (T(1) - resW * resW) * resG;
}

template<typename T> INLINE_FUNC T AddMul(T x, T y, T z) {
	return x + y * z;
}

template<typename T> INLINE_FUNC T MulMulAdd(T x, T y, T z, T w) {
	return x * y + z * w;
}

template<typename T> INLINE_FUNC T AddTanh(T x, T y) {
	return tanh(x + y);
}

template<typename T> INLINE_FUNC T AddTanh3(T x, T y, T z) {
	return tanh(x + y + z);
}

template <typename T> INLINE_FUNC T sgn(T val) {
	if (val < T(0))
		return T(-1);
	if (val > T(0))
		return T(1);
	return T(0);
}

template <typename T> INLINE_FUNC T Clamp(T val, T min, T max) {
	if (val < min)
		return min;
	if (val > max)
		return max;
	return val;
}

template <typename T> INLINE_FUNC T MaskFill(T t, T mask, T defValue) {
	if (mask == T(0))
		return t;
	return defValue;
}

DECLARE_UNARY_ALL_CPU_TYPES(TS_Abs, abs)
DECLARE_UNARY_ALL_CPU_TYPES(TS_Neg, Neg)
DECLARE_UNARY_ALL_CPU_TYPES(TS_Sign, sgn)


DECLARE_UNARY_FLOAT_TYPES(TS_Sqrt, sqrt)
DECLARE_UNARY_FLOAT_TYPES(TS_Exp, exp)
DECLARE_UNARY_FLOAT_TYPES(TS_Log, log)
DECLARE_UNARY_FLOAT_TYPES(TS_Log1p, log1p)
DECLARE_UNARY_FLOAT_TYPES(TS_Floor, floor)
DECLARE_UNARY_FLOAT_TYPES(TS_Ceil, ceil)
DECLARE_UNARY_FLOAT_TYPES(TS_Round, round)
DECLARE_UNARY_FLOAT_TYPES(TS_Trunc, trunc)
DECLARE_UNARY_FLOAT_TYPES(TS_Frac, Frac)

DECLARE_UNARY_FLOAT_TYPES(TS_Sin, sin)
DECLARE_UNARY_FLOAT_TYPES(TS_Cos, cos)
DECLARE_UNARY_FLOAT_TYPES(TS_Tan, tan)
DECLARE_UNARY_FLOAT_TYPES(TS_Asin, asin)
DECLARE_UNARY_FLOAT_TYPES(TS_Acos, acos)
DECLARE_UNARY_FLOAT_TYPES(TS_Atan, atan)
DECLARE_UNARY_FLOAT_TYPES(TS_Sinh, sinh)
DECLARE_UNARY_FLOAT_TYPES(TS_Cosh, cosh)
DECLARE_UNARY_FLOAT_TYPES(TS_Tanh, tanh)

DECLARE_UNARY_FLOAT_TYPES(TS_Relu, relu)

DECLARE_UNARY_FLOAT_TYPES(TS_Sigmoid, Sigmoid)

template <typename T> INLINE_FUNC T relu(T w) {
	if (w < T(0))
		return T(0);
	return w;
}

template <typename T> INLINE_FUNC T relud(T w, T g) {
	if (w > T(0))
		return g;
	return T(0);
}

template <typename T> INLINE_FUNC T AddReluD(T t, T w, T g) {
	if (w > T(0))
		return t + g;
	return t + T(0);
}

template<typename T>
INLINE_FUNC void TanhD_Apply(TensorRef* result, TensorRef* resW, TensorRef* resG)
{
	auto func = [](T *r, T *x, T *y) { *r = TanhD(*x, *y); };
	Apply3<T, T, T>(result, resW, resG, func);
}

int TS_TanhD(TensorRef* result, TensorRef* resW, TensorRef* resG)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, TanhD_Apply, result, resW, resG)
	API_END()
}

template<typename T>
INLINE_FUNC void SigmoidD_Apply(TensorRef* result, TensorRef* resW, TensorRef* resG)
{
	auto func = [](T *r, T *x, T *y) { *r = SigmoidD(*x, *y); };
	Apply3<T, T, T>(result, resW, resG, func);
}

int TS_SigmoidD(TensorRef* result, TensorRef* resW, TensorRef* resG)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, SigmoidD_Apply, result, resW, resG)
		API_END()
}

template<typename T>
INLINE_FUNC void Atan2_Apply(TensorRef* result, TensorRef* srcY, TensorRef* srcX)
{
	auto func = [](T *r, T *y, T *x) { *r = atan2(*y, *x); };
	Apply3<T, T, T>(result, srcY, srcX, func);
}

int TS_Atan2(TensorRef* result, TensorRef* srcY, TensorRef* srcX)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Atan2_Apply, result, srcY, srcX)
	API_END()
}

template<typename T>
INLINE_FUNC void ReluD_Apply(TensorRef* result, TensorRef* srcW, TensorRef* srcG)
{
	auto func = [](T* r, T* y, T* x) { *r = relud(*y, *x); };
	Apply3<T, T, T>(result, srcW, srcG, func);
}

int TS_ReluD(TensorRef* result, TensorRef* srcW, TensorRef* srcG)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, ReluD_Apply, result, srcW, srcG)
		API_END()
}

template<typename T>
INLINE_FUNC void Pow_Apply(TensorRef* result, TensorRef* src, float value)
{
	auto func = [value](T *r, T *s) { *r = pow(*s, (T)value); };
	Apply2<T, T>(result, src, func);
}

int TS_Pow(TensorRef* result, TensorRef* src, float value)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Pow_Apply, result, src, value)
	API_END()
}

template<typename T>
INLINE_FUNC void Tpow_Apply(TensorRef* result, float value, TensorRef* src)
{
	auto func = [value](T *r, T *s) { *r = pow((T)value, *s); };
	Apply2<T, T>(result, src, func);
}

int TS_Tpow(TensorRef* result, float value, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Tpow_Apply, result, value, src)
	API_END()
}

template<typename T>
INLINE_FUNC void Lerp_Apply(TensorRef* result, TensorRef* srcA, TensorRef* srcB, float weight)
{
	auto func = [weight](T *r, T *a, T *b) { *r = Lerp(*a, *b, (T)weight); };
	Apply3<T, T, T>(result, srcA, srcB, func);
}

int TS_Lerp(TensorRef* result, TensorRef* srcA, TensorRef* srcB, float weight)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Lerp_Apply, result, srcA, srcB, weight)
	API_END()
}



template<typename T>
INLINE_FUNC void MulMulAdd_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ, TensorRef* srcW)
{
	auto func = [](T *r, T *x, T *y, T *z, T *w) { *r = MulMulAdd(*x, *y, *z, *w); };
	Apply5<T, T, T, T, T>(result, srcX, srcY, srcZ, srcW, func);
}

int TS_MulMulAdd(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ, TensorRef* srcW)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, MulMulAdd_Apply, result, srcX, srcY, srcZ, srcW)
		API_END()
}


template<typename T>
INLINE_FUNC void AddTanh_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY)
{
	auto func = [](T *r, T *x, T *y) { *r = AddTanh(*x, *y); };
	Apply3<T, T, T>(result, srcX, srcY, func);
}

int TS_AddTanh(TensorRef* result, TensorRef* srcX, TensorRef* srcY)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, AddTanh_Apply, result, srcX, srcY)
		API_END()
}


template<typename T>
INLINE_FUNC void AddTanh3_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	auto func = [](T* r, T* x, T* y, T* z) { *r = AddTanh3(*x, *y, *z); };
	Apply4<T, T, T, T>(result, srcX, srcY, srcZ, func);
}

int TS_AddTanh3(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, AddTanh3_Apply, result, srcX, srcY, srcZ)
		API_END()
}


template<typename T>
INLINE_FUNC void AddReluD_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	auto func = [](T *r, T *x, T *y, T *z) { *r = AddReluD(*x, *y, *z); };
	Apply4<T, T, T, T>(result, srcX, srcY, srcZ, func);
}

int TS_AddReluD(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, AddReluD_Apply, result, srcX, srcY, srcZ)
		API_END()
}


template<typename T>
INLINE_FUNC void AddTanhD_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	auto func = [](T *r, T *x, T *y, T *z) { *r = AddTanhD(*x, *y, *z); };
	Apply4<T, T, T, T>(result, srcX, srcY, srcZ, func);
}

int TS_AddTanhD(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, AddTanhD_Apply, result, srcX, srcY, srcZ)
		API_END()
}


template<typename T>
INLINE_FUNC void AddMul_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	auto func = [](T *r, T *x, T *y, T *z) { *r = AddMul(*x, *y, *z); };
	Apply4<T, T, T, T>(result, srcX, srcY, srcZ, func);
}

int TS_AddMul(TensorRef* result, TensorRef* srcX, TensorRef* srcY, TensorRef* srcZ)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, AddMul_Apply, result, srcX, srcY, srcZ)
		API_END()
}


template<typename T>
INLINE_FUNC void AddMulV_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY, float v)
{
	auto func = [v](T *r, T *x, T *y) { *r = AddMul(*x, *y, (T)v); };
	Apply3<T, T, T>(result, srcX, srcY, func);
}

int TS_AddMulV(TensorRef* result, TensorRef* srcX, TensorRef* srcY, float v)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, AddMulV_Apply, result, srcX, srcY, v)
		API_END()
}


template<typename T>
INLINE_FUNC void MaskFill_Apply(TensorRef* result, TensorRef* srcX, TensorRef* srcY, float v)
{
	auto func = [v](T* r, T* x, T* y) { *r = MaskFill(*x, *y, (T)v); };
	Apply3<T, T, T>(result, srcX, srcY, func);
}

int TS_MaskFill(TensorRef* result, TensorRef* srcX, TensorRef* srcY, float v)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_FLOAT(result->elementType, MaskFill_Apply, result, srcX, srcY, v)
		API_END()
}



template<typename T>
INLINE_FUNC void Clamp_Apply(TensorRef* result, TensorRef* src, float min, float max)
{
	auto func = [min, max](T *r, T *s) { *r = Clamp(*s, (T)min, (T)max); };
	Apply2<T, T>(result, src, func);
}

int TS_Clamp(TensorRef* result, TensorRef* src, float min, float max)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Clamp_Apply, result, src, min, max)
	API_END()
}


#define DECLARE_T_S_ALL_CPU_TYPES(EXPORTNAME, FUNCNAME)\
template<typename T>\
INLINE_FUNC void EXPORTNAME##_Apply(TensorRef* result, TensorRef* src, float value)\
{\
	auto func = [value](T *r, T *s) { *r = FUNCNAME(*s, (T)value); };\
	Apply2<T, T>(result, src, func);\
}\
int EXPORTNAME(TensorRef* result, TensorRef* lhs, float rhs)\
{\
	API_BEGIN()\
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, EXPORTNAME##_Apply, result, lhs, rhs)\
	API_END()\
}

DECLARE_T_S_ALL_CPU_TYPES(TS_Add, add_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_Sub, sub_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_Rsub, rsub_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_Mul, mul_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_Div, div_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_Rdiv, rdiv_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_Mod, Mod_op)

DECLARE_T_S_ALL_CPU_TYPES(TS_gtValue, gt_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_ltValue, lt_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_geValue, ge_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_leValue, le_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_eqValue, eq_op)
DECLARE_T_S_ALL_CPU_TYPES(TS_neValue, ne_op)


#define DECLARE_T_T_ALL_CPU_TYPES(EXPORTNAME, FUNCNAME)\
template<typename T>\
INLINE_FUNC void EXPORTNAME##_Apply(TensorRef* result, TensorRef* lhs, TensorRef* rhs)\
{\
	auto func = [](T *r, T *left, T *right) { *r = FUNCNAME(*left, *right); };\
	Apply3<T, T, T>(result, lhs, rhs, func);\
}\
int EXPORTNAME(TensorRef* result, TensorRef* lhs, TensorRef* rhs)\
{\
	API_BEGIN()\
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, EXPORTNAME##_Apply, result, lhs, rhs)\
	API_END()\
}

DECLARE_T_T_ALL_CPU_TYPES(TS_CAdd, add_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_CSub, sub_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_CMul, mul_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_CDiv, div_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_CMod, Mod_op)

DECLARE_T_T_ALL_CPU_TYPES(TS_gtTensor, gt_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_ltTensor, lt_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_geTensor, ge_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_leTensor, le_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_eqTensor, eq_op)
DECLARE_T_T_ALL_CPU_TYPES(TS_neTensor, ne_op)

