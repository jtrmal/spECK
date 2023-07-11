#pragma once
#include <cuda.h>
#include <cooperative_groups.h>
#include <iostream>
#ifdef __NVCC__
  #define _TARGET_ __host__ __device__
#else
  #define _TARGET_
#endif

#pragma pack(push, 1)
class Semiring {
  public:
  float weight;
    _TARGET_ inline Semiring(): weight(0.0f) {};
    _TARGET_ inline Semiring(float x): weight(x) {};

   _TARGET_ static Semiring zero() {return Semiring(0.0f);};
   _TARGET_ static Semiring one() {return Semiring(1.0f);};
   _TARGET_ float &val() {return weight;};

   _TARGET_ inline Semiring plus(const Semiring &x) const {
    return Semiring(this->weight + x.weight);
   }
   _TARGET_ inline Semiring times(const Semiring &x) const {
    return Semiring(this->weight * x.weight);
   }

};
#pragma pack(pop)
static_assert(sizeof(Semiring) == sizeof(float), "Class size is not equivalent to float");

_TARGET_ inline Semiring  operator* (const Semiring &x, const Semiring& y)
{
    return x.times(y);
}

_TARGET_ inline Semiring  operator+ (const Semiring &x, const Semiring& y)
{
    return x.plus(y);
}



_TARGET_ inline Semiring& operator +=(Semiring& x, const Semiring& y)
{
    x = x.plus(y);
    return x;
}

_TARGET_ inline std::istream& operator>> (std::istream& is, Semiring dt) {
  is >> dt.val();
  return is;
}

