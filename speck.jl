lib="libjulia_specklib"


function speck_spgemm(A, B)
    Uint32 nnzC;

    @ccal lib.julia_multiply_float(A.m, A.n, B.n, 
                                   nnz(A), A::Ptr{Cuint}, A::Ptr{Cuint}, A::Ptr{Cfloat}
                                   nnz(B), B::Ptr{Cuint}, B::Ptr{Cuint}, B::Ptr{Cfloat}
                                   Ref{nnz}, B::Ptr{Cuint}, B::Ptr{Cuint}, B::Ptr{Cfloat}
  )::Cvoid
    @ccal lib._dealloc_matrix()
    return C
end
