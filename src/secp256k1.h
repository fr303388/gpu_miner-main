#pragma once


#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstring>  // For memcpy



/* Limbs of the secp256k1 order. */
#define SECP256K1_N_0 ((uint64_t)0xBFD25E8CD0364141ULL)
#define SECP256K1_N_1 ((uint64_t)0xBAAEDCE6AF48A03BULL)
#define SECP256K1_N_2 ((uint64_t)0xFFFFFFFFFFFFFFFEULL)
#define SECP256K1_N_3 ((uint64_t)0xFFFFFFFFFFFFFFFFULL)

/* Limbs of 2^256 minus the secp256k1 order. */
#define SECP256K1_N_C_0 (~SECP256K1_N_0 + 1)
#define SECP256K1_N_C_1 (~SECP256K1_N_1)
#define SECP256K1_N_C_2 (1)

/* Limbs of half the secp256k1 order. */
#define SECP256K1_N_H_0 ((uint64_t)0xDFE92F46681B20A0ULL)
#define SECP256K1_N_H_1 ((uint64_t)0x5D576E7357A4501DULL)
#define SECP256K1_N_H_2 ((uint64_t)0xFFFFFFFFFFFFFFFFULL)
#define SECP256K1_N_H_3 ((uint64_t)0x7FFFFFFFFFFFFFFFULL)

#define ECMULT_GEN_PREC_BITS 4

#define ECMULT_GEN_PREC_N 64


typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef unsigned long long LONG;
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

typedef uint128_t secp256k1_uint128;
typedef int128_t secp256k1_int128;



/** This field implementation represents the value as 5 uint64_t limbs in base
 *  2^52. */
typedef struct {
   /* A field element f represents the sum(i=0..4, f.n[i] << (i*52)) mod p,
    * where p is the field modulus, 2^256 - 2^32 - 977.
    *
    * The individual limbs f.n[i] can exceed 2^52; the field's magnitude roughly
    * corresponds to how much excess is allowed. The value
    * sum(i=0..4, f.n[i] << (i*52)) may exceed p, unless the field element is
    * normalized. */
    uint64_t n[5];
    /*
     * Magnitude m requires:
     *     n[i] <= 2 * m * (2^52 - 1) for i=0..3
     *     n[4] <= 2 * m * (2^48 - 1)
     *
     * Normalized requires:
     *     n[i] <= (2^52 - 1) for i=0..3
     *     sum(i=0..4, n[i] << (i*52)) < p
     *     (together these imply n[4] <= 2^48 - 1)
     */
    //SECP256K1_FE_VERIFY_FIELDS
} secp256k1_fe;

/* Unpacks a constant into a overlapping multi-limbed FE element. */
#define SECP256K1_FE_CONST_INNER(d7, d6, d5, d4, d3, d2, d1, d0) { \
    (d0) | (((uint64_t)(d1) & 0xFFFFFUL) << 32), \
    ((uint64_t)(d1) >> 20) | (((uint64_t)(d2)) << 12) | (((uint64_t)(d3) & 0xFFUL) << 44), \
    ((uint64_t)(d3) >> 8) | (((uint64_t)(d4) & 0xFFFFFFFUL) << 24), \
    ((uint64_t)(d4) >> 28) | (((uint64_t)(d5)) << 4) | (((uint64_t)(d6) & 0xFFFFUL) << 36), \
    ((uint64_t)(d6) >> 16) | (((uint64_t)(d7)) << 16) \
}

typedef struct {
    uint64_t n[4];
} secp256k1_fe_storage;

#define SECP256K1_FE_STORAGE_CONST(d7, d6, d5, d4, d3, d2, d1, d0) {{ \
    (d0) | (((uint64_t)(d1)) << 32), \
    (d2) | (((uint64_t)(d3)) << 32), \
    (d4) | (((uint64_t)(d5)) << 32), \
    (d6) | (((uint64_t)(d7)) << 32) \
}}

#define SECP256K1_FE_STORAGE_CONST_GET(d) \
    (uint32_t)(d.n[3] >> 32), (uint32_t)d.n[3], \
    (uint32_t)(d.n[2] >> 32), (uint32_t)d.n[2], \
    (uint32_t)(d.n[1] >> 32), (uint32_t)d.n[1], \
    (uint32_t)(d.n[0] >> 32), (uint32_t)d.n[0]



typedef struct {
    uint64_t d[4];
} secp256k1_scalar;

typedef struct {
    secp256k1_fe x; /* actual X: x/z^2 */
    secp256k1_fe y; /* actual Y: y/z^3 */
    secp256k1_fe z;
    int infinity; /* whether this represents the point at infinity */
} secp256k1_gej;

typedef struct {
    secp256k1_fe x;
    secp256k1_fe y;
    int infinity; /* whether this represents the point at infinity */
} secp256k1_ge;

typedef struct {

    /* Blinding values used when computing (n-b)G + bG. */
    secp256k1_scalar blind; /* -b */
    secp256k1_gej initial;  /* bG */

    /* Whether the context has been built. */
    int built;    
} secp256k1_ecmult_gen_context;

typedef struct {
    secp256k1_fe_storage x;
    secp256k1_fe_storage y;
} secp256k1_ge_storage;


#define SECP256K1_GE_STORAGE_CONST(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) {SECP256K1_FE_STORAGE_CONST((a),(b),(c),(d),(e),(f),(g),(h)), SECP256K1_FE_STORAGE_CONST((i),(j),(k),(l),(m),(n),(o),(p))}




#  define secp256k1_fe_normalize secp256k1_fe_impl_normalize
#  define secp256k1_fe_normalize_weak secp256k1_fe_impl_normalize_weak
#  define secp256k1_fe_normalize_var secp256k1_fe_impl_normalize_var
#  define secp256k1_fe_normalizes_to_zero secp256k1_fe_impl_normalizes_to_zero
#  define secp256k1_fe_normalizes_to_zero_var secp256k1_fe_impl_normalizes_to_zero_var
#  define secp256k1_fe_set_int secp256k1_fe_impl_set_int
#  define secp256k1_fe_clear secp256k1_fe_impl_clear
#  define secp256k1_fe_is_zero secp256k1_fe_impl_is_zero
#  define secp256k1_fe_is_odd secp256k1_fe_impl_is_odd
#  define secp256k1_fe_cmp_var secp256k1_fe_impl_cmp_var
#  define secp256k1_fe_set_b32_mod secp256k1_fe_impl_set_b32_mod
#  define secp256k1_fe_set_b32_limit secp256k1_fe_impl_set_b32_limit
#  define secp256k1_fe_get_b32 secp256k1_fe_impl_get_b32
#  define secp256k1_fe_negate_unchecked secp256k1_fe_impl_negate_unchecked
#  define secp256k1_fe_mul_int_unchecked secp256k1_fe_impl_mul_int_unchecked
#  define secp256k1_fe_add secp256k1_fe_impl_add
#  define secp256k1_fe_mul secp256k1_fe_impl_mul
#  define secp256k1_fe_sqr secp256k1_fe_impl_sqr
#  define secp256k1_fe_cmov secp256k1_fe_impl_cmov
#  define secp256k1_fe_to_storage secp256k1_fe_impl_to_storage
#  define secp256k1_fe_from_storage secp256k1_fe_impl_from_storage
#  define secp256k1_fe_inv secp256k1_fe_impl_inv
#  define secp256k1_fe_inv_var secp256k1_fe_impl_inv_var
#  define secp256k1_fe_get_bounds secp256k1_fe_impl_get_bounds
#  define secp256k1_fe_half secp256k1_fe_impl_half
#  define secp256k1_fe_add_int secp256k1_fe_impl_add_int
#  define secp256k1_fe_is_square_var secp256k1_fe_impl_is_square_var




/** Maximum allowed magnitudes for group element coordinates
 *  in affine (x, y) and jacobian (x, y, z) representation. */
#define SECP256K1_GE_X_MAGNITUDE_MAX  4
#define SECP256K1_GE_Y_MAGNITUDE_MAX  3
#define SECP256K1_GEJ_X_MAGNITUDE_MAX 4
#define SECP256K1_GEJ_Y_MAGNITUDE_MAX 4
#define SECP256K1_GEJ_Z_MAGNITUDE_MAX 1


/** This expands to an initializer for a secp256k1_fe valued sum((i*32) * d_i, i=0..7) mod p.
 *
 * It has magnitude 1, unless d_i are all 0, in which case the magnitude is 0.
 * It is normalized, unless sum(2^(i*32) * d_i, i=0..7) >= p.
 *
 * SECP256K1_FE_CONST_INNER is provided by the implementation.
 */
#define SECP256K1_FE_CONST(d7, d6, d5, d4, d3, d2, d1, d0) {SECP256K1_FE_CONST_INNER((d7), (d6), (d5), (d4), (d3), (d2), (d1), (d0))}

