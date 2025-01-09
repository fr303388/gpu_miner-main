// Copyright (c) 2024 The BitcoinPoW Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstring>  // For memcpy

#include "secp256k1.h"

/*
 * sha256.cu Implementation of SHA256 Hashing    
 *
 * Date: 12 June 2019
 * Revision: 1
 * *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

 
/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

const int HDR_DEPTH = 256;
const int d_utxo_set_idx4host_BYTES = 4;
const int STAKE_MODIFIER_BYTES = 32;
const int WALLET_UTXOS_HASH_BYTES = 32; // Will be more than 1 million    
const int WALLET_UTXOS_N_BYTES = 4; // Will be more than 1 million   
const int WALLET_UTXOS_TIME_FROM_BYTES = 4; // Will be more than 1 million  
const int START_TIME_BYTES = 4;
const int HASH_MERKLE_ROOT_BYTES = 32; 
const int HASH_PREV_BLOCK_BYTES = 32; 
const int N_BITS_BYTES = 4; 
const int N_TIME_BYTES = 4; 
const int PREV_STAKE_HASH_BYTES = 32; 
const int PREV_STAKE_N_BYTES = 4; 
const int BLOCK_SIG_BYTES = 80; // 1st byte is length   either 78 or 79, then normal vchsig(starts with 0x30 then total_len-2   then 8 nonce bytes)  use 80 for nice round number


// LARGE array holding the wallet info of hash and n
const int WALLET_UTXOS_LENGTH = 2000000; // Will be more than 1 million



/**************************** DATA TYPES ****************************/


typedef struct {
    uint64_t align1;
    uint8_t h_utxos_hash[WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH];
    uint64_t align2;
    uint8_t h_utxos_n[WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH];
    uint64_t align3;
    uint8_t h_utxos_block_from_time[WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH];
    uint64_t align12;
    uint8_t h_start_time[START_TIME_BYTES];
    uint64_t align11;
    uint8_t h_stake_modifier[STAKE_MODIFIER_BYTES];    
    uint64_t align4;
    uint8_t h_hash_merkle_root[HASH_MERKLE_ROOT_BYTES*HDR_DEPTH];
    uint64_t align5;
    volatile uint8_t h_hash_prev_block[HASH_PREV_BLOCK_BYTES*HDR_DEPTH];
    uint64_t align6;
    uint8_t h_n_bits[N_BITS_BYTES*HDR_DEPTH];
    uint64_t align7;
    uint8_t h_n_time[N_TIME_BYTES*HDR_DEPTH];
    uint64_t align8;
    uint8_t h_prev_stake_hash[PREV_STAKE_HASH_BYTES*HDR_DEPTH];
    uint64_t align9;
    uint8_t h_prev_stake_n[PREV_STAKE_N_BYTES*HDR_DEPTH];
    uint64_t align10;
    uint8_t h_block_sig[BLOCK_SIG_BYTES*HDR_DEPTH];
} STAGE1_S;








typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} CUDA_SHA256_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
__constant__ WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};



__constant__ secp256k1_fe secp256k1_fe_one = SECP256K1_FE_CONST(0, 0, 0, 0, 0, 0, 0, 1);

__constant__ secp256k1_fe secp256k1_const_beta = SECP256K1_FE_CONST(
    0x7ae96a2bul, 0x657c0710ul, 0x6e64479eul, 0xac3434e9ul,
    0x9cf04975ul, 0x12f58995ul, 0xc1396c28ul, 0x719501eeul
);



/*********************** FUNCTION DEFINITIONS ***********************/
__device__  __forceinline__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			cuda_sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}


__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[]) //, WORD nonce[])
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		cuda_sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	cuda_sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;      
	}

}



#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <string.h>
#include <unistd.h>

#define SHM_KEY 12345          // Key for shared memory
#define SEM_KEY 56789          // Key for semaphore set
#define SHM_SIZE 1024          // Size of shared memory

// Semaphore operations
#define WAIT 0
#define SIGNAL 1

// Union for semaphore operations
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
};

#include <iomanip>
#include <iostream>

 __device__ unsigned int secp256k1_scalar_get_bits(const secp256k1_scalar *a, unsigned int offset, unsigned int count) {
    return (a->d[offset >> 6] >> (offset & 0x3F)) & ((((uint64_t)1) << count) - 1);
}



__device__ int secp256k1_scalar_check_overflow(const secp256k1_scalar *a) {

    if (a->d[3] < SECP256K1_N_3) { return 0; }
    if (a->d[2] < SECP256K1_N_2) { return 0; }
    if (a->d[2] > SECP256K1_N_2) { return 1; }
    if (a->d[1] < SECP256K1_N_1) { return 0; }
    if (a->d[1] > SECP256K1_N_1) { return 1; }
    return (a->d[0] >= SECP256K1_N_0);
}



__device__ int secp256k1_scalar_reduce(secp256k1_scalar *r, unsigned int overflow) {
    uint128_t t;

    if ( !overflow ) { return overflow; }

    t  = (uint128_t)r->d[0] + SECP256K1_N_C_0; r->d[0] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)r->d[1] + SECP256K1_N_C_1; r->d[1] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)r->d[2] + SECP256K1_N_C_2; r->d[2] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t +=  (uint64_t)r->d[3]                  ; r->d[3] = t & 0xFFFFFFFFFFFFFFFFULL;
    return overflow;
}



__device__ int secp256k1_scalar_add(secp256k1_scalar *r, const secp256k1_scalar *a, const secp256k1_scalar *b) {
    uint128_t t;
    t  = (uint128_t)a->d[0] + b->d[0]; r->d[0] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)a->d[1] + b->d[1]; r->d[1] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)a->d[2] + b->d[2]; r->d[2] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)a->d[3] + b->d[3]; r->d[3] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;

    if ( t > 0 ) { return secp256k1_scalar_reduce(r, 1); }
    if ( secp256k1_scalar_check_overflow(r) ) { return secp256k1_scalar_reduce(r, 1); }
    return 0;

}


__device__ void secp256k1_fe_storage_cmov(secp256k1_fe_storage *r, const secp256k1_fe_storage *a, int flag) {
    uint64_t mask0, mask1;
    volatile int vflag = flag;
    // SECP256K1_CHECKMEM_CHECK_VERIFY(r->n, sizeof(r->n));
    mask0 = vflag + ~((uint64_t)0);
    mask1 = ~mask0;
    r->n[0] = (r->n[0] & mask0) | (a->n[0] & mask1);
    r->n[1] = (r->n[1] & mask0) | (a->n[1] & mask1);
    r->n[2] = (r->n[2] & mask0) | (a->n[2] & mask1);
    r->n[3] = (r->n[3] & mask0) | (a->n[3] & mask1);
}

__device__  void secp256k1_ge_storage_cmov(secp256k1_ge_storage *r, const secp256k1_ge_storage *a, int flag) {
    secp256k1_fe_storage_cmov(&r->x, &a->x, flag);
    secp256k1_fe_storage_cmov(&r->y, &a->y, flag);
}

__device__  void secp256k1_fe_impl_from_storage(secp256k1_fe *r, const secp256k1_fe_storage *a) {
    r->n[0] = a->n[0] & 0xFFFFFFFFFFFFFULL;
    r->n[1] = a->n[0] >> 52 | ((a->n[1] << 12) & 0xFFFFFFFFFFFFFULL);
    r->n[2] = a->n[1] >> 40 | ((a->n[2] << 24) & 0xFFFFFFFFFFFFFULL);
    r->n[3] = a->n[2] >> 28 | ((a->n[3] << 36) & 0xFFFFFFFFFFFFFULL);
    r->n[4] = a->n[3] >> 16;
}

__device__ void secp256k1_ge_from_storage(secp256k1_ge *r, const secp256k1_ge_storage *a) {
    secp256k1_fe_from_storage(&r->x, &a->x);
    secp256k1_fe_from_storage(&r->y, &a->y);
    r->infinity = 0;
}



///////////////////////////////////////


__device__ void secp256k1_u128_load(secp256k1_uint128 *r, uint64_t hi, uint64_t lo) {
    *r = (((uint128_t)hi) << 64) + lo;
}

__device__ void secp256k1_u128_mul(secp256k1_uint128 *r, uint64_t a, uint64_t b) {
   *r = (uint128_t)a * b;
}

__device__ void secp256k1_u128_accum_mul(secp256k1_uint128 *r, uint64_t a, uint64_t b) {
   *r += (uint128_t)a * b;
}

__device__ void secp256k1_u128_accum_u64(secp256k1_uint128 *r, uint64_t a) {
   *r += a;
}

__device__ void secp256k1_u128_rshift(secp256k1_uint128 *r, unsigned int n) {
   
   *r >>= n;
}

__device__ uint64_t secp256k1_u128_to_u64(const secp256k1_uint128 *a) {
   return (uint64_t)(*a);
}

__device__ uint64_t secp256k1_u128_hi_u64(const secp256k1_uint128 *a) {
   return (uint64_t)(*a >> 64);
}

__device__ void secp256k1_u128_from_u64(secp256k1_uint128 *r, uint64_t a) {
   *r = a;
}

__device__ int secp256k1_u128_check_bits(const secp256k1_uint128 *r, unsigned int n) {
   
   return (*r >> n == 0);
}

__device__ void secp256k1_i128_load(secp256k1_int128 *r, int64_t hi, uint64_t lo) {
    *r = (((uint128_t)(uint64_t)hi) << 64) + lo;
}

__device__ void secp256k1_i128_mul(secp256k1_int128 *r, int64_t a, int64_t b) {
   *r = (int128_t)a * b;
}

__device__ void secp256k1_i128_accum_mul(secp256k1_int128 *r, int64_t a, int64_t b) {
   int128_t ab = (int128_t)a * b;
   
   *r += ab;
}

__device__ void secp256k1_i128_det(secp256k1_int128 *r, int64_t a, int64_t b, int64_t c, int64_t d) {
   int128_t ad = (int128_t)a * d;
   int128_t bc = (int128_t)b * c;
   
   *r = ad - bc;
}

__device__ void secp256k1_i128_rshift(secp256k1_int128 *r, unsigned int n) {
   
   *r >>= n;
}

__device__ uint64_t secp256k1_i128_to_u64(const secp256k1_int128 *a) {
   return (uint64_t)*a;
}

__device__ int64_t secp256k1_i128_to_i64(const secp256k1_int128 *a) {
   
   return *a;
}

__device__ void secp256k1_i128_from_i64(secp256k1_int128 *r, int64_t a) {
   *r = a;
}

__device__ int secp256k1_i128_eq_var(const secp256k1_int128 *a, const secp256k1_int128 *b) {
   return *a == *b;
}

__device__ int secp256k1_i128_check_pow2(const secp256k1_int128 *r, unsigned int n, int sign) {
   
   
   return (*r == (int128_t)((uint128_t)sign << n));
}



/////////////////////////////////////





__device__  void secp256k1_fe_sqr_inner(uint64_t *r, const uint64_t *a) {
    secp256k1_uint128 c, d;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    int64_t t3, t4, tx, u0;
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;


    /**  [... a b c] is a shorthand for ... + a<<104 + b<<52 + c<<0 mod n.
     *  px is a shorthand for sum(a[i]*a[x-i], i=0..x).
     *  Note that [x 0 0 0 0 0] = [x*R].
     */

    secp256k1_u128_mul(&d, a0*2, a3);
    secp256k1_u128_accum_mul(&d, a1*2, a2);
    
    /* [d 0 0 0] = [p3 0 0 0] */
    secp256k1_u128_mul(&c, a4, a4);
    
    /* [c 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    secp256k1_u128_accum_mul(&d, R, secp256k1_u128_to_u64(&c)); secp256k1_u128_rshift(&c, 64);
    
    
    /* [(c<<12) 0 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    t3 = secp256k1_u128_to_u64(&d) & M; secp256k1_u128_rshift(&d, 52);
    
    
    /* [(c<<12) 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */

    a4 *= 2;
    secp256k1_u128_accum_mul(&d, a0, a4);
    secp256k1_u128_accum_mul(&d, a1*2, a3);
    secp256k1_u128_accum_mul(&d, a2, a2);
    
    /* [(c<<12) 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    secp256k1_u128_accum_mul(&d, R << 12, secp256k1_u128_to_u64(&c));
    
    /* [d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    t4 = secp256k1_u128_to_u64(&d) & M; secp256k1_u128_rshift(&d, 52);
    
    
    /* [d t4 t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    tx = (t4 >> 48); t4 &= (M >> 4);
    
    
    /* [d t4+(tx<<48) t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */

    secp256k1_u128_mul(&c, a0, a0);
    
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 0 p4 p3 0 0 p0] */
    secp256k1_u128_accum_mul(&d, a1, a4);
    secp256k1_u128_accum_mul(&d, a2*2, a3);
    
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = secp256k1_u128_to_u64(&d) & M; secp256k1_u128_rshift(&d, 52);
    
    
    /* [d u0 t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    /* [d 0 t4+(tx<<48)+(u0<<52) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = (u0 << 4) | tx;
    
    /* [d 0 t4+(u0<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    secp256k1_u128_accum_mul(&c, u0, R >> 4);
    
    /* [d 0 t4 t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    r[0] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 0 p0] */

    a0 *= 2;
    secp256k1_u128_accum_mul(&c, a0, a1);
    
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 p1 p0] */
    secp256k1_u128_accum_mul(&d, a2, a4);
    secp256k1_u128_accum_mul(&d, a3, a3);
    
    /* [d 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    secp256k1_u128_accum_mul(&c, secp256k1_u128_to_u64(&d) & M, R); secp256k1_u128_rshift(&d, 52);
    
    
    /* [d 0 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    r[1] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */

    secp256k1_u128_accum_mul(&c, a0, a2);
    secp256k1_u128_accum_mul(&c, a1, a1);
    
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 p2 p1 p0] */
    secp256k1_u128_accum_mul(&d, a3, a4);
    
    /* [d 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    secp256k1_u128_accum_mul(&c, R, secp256k1_u128_to_u64(&d)); secp256k1_u128_rshift(&d, 64);
    
    
    /* [(d<<12) 0 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[2] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [(d<<12) 0 0 0 t4 t3+c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */

    secp256k1_u128_accum_mul(&c, R << 12, secp256k1_u128_to_u64(&d));
    secp256k1_u128_accum_u64(&c, t3);
    
    /* [t4 c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[3] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [t4+c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[4] = secp256k1_u128_to_u64(&c) + t4;
    
    /* [r4 r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
}




__device__ void secp256k1_fe_impl_sqr(secp256k1_fe *r, const secp256k1_fe *a) {
    secp256k1_fe_sqr_inner(r->n, a->n);
}



__device__ void secp256k1_fe_impl_get_bounds(secp256k1_fe *r, int m) {
    r->n[0] = 0xFFFFFFFFFFFFFULL * 2 * m;
    r->n[1] = 0xFFFFFFFFFFFFFULL * 2 * m;
    r->n[2] = 0xFFFFFFFFFFFFFULL * 2 * m;
    r->n[3] = 0xFFFFFFFFFFFFFULL * 2 * m;
    r->n[4] = 0x0FFFFFFFFFFFFULL * 2 * m;
}

__device__ void secp256k1_fe_impl_normalize(secp256k1_fe *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];

    /* Reduce t4 at the start so there will be at most a single carry from the first pass */
    uint64_t m;
    uint64_t x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFFULL;

    /* The first pass ensures the magnitude is 1, ... */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL; m = t1;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL; m &= t2;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL; m &= t3;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */
    

    /* At most a single final reduction is needed; check if the value is >= the field characteristic */
    x = (t4 >> 48) | ((t4 == 0x0FFFFFFFFFFFFULL) & (m == 0xFFFFFFFFFFFFFULL)
        & (t0 >= 0xFFFFEFFFFFC2FULL));

    /* Apply the final reduction (for constant-time behaviour, we do it always) */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL;

    /* If t4 didn't carry to bit 48 already, then it should have after any final reduction */
    

    /* Mask off the possible multiple of 2^256 from the final reduction */
    t4 &= 0x0FFFFFFFFFFFFULL;

    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}

__device__ void secp256k1_fe_impl_normalize_weak(secp256k1_fe *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];

    /* Reduce t4 at the start so there will be at most a single carry from the first pass */
    uint64_t x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFFULL;

    /* The first pass ensures the magnitude is 1, ... */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */
    

    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}

__device__ void secp256k1_fe_impl_normalize_var(secp256k1_fe *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];

    /* Reduce t4 at the start so there will be at most a single carry from the first pass */
    uint64_t m;
    uint64_t x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFFULL;

    /* The first pass ensures the magnitude is 1, ... */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL; m = t1;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL; m &= t2;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL; m &= t3;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */
    

    /* At most a single final reduction is needed; check if the value is >= the field characteristic */
    x = (t4 >> 48) | ((t4 == 0x0FFFFFFFFFFFFULL) & (m == 0xFFFFFFFFFFFFFULL)
        & (t0 >= 0xFFFFEFFFFFC2FULL));

    if (x) {
        t0 += 0x1000003D1ULL;
        t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL;
        t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL;
        t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL;
        t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL;

        /* If t4 didn't carry to bit 48 already, then it should have after any final reduction */
        

        /* Mask off the possible multiple of 2^256 from the final reduction */
        t4 &= 0x0FFFFFFFFFFFFULL;
    }

    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}

__device__ int secp256k1_fe_impl_normalizes_to_zero(const secp256k1_fe *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];

    /* z0 tracks a possible raw value of 0, z1 tracks a possible raw value of P */
    uint64_t z0, z1;

    /* Reduce t4 at the start so there will be at most a single carry from the first pass */
    uint64_t x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFFULL;

    /* The first pass ensures the magnitude is 1, ... */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= 0xFFFFFFFFFFFFFULL; z0  = t0; z1  = t0 ^ 0x1000003D0ULL;
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL; z0 |= t1; z1 &= t1;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL; z0 |= t2; z1 &= t2;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL; z0 |= t3; z1 &= t3;
                                                z0 |= t4; z1 &= t4 ^ 0xF000000000000ULL;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */
    

    return (z0 == 0) | (z1 == 0xFFFFFFFFFFFFFULL);
}

__device__ int secp256k1_fe_impl_normalizes_to_zero_var(const secp256k1_fe *r) {
    uint64_t t0, t1, t2, t3, t4;
    uint64_t z0, z1;
    uint64_t x;

    t0 = r->n[0];
    t4 = r->n[4];

    /* Reduce t4 at the start so there will be at most a single carry from the first pass */
    x = t4 >> 48;

    /* The first pass ensures the magnitude is 1, ... */
    t0 += x * 0x1000003D1ULL;

    /* z0 tracks a possible raw value of 0, z1 tracks a possible raw value of P */
    z0 = t0 & 0xFFFFFFFFFFFFFULL;
    z1 = z0 ^ 0x1000003D0ULL;

    /* Fast return path should catch the majority of cases */
    if ((z0 != 0ULL) & (z1 != 0xFFFFFFFFFFFFFULL)) {
        return 0;
    }

    t1 = r->n[1];
    t2 = r->n[2];
    t3 = r->n[3];

    t4 &= 0x0FFFFFFFFFFFFULL;

    t1 += (t0 >> 52);
    t2 += (t1 >> 52); t1 &= 0xFFFFFFFFFFFFFULL; z0 |= t1; z1 &= t1;
    t3 += (t2 >> 52); t2 &= 0xFFFFFFFFFFFFFULL; z0 |= t2; z1 &= t2;
    t4 += (t3 >> 52); t3 &= 0xFFFFFFFFFFFFFULL; z0 |= t3; z1 &= t3;
                                                z0 |= t4; z1 &= t4 ^ 0xF000000000000ULL;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */
    

    return (z0 == 0) | (z1 == 0xFFFFFFFFFFFFFULL);
}

__device__ void secp256k1_fe_impl_set_int(secp256k1_fe *r, int a) {
    r->n[0] = a;
    r->n[1] = r->n[2] = r->n[3] = r->n[4] = 0;
}

__device__ int secp256k1_fe_impl_is_zero(const secp256k1_fe *a) {
    const uint64_t *t = a->n;
    return (t[0] | t[1] | t[2] | t[3] | t[4]) == 0;
}

__device__ int secp256k1_fe_impl_is_odd(const secp256k1_fe *a) {
    return a->n[0] & 1;
}

__device__ void secp256k1_fe_impl_clear(secp256k1_fe *a) {
    int i;
    for (i=0; i<5; i++) {
        a->n[i] = 0;
    }
}

__device__ int secp256k1_fe_impl_cmp_var(const secp256k1_fe *a, const secp256k1_fe *b) {
    int i;
    for (i = 4; i >= 0; i--) {
        if (a->n[i] > b->n[i]) {
            return 1;
        }
        if (a->n[i] < b->n[i]) {
            return -1;
        }
    }
    return 0;
}

__device__ void secp256k1_fe_impl_set_b32_mod(secp256k1_fe *r, const unsigned char *a) {
    r->n[0] = (uint64_t)a[31]
            | ((uint64_t)a[30] << 8)
            | ((uint64_t)a[29] << 16)
            | ((uint64_t)a[28] << 24)
            | ((uint64_t)a[27] << 32)
            | ((uint64_t)a[26] << 40)
            | ((uint64_t)(a[25] & 0xF)  << 48);
    r->n[1] = (uint64_t)((a[25] >> 4) & 0xF)
            | ((uint64_t)a[24] << 4)
            | ((uint64_t)a[23] << 12)
            | ((uint64_t)a[22] << 20)
            | ((uint64_t)a[21] << 28)
            | ((uint64_t)a[20] << 36)
            | ((uint64_t)a[19] << 44);
    r->n[2] = (uint64_t)a[18]
            | ((uint64_t)a[17] << 8)
            | ((uint64_t)a[16] << 16)
            | ((uint64_t)a[15] << 24)
            | ((uint64_t)a[14] << 32)
            | ((uint64_t)a[13] << 40)
            | ((uint64_t)(a[12] & 0xF) << 48);
    r->n[3] = (uint64_t)((a[12] >> 4) & 0xF)
            | ((uint64_t)a[11] << 4)
            | ((uint64_t)a[10] << 12)
            | ((uint64_t)a[9]  << 20)
            | ((uint64_t)a[8]  << 28)
            | ((uint64_t)a[7]  << 36)
            | ((uint64_t)a[6]  << 44);
    r->n[4] = (uint64_t)a[5]
            | ((uint64_t)a[4] << 8)
            | ((uint64_t)a[3] << 16)
            | ((uint64_t)a[2] << 24)
            | ((uint64_t)a[1] << 32)
            | ((uint64_t)a[0] << 40);
}

__device__ int secp256k1_fe_impl_set_b32_limit(secp256k1_fe *r, const unsigned char *a) {
    secp256k1_fe_impl_set_b32_mod(r, a);
    return !((r->n[4] == 0x0FFFFFFFFFFFFULL) & ((r->n[3] & r->n[2] & r->n[1]) == 0xFFFFFFFFFFFFFULL) & (r->n[0] >= 0xFFFFEFFFFFC2FULL));
}

/** Convert a field element to a 32-byte big endian value. Requires the input to be normalized */
__device__ void secp256k1_fe_impl_get_b32(unsigned char *r, const secp256k1_fe *a) {
    r[0] = (a->n[4] >> 40) & 0xFF;
    r[1] = (a->n[4] >> 32) & 0xFF;
    r[2] = (a->n[4] >> 24) & 0xFF;
    r[3] = (a->n[4] >> 16) & 0xFF;
    r[4] = (a->n[4] >> 8) & 0xFF;
    r[5] = a->n[4] & 0xFF;
    r[6] = (a->n[3] >> 44) & 0xFF;
    r[7] = (a->n[3] >> 36) & 0xFF;
    r[8] = (a->n[3] >> 28) & 0xFF;
    r[9] = (a->n[3] >> 20) & 0xFF;
    r[10] = (a->n[3] >> 12) & 0xFF;
    r[11] = (a->n[3] >> 4) & 0xFF;
    r[12] = ((a->n[2] >> 48) & 0xF) | ((a->n[3] & 0xF) << 4);
    r[13] = (a->n[2] >> 40) & 0xFF;
    r[14] = (a->n[2] >> 32) & 0xFF;
    r[15] = (a->n[2] >> 24) & 0xFF;
    r[16] = (a->n[2] >> 16) & 0xFF;
    r[17] = (a->n[2] >> 8) & 0xFF;
    r[18] = a->n[2] & 0xFF;
    r[19] = (a->n[1] >> 44) & 0xFF;
    r[20] = (a->n[1] >> 36) & 0xFF;
    r[21] = (a->n[1] >> 28) & 0xFF;
    r[22] = (a->n[1] >> 20) & 0xFF;
    r[23] = (a->n[1] >> 12) & 0xFF;
    r[24] = (a->n[1] >> 4) & 0xFF;
    r[25] = ((a->n[0] >> 48) & 0xF) | ((a->n[1] & 0xF) << 4);
    r[26] = (a->n[0] >> 40) & 0xFF;
    r[27] = (a->n[0] >> 32) & 0xFF;
    r[28] = (a->n[0] >> 24) & 0xFF;
    r[29] = (a->n[0] >> 16) & 0xFF;
    r[30] = (a->n[0] >> 8) & 0xFF;
    r[31] = a->n[0] & 0xFF;
}

__device__ void secp256k1_fe_impl_negate_unchecked(secp256k1_fe *r, const secp256k1_fe *a, int m) {
    /* For all legal values of m (0..31), the following properties hold: */
    
    
    

    /* Due to the properties above, the left hand in the subtractions below is never less than
     * the right hand. */
    r->n[0] = 0xFFFFEFFFFFC2FULL * 2 * (m + 1) - a->n[0];
    r->n[1] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[1];
    r->n[2] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[2];
    r->n[3] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[3];
    r->n[4] = 0x0FFFFFFFFFFFFULL * 2 * (m + 1) - a->n[4];
}

__device__ void secp256k1_fe_impl_mul_int_unchecked(secp256k1_fe *r, int a) {
    r->n[0] *= a;
    r->n[1] *= a;
    r->n[2] *= a;
    r->n[3] *= a;
    r->n[4] *= a;
}

__device__ void secp256k1_fe_impl_add_int(secp256k1_fe *r, int a) {
    r->n[0] += a;
}

__device__ void secp256k1_fe_impl_add(secp256k1_fe *r, const secp256k1_fe *a) {
    r->n[0] += a->n[0];
    r->n[1] += a->n[1];
    r->n[2] += a->n[2];
    r->n[3] += a->n[3];
    r->n[4] += a->n[4];
}







__device__ void secp256k1_fe_mul_inner(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    secp256k1_uint128 c, d;
    uint64_t t3, t4, tx, u0;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;


    /*  [... a b c] is a shorthand for ... + a<<104 + b<<52 + c<<0 mod n.
     *  for 0 <= x <= 4, px is a shorthand for sum(a[i]*b[x-i], i=0..x).
     *  for 4 <= x <= 8, px is a shorthand for sum(a[i]*b[x-i], i=(x-4)..4)
     *  Note that [x 0 0 0 0 0] = [x*R].
     */

    secp256k1_u128_mul(&d, a0, b[3]);
    secp256k1_u128_accum_mul(&d, a1, b[2]);
    secp256k1_u128_accum_mul(&d, a2, b[1]);
    secp256k1_u128_accum_mul(&d, a3, b[0]);
    
    /* [d 0 0 0] = [p3 0 0 0] */
    secp256k1_u128_mul(&c, a4, b[4]);
    
    /* [c 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    secp256k1_u128_accum_mul(&d, R, secp256k1_u128_to_u64(&c)); secp256k1_u128_rshift(&c, 64);
    
    
    /* [(c<<12) 0 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    t3 = secp256k1_u128_to_u64(&d) & M; secp256k1_u128_rshift(&d, 52);
    
    
    /* [(c<<12) 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */

    secp256k1_u128_accum_mul(&d, a0, b[4]);
    secp256k1_u128_accum_mul(&d, a1, b[3]);
    secp256k1_u128_accum_mul(&d, a2, b[2]);
    secp256k1_u128_accum_mul(&d, a3, b[1]);
    secp256k1_u128_accum_mul(&d, a4, b[0]);
    
    /* [(c<<12) 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    secp256k1_u128_accum_mul(&d, R << 12, secp256k1_u128_to_u64(&c));
    
    /* [d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    t4 = secp256k1_u128_to_u64(&d) & M; secp256k1_u128_rshift(&d, 52);
    
    
    /* [d t4 t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    tx = (t4 >> 48); t4 &= (M >> 4);
    
    
    /* [d t4+(tx<<48) t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */

    secp256k1_u128_mul(&c, a0, b[0]);
    
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 0 p4 p3 0 0 p0] */
    secp256k1_u128_accum_mul(&d, a1, b[4]);
    secp256k1_u128_accum_mul(&d, a2, b[3]);
    secp256k1_u128_accum_mul(&d, a3, b[2]);
    secp256k1_u128_accum_mul(&d, a4, b[1]);
    
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = secp256k1_u128_to_u64(&d) & M; secp256k1_u128_rshift(&d, 52);
    
    
    /* [d u0 t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    /* [d 0 t4+(tx<<48)+(u0<<52) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = (u0 << 4) | tx;
    
    /* [d 0 t4+(u0<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    secp256k1_u128_accum_mul(&c, u0, R >> 4);
    
    /* [d 0 t4 t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    r[0] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 0 p0] */

    secp256k1_u128_accum_mul(&c, a0, b[1]);
    secp256k1_u128_accum_mul(&c, a1, b[0]);
    
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 p1 p0] */
    secp256k1_u128_accum_mul(&d, a2, b[4]);
    secp256k1_u128_accum_mul(&d, a3, b[3]);
    secp256k1_u128_accum_mul(&d, a4, b[2]);
    
    /* [d 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    secp256k1_u128_accum_mul(&c, secp256k1_u128_to_u64(&d) & M, R); secp256k1_u128_rshift(&d, 52);
    
    
    /* [d 0 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    r[1] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */

    secp256k1_u128_accum_mul(&c, a0, b[2]);
    secp256k1_u128_accum_mul(&c, a1, b[1]);
    secp256k1_u128_accum_mul(&c, a2, b[0]);
    
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 p2 p1 p0] */
    secp256k1_u128_accum_mul(&d, a3, b[4]);
    secp256k1_u128_accum_mul(&d, a4, b[3]);
    
    /* [d 0 0 t4 t3 c t1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    secp256k1_u128_accum_mul(&c, R, secp256k1_u128_to_u64(&d)); secp256k1_u128_rshift(&d, 64);
    
    
    /* [(d<<12) 0 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */

    r[2] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [(d<<12) 0 0 0 t4 t3+c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    secp256k1_u128_accum_mul(&c, R << 12, secp256k1_u128_to_u64(&d));
    secp256k1_u128_accum_u64(&c, t3);
    
    /* [t4 c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[3] = secp256k1_u128_to_u64(&c) & M; secp256k1_u128_rshift(&c, 52);
    
    
    /* [t4+c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[4] = secp256k1_u128_to_u64(&c) + t4;
    
    /* [r4 r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
}






__device__  void secp256k1_fe_impl_mul(secp256k1_fe *r, const secp256k1_fe *a, const secp256k1_fe * b) {
    secp256k1_fe_mul_inner(r->n, a->n, b->n);
}

// __device__  void secp256k1_fe_impl_sqr(secp256k1_fe *r, const secp256k1_fe *a) {
//     secp256k1_fe_sqr_inner(r->n, a->n);
// }

__device__ void secp256k1_fe_impl_cmov(secp256k1_fe *r, const secp256k1_fe *a, int flag) {
    uint64_t mask0, mask1;
    volatile int vflag = flag;

    mask0 = vflag + ~((uint64_t)0);
    mask1 = ~mask0;
    r->n[0] = (r->n[0] & mask0) | (a->n[0] & mask1);
    r->n[1] = (r->n[1] & mask0) | (a->n[1] & mask1);
    r->n[2] = (r->n[2] & mask0) | (a->n[2] & mask1);
    r->n[3] = (r->n[3] & mask0) | (a->n[3] & mask1);
    r->n[4] = (r->n[4] & mask0) | (a->n[4] & mask1);
}

__device__ void secp256k1_fe_impl_half(secp256k1_fe *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];
    uint64_t one = (uint64_t)1;
    uint64_t mask = -(t0 & one) >> 12;

    /* Bounds analysis (over the rationals).
     *
     * Let m = r->magnitude
     *     C = 0xFFFFFFFFFFFFFULL * 2
     *     D = 0x0FFFFFFFFFFFFULL * 2
     *
     * Initial bounds: t0..t3 <= C * m
     *                     t4 <= D * m
     */

    t0 += 0xFFFFEFFFFFC2FULL & mask;
    t1 += mask;
    t2 += mask;
    t3 += mask;
    t4 += mask >> 4;

    

    /* t0..t3: added <= C/2
     *     t4: added <= D/2
     *
     * Current bounds: t0..t3 <= C * (m + 1/2)
     *                     t4 <= D * (m + 1/2)
     */

    r->n[0] = (t0 >> 1) + ((t1 & one) << 51);
    r->n[1] = (t1 >> 1) + ((t2 & one) << 51);
    r->n[2] = (t2 >> 1) + ((t3 & one) << 51);
    r->n[3] = (t3 >> 1) + ((t4 & one) << 51);
    r->n[4] = (t4 >> 1);

    /* t0..t3: shifted right and added <= C/4 + 1/2
     *     t4: shifted right
     *
     * Current bounds: t0..t3 <= C * (m/2 + 1/2)
     *                     t4 <= D * (m/2 + 1/4)
     *
     * Therefore the output magnitude (M) has to be set such that:
     *     t0..t3: C * M >= C * (m/2 + 1/2)
     *         t4: D * M >= D * (m/2 + 1/4)
     *
     * It suffices for all limbs that, for any input magnitude m:
     *     M >= m/2 + 1/2
     *
     * and since we want the smallest such integer value for M:
     *     M == floor(m/2) + 1
     */
}


__device__ void secp256k1_fe_impl_to_storage(secp256k1_fe_storage *r, const secp256k1_fe *a) {
    r->n[0] = a->n[0] | a->n[1] << 52;
    r->n[1] = a->n[1] >> 12 | a->n[2] << 40;
    r->n[2] = a->n[2] >> 24 | a->n[3] << 28;
    r->n[3] = a->n[3] >> 36 | a->n[4] << 16;
}




__device__ void secp256k1_gej_add_ge(secp256k1_gej *r, const secp256k1_gej *a, const secp256k1_ge *b) {
    /* Operations: 7 mul, 5 sqr, 21 add/cmov/half/mul_int/negate/normalizes_to_zero */
    secp256k1_fe zz, u1, u2, s1, s2, t, tt, m, n, q, rr;
    secp256k1_fe m_alt, rr_alt;
    int degenerate;


    /*  In:
     *    Eric Brier and Marc Joye, Weierstrass Elliptic Curves and Side-Channel Attacks.
     *    In D. Naccache and P. Paillier, Eds., Public Key Cryptography, vol. 2274 of Lecture Notes in Computer Science, pages 335-345. Springer-Verlag, 2002.
     *  we find as solution for a unified addition/doubling formula:
     *    lambda = ((x1 + x2)^2 - x1 * x2 + a) / (y1 + y2), with a = 0 for secp256k1's curve equation.
     *    x3 = lambda^2 - (x1 + x2)
     *    2*y3 = lambda * (x1 + x2 - 2 * x3) - (y1 + y2).
     *
     *  Substituting x_i = Xi / Zi^2 and yi = Yi / Zi^3, for i=1,2,3, gives:
     *    U1 = X1*Z2^2, U2 = X2*Z1^2
     *    S1 = Y1*Z2^3, S2 = Y2*Z1^3
     *    Z = Z1*Z2
     *    T = U1+U2
     *    M = S1+S2
     *    Q = -T*M^2
     *    R = T^2-U1*U2
     *    X3 = R^2+Q
     *    Y3 = -(R*(2*X3+Q)+M^4)/2
     *    Z3 = M*Z
     *  (Note that the paper uses xi = Xi / Zi and yi = Yi / Zi instead.)
     *
     *  This formula has the benefit of being the same for both addition
     *  of distinct points and doubling. However, it breaks down in the
     *  case that either point is infinity, or that y1 = -y2. We handle
     *  these cases in the following ways:
     *
     *    - If b is infinity we simply bail by means of a 
     *
     *    - If a is infinity, we detect this, and at the end of the
     *      computation replace the result (which will be meaningless,
     *      but we compute to be constant-time) with b.x : b.y : 1.
     *
     *    - If a = -b, we have y1 = -y2, which is a degenerate case.
     *      But here the answer is infinity, so we simply set the
     *      infinity flag of the result, overriding the computed values
     *      without even needing to cmov.
     *
     *    - If y1 = -y2 but x1 != x2, which does occur thanks to certain
     *      properties of our curve (specifically, 1 has nontrivial cube
     *      roots in our field, and the curve equation has no x coefficient)
     *      then the answer is not infinity but also not given by the above
     *      equation. In this case, we cmov in place an alternate expression
     *      for lambda. Specifically (y1 - y2)/(x1 - x2). Where both these
     *      expressions for lambda are defined, they are equal, and can be
     *      obtained from each other by multiplication by (y1 + y2)/(y1 + y2)
     *      then substitution of x^3 + 7 for y^2 (using the curve equation).
     *      For all pairs of nonzero points (a, b) at least one is defined,
     *      so this covers everything.
     */

    secp256k1_fe_sqr(&zz, &a->z);                       /* z = Z1^2 */
    u1 = a->x;                                          /* u1 = U1 = X1*Z2^2 (GEJ_X_M) */
    secp256k1_fe_mul(&u2, &b->x, &zz);                  /* u2 = U2 = X2*Z1^2 (1) */
    s1 = a->y;                                          /* s1 = S1 = Y1*Z2^3 (GEJ_Y_M) */
    secp256k1_fe_mul(&s2, &b->y, &zz);                  /* s2 = Y2*Z1^2 (1) */
    secp256k1_fe_mul(&s2, &s2, &a->z);                  /* s2 = S2 = Y2*Z1^3 (1) */
    t = u1; secp256k1_fe_add(&t, &u2);                  /* t = T = U1+U2 (GEJ_X_M+1) */
    m = s1; secp256k1_fe_add(&m, &s2);                  /* m = M = S1+S2 (GEJ_Y_M+1) */
    secp256k1_fe_sqr(&rr, &t);                          /* rr = T^2 (1) */
    secp256k1_fe_negate_unchecked(&m_alt, &u2, 1);                /* Malt = -X2*Z1^2 (2) */
    secp256k1_fe_mul(&tt, &u1, &m_alt);                 /* tt = -U1*U2 (1) */
    secp256k1_fe_add(&rr, &tt);                         /* rr = R = T^2-U1*U2 (2) */
    /* If lambda = R/M = R/0 we have a problem (except in the "trivial"
     * case that Z = z1z2 = 0, and this is special-cased later on). */
    degenerate = secp256k1_fe_normalizes_to_zero(&m);
    /* This only occurs when y1 == -y2 and x1^3 == x2^3, but x1 != x2.
     * This means either x1 == beta*x2 or beta*x1 == x2, where beta is
     * a nontrivial cube root of one. In either case, an alternate
     * non-indeterminate expression for lambda is (y1 - y2)/(x1 - x2),
     * so we set R/M equal to this. */
    rr_alt = s1;
    secp256k1_fe_mul_int_unchecked(&rr_alt, 2);       /* rr_alt = Y1*Z2^3 - Y2*Z1^3 (GEJ_Y_M*2) */
    secp256k1_fe_add(&m_alt, &u1);          /* Malt = X1*Z2^2 - X2*Z1^2 (GEJ_X_M+2) */

    secp256k1_fe_cmov(&rr_alt, &rr, !degenerate);       /* rr_alt (GEJ_Y_M*2) */
    secp256k1_fe_cmov(&m_alt, &m, !degenerate);         /* m_alt (GEJ_X_M+2) */
    /* Now Ralt / Malt = lambda and is guaranteed not to be Ralt / 0.
     * From here on out Ralt and Malt represent the numerator
     * and denominator of lambda; R and M represent the explicit
     * expressions x1^2 + x2^2 + x1x2 and y1 + y2. */
    secp256k1_fe_sqr(&n, &m_alt);                       /* n = Malt^2 (1) */
    secp256k1_fe_negate_unchecked(&q, &t,
        SECP256K1_GEJ_X_MAGNITUDE_MAX + 1);             /* q = -T (GEJ_X_M+2) */
    secp256k1_fe_mul(&q, &q, &n);                       /* q = Q = -T*Malt^2 (1) */
    /* These two lines use the observation that either M == Malt or M == 0,
     * so M^3 * Malt is either Malt^4 (which is computed by squaring), or
     * zero (which is "computed" by cmov). So the cost is one squaring
     * versus two multiplications. */
    secp256k1_fe_sqr(&n, &n);                           /* n = Malt^4 (1) */
    secp256k1_fe_cmov(&n, &m, degenerate);              /* n = M^3 * Malt (GEJ_Y_M+1) */
    secp256k1_fe_sqr(&t, &rr_alt);                      /* t = Ralt^2 (1) */
    secp256k1_fe_mul(&r->z, &a->z, &m_alt);             /* r->z = Z3 = Malt*Z (1) */
    secp256k1_fe_add(&t, &q);                           /* t = Ralt^2 + Q (2) */
    r->x = t;                                           /* r->x = X3 = Ralt^2 + Q (2) */
    secp256k1_fe_mul_int_unchecked(&t, 2);                        /* t = 2*X3 (4) */
    secp256k1_fe_add(&t, &q);                           /* t = 2*X3 + Q (5) */
    secp256k1_fe_mul(&t, &t, &rr_alt);                  /* t = Ralt*(2*X3 + Q) (1) */
    secp256k1_fe_add(&t, &n);                           /* t = Ralt*(2*X3 + Q) + M^3*Malt (GEJ_Y_M+2) */
    secp256k1_fe_negate_unchecked(&r->y, &t,
        SECP256K1_GEJ_Y_MAGNITUDE_MAX + 2);             /* r->y = -(Ralt*(2*X3 + Q) + M^3*Malt) (GEJ_Y_M+3) */
    secp256k1_fe_half(&r->y);                           /* r->y = Y3 = -(Ralt*(2*X3 + Q) + M^3*Malt)/2 ((GEJ_Y_M+3)/2 + 1) */

    /* In case a->infinity == 1, replace r with (b->x, b->y, 1). */
    secp256k1_fe_cmov(&r->x, &b->x, a->infinity);
    secp256k1_fe_cmov(&r->y, &b->y, a->infinity);
    secp256k1_fe_cmov(&r->z, &secp256k1_fe_one, a->infinity);

    /* Set r->infinity if r->z is 0.
     *
     * If a->infinity is set, then r->infinity = (r->z == 0) = (1 == 0) = false,
     * which is correct because the function assumes that b is not infinity.
     *
     * Now assume !a->infinity. This implies Z = Z1 != 0.
     *
     * Case y1 = -y2:
     * In this case we could have a = -b, namely if x1 = x2.
     * We have degenerate = true, r->z = (x1 - x2) * Z.
     * Then r->infinity = ((x1 - x2)Z == 0) = (x1 == x2) = (a == -b).
     *
     * Case y1 != -y2:
     * In this case, we can't have a = -b.
     * We have degenerate = false, r->z = (y1 + y2) * Z.
     * Then r->infinity = ((y1 + y2)Z == 0) = (y1 == -y2) = false. */
    r->infinity = secp256k1_fe_normalizes_to_zero(&r->z);


}




#include "table.cu"

//extern __constant__ secp256k1_ge_storage secp256k1_ecmult_gen_prec_table[128][4];

//__global__ void secp256k1_ecmult_gen(BYTE* indata, BYTE* outdata, WORD total_threads)
__device__ void secp256k1_ecmult_gen(const secp256k1_ecmult_gen_context *ctx, secp256k1_gej *r, const secp256k1_scalar *gn)
{

    int bits = 2;
    int g = 4;
    int n = 128;

    secp256k1_ge add;
    secp256k1_ge_storage adds;
    secp256k1_scalar gnb;
    int i, j, n_i;
    
    memset(&adds, 0, sizeof(adds));
    *r = ctx->initial;
    /* Blind scalar/point multiplication by computing (n-b)G + bG instead of nG. */
    secp256k1_scalar_add(&gnb, gn, &ctx->blind);
    add.infinity = 0;
    for (i = 0; i < n; i++) {
        n_i = secp256k1_scalar_get_bits(&gnb, i * bits, bits);
        for (j = 0; j < g; j++) {
            /** This uses a conditional move to avoid any secret data in array indexes.
             *   _Any_ use of secret indexes has been demonstrated to result in timing
             *   sidechannels, even when the cache-line access patterns are uniform.
             *  See also:
             *   "A word of warning", CHES 2013 Rump Session, by Daniel J. Bernstein and Peter Schwabe
             *    (https://cryptojedi.org/peter/data/chesrump-20130822.pdf) and
             *   "Cache Attacks and Countermeasures: the Case of AES", RSA 2006,
             *    by Dag Arne Osvik, Adi Shamir, and Eran Tromer
             *    (https://www.tau.ac.il/~tromer/papers/cache.pdf)
             */
            secp256k1_ge_storage_cmov(&adds, &secp256k1_ecmult_gen_prec_table[i][j], j == n_i);
        }
        secp256k1_ge_from_storage(&add, &adds);
        secp256k1_gej_add_ge(r, r, &add);
    }
    n_i = 0;
    // secp256k1_ge_clear(&add);
    // secp256k1_scalar_clear(&gnb);


    // copy the r to the output - only for testing
    //memcpy(outdata, r, 128);
}


/* A signed 62-bit limb representation of integers.
 *
 * Its value is sum(v[i] * 2^(62*i), i=0..4). */
typedef struct {
    int64_t v[5];
} secp256k1_modinv64_signed62;

typedef struct {
    /* The modulus in signed62 notation, must be odd and in [3, 2^256]. */
    secp256k1_modinv64_signed62 modulus;

    /* modulus^{-1} mod 2^62 */
    uint64_t modulus_inv62;
} secp256k1_modinv64_modinfo;


__device__ void secp256k1_fe_to_signed62(secp256k1_modinv64_signed62 *r, const secp256k1_fe *a) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const uint64_t a0 = a->n[0], a1 = a->n[1], a2 = a->n[2], a3 = a->n[3], a4 = a->n[4];

    r->v[0] = (a0       | a1 << 52) & M62;
    r->v[1] = (a1 >> 10 | a2 << 42) & M62;
    r->v[2] = (a2 >> 20 | a3 << 32) & M62;
    r->v[3] = (a3 >> 30 | a4 << 22) & M62;
    r->v[4] =  a4 >> 40;
}


typedef struct {
    int64_t u, v, q, r;
} secp256k1_modinv64_trans2x2;



__device__ int64_t secp256k1_modinv64_divsteps_59(int64_t zeta, uint64_t f0, uint64_t g0, secp256k1_modinv64_trans2x2 *t) {
    /* u,v,q,r are the elements of the transformation matrix being built up,
     * starting with the identity matrix times 8 (because the caller expects
     * a result scaled by 2^62). Semantically they are signed integers
     * in range [-2^62,2^62], but here represented as unsigned mod 2^64. This
     * permits left shifting (which is UB for negative numbers). The range
     * being inside [-2^63,2^63) means that casting to signed works correctly.
     */
    uint64_t u = 8, v = 0, q = 0, r = 8;
    volatile uint64_t c1, c2;
    uint64_t mask1, mask2, f = f0, g = g0, x, y, z;
    int i;

    for (i = 3; i < 62; ++i) {

        /* Compute conditional masks for (zeta < 0) and for (g & 1). */
        c1 = zeta >> 63;
        mask1 = c1;
        c2 = g & 1;
        mask2 = -c2;
        /* Compute x,y,z, conditionally negated versions of f,u,v. */
        x = (f ^ mask1) - mask1;
        y = (u ^ mask1) - mask1;
        z = (v ^ mask1) - mask1;
        /* Conditionally add x,y,z to g,q,r. */
        g += x & mask2;
        q += y & mask2;
        r += z & mask2;
        /* In what follows, c1 is a condition mask for (zeta < 0) and (g & 1). */
        mask1 &= mask2;
        /* Conditionally change zeta into -zeta-2 or zeta-1. */
        zeta = (zeta ^ mask1) - 1;
        /* Conditionally add g,q,r to f,u,v. */
        f += g & mask1;
        u += q & mask1;
        v += r & mask1;
        /* Shifts */
        g >>= 1;
        u <<= 1;
        v <<= 1;
        /* Bounds on zeta that follow from the bounds on iteration count (max 10*59 divsteps). */

    }
    /* Return data in t and return value. */
    t->u = (int64_t)u;
    t->v = (int64_t)v;
    t->q = (int64_t)q;
    t->r = (int64_t)r;

    return zeta;
}


__device__ void secp256k1_modinv64_update_de_62(secp256k1_modinv64_signed62 *d, secp256k1_modinv64_signed62 *e, const secp256k1_modinv64_trans2x2 *t, const secp256k1_modinv64_modinfo* modinfo) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t d0 = d->v[0], d1 = d->v[1], d2 = d->v[2], d3 = d->v[3], d4 = d->v[4];
    const int64_t e0 = e->v[0], e1 = e->v[1], e2 = e->v[2], e3 = e->v[3], e4 = e->v[4];
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
    int64_t md, me, sd, se;
    secp256k1_int128 cd, ce;

    /* [md,me] start as zero; plus [u,q] if d is negative; plus [v,r] if e is negative. */
    sd = d4 >> 63;
    se = e4 >> 63;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);
    /* Begin computing t*[d,e]. */
    secp256k1_i128_mul(&cd, u, d0);
    secp256k1_i128_accum_mul(&cd, v, e0);
    secp256k1_i128_mul(&ce, q, d0);
    secp256k1_i128_accum_mul(&ce, r, e0);
    /* Correct md,me so that t*[d,e]+modulus*[md,me] has 62 zero bottom bits. */
    md -= (modinfo->modulus_inv62 * secp256k1_i128_to_u64(&cd) + md) & M62;
    me -= (modinfo->modulus_inv62 * secp256k1_i128_to_u64(&ce) + me) & M62;
    /* Update the beginning of computation for t*[d,e]+modulus*[md,me] now md,me are known. */
    secp256k1_i128_accum_mul(&cd, modinfo->modulus.v[0], md);
    secp256k1_i128_accum_mul(&ce, modinfo->modulus.v[0], me);
    /* Verify that the low 62 bits of the computation are indeed zero, and then throw them away. */
    secp256k1_i128_rshift(&cd, 62);
    secp256k1_i128_rshift(&ce, 62);
    /* Compute limb 1 of t*[d,e]+modulus*[md,me], and store it as output limb 0 (= down shift). */
    secp256k1_i128_accum_mul(&cd, u, d1);
    secp256k1_i128_accum_mul(&cd, v, e1);
    secp256k1_i128_accum_mul(&ce, q, d1);
    secp256k1_i128_accum_mul(&ce, r, e1);
    if (modinfo->modulus.v[1]) { /* Optimize for the case where limb of modulus is zero. */
        secp256k1_i128_accum_mul(&cd, modinfo->modulus.v[1], md);
        secp256k1_i128_accum_mul(&ce, modinfo->modulus.v[1], me);
    }
    d->v[0] = secp256k1_i128_to_u64(&cd) & M62; secp256k1_i128_rshift(&cd, 62);
    e->v[0] = secp256k1_i128_to_u64(&ce) & M62; secp256k1_i128_rshift(&ce, 62);
    /* Compute limb 2 of t*[d,e]+modulus*[md,me], and store it as output limb 1. */
    secp256k1_i128_accum_mul(&cd, u, d2);
    secp256k1_i128_accum_mul(&cd, v, e2);
    secp256k1_i128_accum_mul(&ce, q, d2);
    secp256k1_i128_accum_mul(&ce, r, e2);
    if (modinfo->modulus.v[2]) { /* Optimize for the case where limb of modulus is zero. */
        secp256k1_i128_accum_mul(&cd, modinfo->modulus.v[2], md);
        secp256k1_i128_accum_mul(&ce, modinfo->modulus.v[2], me);
    }
    d->v[1] = secp256k1_i128_to_u64(&cd) & M62; secp256k1_i128_rshift(&cd, 62);
    e->v[1] = secp256k1_i128_to_u64(&ce) & M62; secp256k1_i128_rshift(&ce, 62);
    /* Compute limb 3 of t*[d,e]+modulus*[md,me], and store it as output limb 2. */
    secp256k1_i128_accum_mul(&cd, u, d3);
    secp256k1_i128_accum_mul(&cd, v, e3);
    secp256k1_i128_accum_mul(&ce, q, d3);
    secp256k1_i128_accum_mul(&ce, r, e3);
    if (modinfo->modulus.v[3]) { /* Optimize for the case where limb of modulus is zero. */
        secp256k1_i128_accum_mul(&cd, modinfo->modulus.v[3], md);
        secp256k1_i128_accum_mul(&ce, modinfo->modulus.v[3], me);
    }
    d->v[2] = secp256k1_i128_to_u64(&cd) & M62; secp256k1_i128_rshift(&cd, 62);
    e->v[2] = secp256k1_i128_to_u64(&ce) & M62; secp256k1_i128_rshift(&ce, 62);
    /* Compute limb 4 of t*[d,e]+modulus*[md,me], and store it as output limb 3. */
    secp256k1_i128_accum_mul(&cd, u, d4);
    secp256k1_i128_accum_mul(&cd, v, e4);
    secp256k1_i128_accum_mul(&ce, q, d4);
    secp256k1_i128_accum_mul(&ce, r, e4);
    secp256k1_i128_accum_mul(&cd, modinfo->modulus.v[4], md);
    secp256k1_i128_accum_mul(&ce, modinfo->modulus.v[4], me);
    d->v[3] = secp256k1_i128_to_u64(&cd) & M62; secp256k1_i128_rshift(&cd, 62);
    e->v[3] = secp256k1_i128_to_u64(&ce) & M62; secp256k1_i128_rshift(&ce, 62);
    /* What remains is limb 5 of t*[d,e]+modulus*[md,me]; store it as output limb 4. */
    d->v[4] = secp256k1_i128_to_i64(&cd);
    e->v[4] = secp256k1_i128_to_i64(&ce);

}



__device__ void secp256k1_modinv64_update_fg_62(secp256k1_modinv64_signed62 *f, secp256k1_modinv64_signed62 *g, const secp256k1_modinv64_trans2x2 *t) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t f0 = f->v[0], f1 = f->v[1], f2 = f->v[2], f3 = f->v[3], f4 = f->v[4];
    const int64_t g0 = g->v[0], g1 = g->v[1], g2 = g->v[2], g3 = g->v[3], g4 = g->v[4];
    const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
    secp256k1_int128 cf, cg;
    /* Start computing t*[f,g]. */
    secp256k1_i128_mul(&cf, u, f0);
    secp256k1_i128_accum_mul(&cf, v, g0);
    secp256k1_i128_mul(&cg, q, f0);
    secp256k1_i128_accum_mul(&cg, r, g0);
    /* Verify that the bottom 62 bits of the result are zero, and then throw them away. */
    secp256k1_i128_rshift(&cf, 62);
    secp256k1_i128_rshift(&cg, 62);
    /* Compute limb 1 of t*[f,g], and store it as output limb 0 (= down shift). */
    secp256k1_i128_accum_mul(&cf, u, f1);
    secp256k1_i128_accum_mul(&cf, v, g1);
    secp256k1_i128_accum_mul(&cg, q, f1);
    secp256k1_i128_accum_mul(&cg, r, g1);
    f->v[0] = secp256k1_i128_to_u64(&cf) & M62; secp256k1_i128_rshift(&cf, 62);
    g->v[0] = secp256k1_i128_to_u64(&cg) & M62; secp256k1_i128_rshift(&cg, 62);
    /* Compute limb 2 of t*[f,g], and store it as output limb 1. */
    secp256k1_i128_accum_mul(&cf, u, f2);
    secp256k1_i128_accum_mul(&cf, v, g2);
    secp256k1_i128_accum_mul(&cg, q, f2);
    secp256k1_i128_accum_mul(&cg, r, g2);
    f->v[1] = secp256k1_i128_to_u64(&cf) & M62; secp256k1_i128_rshift(&cf, 62);
    g->v[1] = secp256k1_i128_to_u64(&cg) & M62; secp256k1_i128_rshift(&cg, 62);
    /* Compute limb 3 of t*[f,g], and store it as output limb 2. */
    secp256k1_i128_accum_mul(&cf, u, f3);
    secp256k1_i128_accum_mul(&cf, v, g3);
    secp256k1_i128_accum_mul(&cg, q, f3);
    secp256k1_i128_accum_mul(&cg, r, g3);
    f->v[2] = secp256k1_i128_to_u64(&cf) & M62; secp256k1_i128_rshift(&cf, 62);
    g->v[2] = secp256k1_i128_to_u64(&cg) & M62; secp256k1_i128_rshift(&cg, 62);
    /* Compute limb 4 of t*[f,g], and store it as output limb 3. */
    secp256k1_i128_accum_mul(&cf, u, f4);
    secp256k1_i128_accum_mul(&cf, v, g4);
    secp256k1_i128_accum_mul(&cg, q, f4);
    secp256k1_i128_accum_mul(&cg, r, g4);
    f->v[3] = secp256k1_i128_to_u64(&cf) & M62; secp256k1_i128_rshift(&cf, 62);
    g->v[3] = secp256k1_i128_to_u64(&cg) & M62; secp256k1_i128_rshift(&cg, 62);
    /* What remains is limb 5 of t*[f,g]; store it as output limb 4. */
    f->v[4] = secp256k1_i128_to_i64(&cf);
    g->v[4] = secp256k1_i128_to_i64(&cg);
}



__device__ void secp256k1_modinv64_normalize_62(secp256k1_modinv64_signed62 *r, int64_t sign, const secp256k1_modinv64_modinfo *modinfo) {
    const int64_t M62 = (int64_t)(UINT64_MAX >> 2);
    int64_t r0 = r->v[0], r1 = r->v[1], r2 = r->v[2], r3 = r->v[3], r4 = r->v[4];
    volatile int64_t cond_add, cond_negate;


    /* In a first step, add the modulus if the input is negative, and then negate if requested.
     * This brings r from range (-2*modulus,modulus) to range (-modulus,modulus). As all input
     * limbs are in range (-2^62,2^62), this cannot overflow an int64_t. Note that the right
     * shifts below are signed sign-extending shifts (see assumptions.h for tests that that is
     * indeed the behavior of the right shift operator). */
    cond_add = r4 >> 63;
    r0 += modinfo->modulus.v[0] & cond_add;
    r1 += modinfo->modulus.v[1] & cond_add;
    r2 += modinfo->modulus.v[2] & cond_add;
    r3 += modinfo->modulus.v[3] & cond_add;
    r4 += modinfo->modulus.v[4] & cond_add;
    cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    /* Propagate the top bits, to bring limbs back to range (-2^62,2^62). */
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    /* In a second step add the modulus again if the result is still negative, bringing
     * r to range [0,modulus). */
    cond_add = r4 >> 63;
    r0 += modinfo->modulus.v[0] & cond_add;
    r1 += modinfo->modulus.v[1] & cond_add;
    r2 += modinfo->modulus.v[2] & cond_add;
    r3 += modinfo->modulus.v[3] & cond_add;
    r4 += modinfo->modulus.v[4] & cond_add;
    /* And propagate again. */
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r->v[0] = r0;
    r->v[1] = r1;
    r->v[2] = r2;
    r->v[3] = r3;
    r->v[4] = r4;


}





/* Compute the inverse of x modulo modinfo->modulus, and replace x with it (constant time in x). */
__device__ void secp256k1_modinv64(secp256k1_modinv64_signed62 *x, const secp256k1_modinv64_modinfo *modinfo) {
    /* Start with d=0, e=1, f=modulus, g=x, zeta=-1. */
    secp256k1_modinv64_signed62 d = {{0, 0, 0, 0, 0}};
    secp256k1_modinv64_signed62 e = {{1, 0, 0, 0, 0}};
    secp256k1_modinv64_signed62 f = modinfo->modulus;
    secp256k1_modinv64_signed62 g = *x;
    int i;
    int64_t zeta = -1; /* zeta = -(delta+1/2); delta starts at 1/2. */

    /* Do 10 iterations of 59 divsteps each = 590 divsteps. This suffices for 256-bit inputs. */
    for (i = 0; i < 10; ++i) {
        /* Compute transition matrix and new zeta after 59 divsteps. */
        secp256k1_modinv64_trans2x2 t;
        zeta = secp256k1_modinv64_divsteps_59(zeta, f.v[0], g.v[0], &t);
        /* Update d,e using that transition matrix. */
        secp256k1_modinv64_update_de_62(&d, &e, &t, modinfo);
        /* Update f,g using that transition matrix. */

        secp256k1_modinv64_update_fg_62(&f, &g, &t);

    }

    /* At this point sufficient iterations have been performed that g must have reached 0
     * and (if g was not originally 0) f must now equal +/- GCD of the initial f, g
     * values i.e. +/- 1, and d now contains +/- the modular inverse. */


    /* Optionally negate d, normalize to [0,modulus), and return it. */
    secp256k1_modinv64_normalize_62(&d, f.v[4], modinfo);
    *x = d;
}



__device__ void secp256k1_fe_from_signed62(secp256k1_fe *r, const secp256k1_modinv64_signed62 *a) {
    const uint64_t M52 = UINT64_MAX >> 12;
    const uint64_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2], a3 = a->v[3], a4 = a->v[4];

    /* The output from secp256k1_modinv64{_var} should be normalized to range [0,modulus), and
     * have limbs in [0,2^62). The modulus is < 2^256, so the top limb must be below 2^(256-62*4).
     */


    r->n[0] =  a0                   & M52;
    r->n[1] = (a0 >> 52 | a1 << 10) & M52;
    r->n[2] = (a1 >> 42 | a2 << 20) & M52;
    r->n[3] = (a2 >> 32 | a3 << 30) & M52;
    r->n[4] = (a3 >> 22 | a4 << 40);
}


__constant__ secp256k1_modinv64_modinfo secp256k1_const_modinfo_fe = {
    {{-0x1000003D1LL, 0, 0, 0, 256}},
    0x27C7F6E22DDACACFLL
};

__device__ void secp256k1_fe_impl_inv(secp256k1_fe *r, const secp256k1_fe *x) {
    secp256k1_fe tmp = *x;
    secp256k1_modinv64_signed62 s;

    secp256k1_fe_normalize(&tmp);
    secp256k1_fe_to_signed62(&s, &tmp);
    secp256k1_modinv64(&s, &secp256k1_const_modinfo_fe);
    secp256k1_fe_from_signed62(r, &s);
}


__device__ void secp256k1_ge_set_gej(secp256k1_ge *r, secp256k1_gej *a) {
    secp256k1_fe z2, z3;

    r->infinity = a->infinity;
    secp256k1_fe_inv(&a->z, &a->z);
    secp256k1_fe_sqr(&z2, &a->z);
    secp256k1_fe_mul(&z3, &a->z, &z2);
    secp256k1_fe_mul(&a->x, &a->x, &z2);
    secp256k1_fe_mul(&a->y, &a->y, &z3);
    secp256k1_fe_set_int(&a->z, 1);
    r->x = a->x;
    r->y = a->y;

}


__device__ void secp256k1_scalar_set_b32(secp256k1_scalar *r, const unsigned char *b32, int *overflow) {
    int over;
    r->d[0] = (uint64_t)b32[31] | (uint64_t)b32[30] << 8 | (uint64_t)b32[29] << 16 | (uint64_t)b32[28] << 24 | (uint64_t)b32[27] << 32 | (uint64_t)b32[26] << 40 | (uint64_t)b32[25] << 48 | (uint64_t)b32[24] << 56;
    r->d[1] = (uint64_t)b32[23] | (uint64_t)b32[22] << 8 | (uint64_t)b32[21] << 16 | (uint64_t)b32[20] << 24 | (uint64_t)b32[19] << 32 | (uint64_t)b32[18] << 40 | (uint64_t)b32[17] << 48 | (uint64_t)b32[16] << 56;
    r->d[2] = (uint64_t)b32[15] | (uint64_t)b32[14] << 8 | (uint64_t)b32[13] << 16 | (uint64_t)b32[12] << 24 | (uint64_t)b32[11] << 32 | (uint64_t)b32[10] << 40 | (uint64_t)b32[9] << 48 | (uint64_t)b32[8] << 56;
    r->d[3] = (uint64_t)b32[7] | (uint64_t)b32[6] << 8 | (uint64_t)b32[5] << 16 | (uint64_t)b32[4] << 24 | (uint64_t)b32[3] << 32 | (uint64_t)b32[2] << 40 | (uint64_t)b32[1] << 48 | (uint64_t)b32[0] << 56;
    over = secp256k1_scalar_reduce(r, secp256k1_scalar_check_overflow(r));
    if (overflow) {
        *overflow = over;
    }
}


#define muladd_fast(a,b) { \
    uint64_t tl, th; \
    { \
        uint128_t t = (uint128_t)a * b; \
        th = t >> 64;         /* at most 0xFFFFFFFFFFFFFFFE */ \
        tl = t; \
    } \
    c0 += tl;                 /* overflow is handled on the next line */ \
    th += (c0 < tl) ? 1 : 0;  /* at most 0xFFFFFFFFFFFFFFFF */ \
    c1 += th;                 /* never overflows by contract (verified in the next line) */ \
}

#define extract_fast(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = 0; \
}

#define muladd(a,b) { \
    uint64_t tl, th; \
    { \
        uint128_t t = (uint128_t)a * b; \
        th = t >> 64;         /* at most 0xFFFFFFFFFFFFFFFE */ \
        tl = t; \
    } \
    c0 += tl;                 /* overflow is handled on the next line */ \
    th += (c0 < tl) ? 1 : 0;  /* at most 0xFFFFFFFFFFFFFFFF */ \
    c1 += th;                 /* overflow is handled on the next line */ \
    c2 += (c1 < th) ? 1 : 0;  /* never overflows by contract (verified in the next line) */ \
}

#define extract(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = c2; \
    c2 = 0; \
}


__device__ void secp256k1_scalar_mul_512(uint64_t l[8], const secp256k1_scalar *a, const secp256k1_scalar *b) {

    /* 160 bit accumulator. */
    uint64_t c0 = 0, c1 = 0;
    uint32_t c2 = 0;

    /* l[0..7] = a[0..3] * b[0..3]. */
    muladd_fast(a->d[0], b->d[0]);
    extract_fast(l[0]);
    muladd(a->d[0], b->d[1]);
    muladd(a->d[1], b->d[0]);
    extract(l[1]);
    muladd(a->d[0], b->d[2]);
    muladd(a->d[1], b->d[1]);
    muladd(a->d[2], b->d[0]);
    extract(l[2]);
    muladd(a->d[0], b->d[3]);
    muladd(a->d[1], b->d[2]);
    muladd(a->d[2], b->d[1]);
    muladd(a->d[3], b->d[0]);
    extract(l[3]);
    muladd(a->d[1], b->d[3]);
    muladd(a->d[2], b->d[2]);
    muladd(a->d[3], b->d[1]);
    extract(l[4]);
    muladd(a->d[2], b->d[3]);
    muladd(a->d[3], b->d[2]);
    extract(l[5]);
    muladd_fast(a->d[3], b->d[3]);
    extract_fast(l[6]);

    l[7] = c0;

}

#define sumadd_fast(a) { \
    c0 += (a);                 /* overflow is handled on the next line */ \
    c1 += (c0 < (a)) ? 1 : 0;  /* never overflows by contract (verified the next line) */ \
}

#define sumadd(a) { \
    unsigned int over; \
    c0 += (a);                  /* overflow is handled on the next line */ \
    over = (c0 < (a)) ? 1 : 0; \
    c1 += over;                 /* overflow is handled on the next line */ \
    c2 += (c1 < over) ? 1 : 0;  /* never overflows by contract */ \
}


__device__ void secp256k1_scalar_reduce_512(secp256k1_scalar *r, const uint64_t *l) {

    uint128_t c;
    uint64_t c0, c1, c2;
    uint64_t n0 = l[4], n1 = l[5], n2 = l[6], n3 = l[7];
    uint64_t m0, m1, m2, m3, m4, m5;
    uint32_t m6;
    uint64_t p0, p1, p2, p3;
    uint32_t p4;

    /* Reduce 512 bits into 385. */
    /* m[0..6] = l[0..3] + n[0..3] * SECP256K1_N_C. */
    c0 = l[0]; c1 = 0; c2 = 0;
    muladd_fast(n0, SECP256K1_N_C_0);
    extract_fast(m0);
    sumadd_fast(l[1]);
    muladd(n1, SECP256K1_N_C_0);
    muladd(n0, SECP256K1_N_C_1);
    extract(m1);
    sumadd(l[2]);
    muladd(n2, SECP256K1_N_C_0);
    muladd(n1, SECP256K1_N_C_1);
    sumadd(n0);
    extract(m2);
    sumadd(l[3]);
    muladd(n3, SECP256K1_N_C_0);
    muladd(n2, SECP256K1_N_C_1);
    sumadd(n1);
    extract(m3);
    muladd(n3, SECP256K1_N_C_1);
    sumadd(n2);
    extract(m4);
    sumadd_fast(n3);
    extract_fast(m5);

    m6 = c0;

    /* Reduce 385 bits into 258. */
    /* p[0..4] = m[0..3] + m[4..6] * SECP256K1_N_C. */
    c0 = m0; c1 = 0; c2 = 0;
    muladd_fast(m4, SECP256K1_N_C_0);
    extract_fast(p0);
    sumadd_fast(m1);
    muladd(m5, SECP256K1_N_C_0);
    muladd(m4, SECP256K1_N_C_1);
    extract(p1);
    sumadd(m2);
    muladd(m6, SECP256K1_N_C_0);
    muladd(m5, SECP256K1_N_C_1);
    sumadd(m4);
    extract(p2);
    sumadd_fast(m3);
    muladd_fast(m6, SECP256K1_N_C_1);
    sumadd_fast(m5);
    extract_fast(p3);
    p4 = c0 + m6;


    /* Reduce 258 bits into 256. */
    /* r[0..3] = p[0..3] + p[4] * SECP256K1_N_C. */
    c = p0 + (uint128_t)SECP256K1_N_C_0 * p4;
    r->d[0] = c & 0xFFFFFFFFFFFFFFFFULL; c >>= 64;
    c += p1 + (uint128_t)SECP256K1_N_C_1 * p4;
    r->d[1] = c & 0xFFFFFFFFFFFFFFFFULL; c >>= 64;
    c += p2 + (uint128_t)p4;
    r->d[2] = c & 0xFFFFFFFFFFFFFFFFULL; c >>= 64;
    c += p3;
    r->d[3] = c & 0xFFFFFFFFFFFFFFFFULL; c >>= 64;


    /* Final reduction of r. */
    secp256k1_scalar_reduce(r, c + secp256k1_scalar_check_overflow(r));
}






__device__ void secp256k1_scalar_mul(secp256k1_scalar *r, const secp256k1_scalar *a, const secp256k1_scalar *b) {
    uint64_t l[8];
    secp256k1_scalar_mul_512(l, a, b);
    secp256k1_scalar_reduce_512(r, l);
}


__device__ void secp256k1_scalar_to_signed62(secp256k1_modinv64_signed62 *r, const secp256k1_scalar *a) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const uint64_t a0 = a->d[0], a1 = a->d[1], a2 = a->d[2], a3 = a->d[3];


    r->v[0] =  a0                   & M62;
    r->v[1] = (a0 >> 62 | a1 <<  2) & M62;
    r->v[2] = (a1 >> 60 | a2 <<  4) & M62;
    r->v[3] = (a2 >> 58 | a3 <<  6) & M62;
    r->v[4] =  a3 >> 56;
}

__constant__ secp256k1_modinv64_modinfo secp256k1_const_modinfo_scalar = {
    {{0x3FD25E8CD0364141LL, 0x2ABB739ABD2280EELL, -0x15LL, 0, 256}},
    0x34F20099AA774EC1LL
};

__device__ void secp256k1_scalar_from_signed62(secp256k1_scalar *r, const secp256k1_modinv64_signed62 *a) {
    const uint64_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2], a3 = a->v[3], a4 = a->v[4];

    /* The output from secp256k1_modinv64{_var} should be normalized to range [0,modulus), and
     * have limbs in [0,2^62). The modulus is < 2^256, so the top limb must be below 2^(256-62*4).
     */


    r->d[0] = a0      | a1 << 62;
    r->d[1] = a1 >> 2 | a2 << 60;
    r->d[2] = a2 >> 4 | a3 << 58;
    r->d[3] = a3 >> 6 | a4 << 56;


}

__device__ void secp256k1_scalar_inverse(secp256k1_scalar *r, const secp256k1_scalar *x) {
    secp256k1_modinv64_signed62 s;


    secp256k1_scalar_to_signed62(&s, x);
    secp256k1_modinv64(&s, &secp256k1_const_modinfo_scalar);
    secp256k1_scalar_from_signed62(r, &s);


}

__device__ int secp256k1_scalar_is_zero(const secp256k1_scalar *a) {
    return (a->d[0] | a->d[1] | a->d[2] | a->d[3]) == 0;
}

__device__ int secp256k1_scalar_is_high(const secp256k1_scalar *a) {
    int yes = 0;
    int no = 0;
    no |= (a->d[3] < SECP256K1_N_H_3);
    yes |= (a->d[3] > SECP256K1_N_H_3) & ~no;
    no |= (a->d[2] < SECP256K1_N_H_2) & ~yes; /* No need for a > check. */
    no |= (a->d[1] < SECP256K1_N_H_1) & ~yes;
    yes |= (a->d[1] > SECP256K1_N_H_1) & ~no;
    yes |= (a->d[0] > SECP256K1_N_H_0) & ~no;
    return yes;
}

__device__ void secp256k1_scalar_negate(secp256k1_scalar *r, const secp256k1_scalar *a) {
    uint128_t t;
    if ( secp256k1_scalar_is_zero(a) ) { return; }

    t  = (uint128_t)(~a->d[0]) + SECP256K1_N_0 + 1; r->d[0] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)(~a->d[1]) + SECP256K1_N_1    ; r->d[1] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)(~a->d[2]) + SECP256K1_N_2    ; r->d[2] = t & 0xFFFFFFFFFFFFFFFFULL; t >>= 64;
    t += (uint128_t)(~a->d[3]) + SECP256K1_N_3    ; r->d[3] = t & 0xFFFFFFFFFFFFFFFFULL;
}

__device__ int secp256k1_ecdsa_sig_sign(const secp256k1_ecmult_gen_context *ctx, secp256k1_scalar *sigr, secp256k1_scalar *sigs, const secp256k1_scalar *seckey, const secp256k1_scalar *message, const secp256k1_scalar *nonce, int *recid) {
    unsigned char b[32];
    secp256k1_gej rp;
    memset(&rp,0,sizeof(rp));
    secp256k1_ge r;
    secp256k1_scalar n;
    int overflow = 0;

    secp256k1_ecmult_gen(ctx, &rp, nonce);
    secp256k1_ge_set_gej(&r, &rp);
    secp256k1_fe_normalize(&r.x);
    secp256k1_fe_normalize(&r.y);
    secp256k1_fe_get_b32(b, &r.x);
    secp256k1_scalar_set_b32(sigr, b, &overflow);
    // if (secp256k1_scalar_is_zero(sigr)) {
    //     /* P.x = order is on the curve, so technically sig->r could end up zero, which would be an invalid signature.
    //      * This branch is cryptographically unreachable as hitting it requires finding the discrete log of P.x = N.
    //      */
    //     secp256k1_gej_clear(&rp);
    //     secp256k1_ge_clear(&r);
    //     return 0;
    // }
    // if (recid) {
    //     /* The overflow condition is cryptographically unreachable as hitting it requires finding the discrete log
    //      * of some P where P.x >= order, and only 1 in about 2^127 points meet this criteria.
    //      */
    //     *recid = (overflow ? 2 : 0) | (secp256k1_fe_is_odd(&r.y) ? 1 : 0);
    // }
    secp256k1_scalar_mul(&n, sigr, seckey);
    secp256k1_scalar_add(&n, &n, message);
    secp256k1_scalar_inverse(sigs, nonce);
    secp256k1_scalar_mul(sigs, sigs, &n);
    // secp256k1_scalar_clear(&n);
    // secp256k1_gej_clear(&rp);
    // secp256k1_ge_clear(&r);
    if (secp256k1_scalar_is_zero(sigs)) {
        return 0;
    }
    if (secp256k1_scalar_is_high(sigs)) {
        secp256k1_scalar_negate(sigs, sigs);
        if (recid) {
            *recid ^= 1;
        }
    }
    return 1;
}


typedef struct {
    unsigned char data[64];
} secp256k1_ecdsa_signature;


typedef struct {
    unsigned char v[32];
    unsigned char k[32];
    int retry;
} secp256k1_rfc6979_hmac_sha256;

__device__ void secp256k1_scalar_get_b32(unsigned char *bin, const secp256k1_scalar* a) {
    bin[0] = a->d[3] >> 56; bin[1] = a->d[3] >> 48; bin[2] = a->d[3] >> 40; bin[3] = a->d[3] >> 32; bin[4] = a->d[3] >> 24; bin[5] = a->d[3] >> 16; bin[6] = a->d[3] >> 8; bin[7] = a->d[3];
    bin[8] = a->d[2] >> 56; bin[9] = a->d[2] >> 48; bin[10] = a->d[2] >> 40; bin[11] = a->d[2] >> 32; bin[12] = a->d[2] >> 24; bin[13] = a->d[2] >> 16; bin[14] = a->d[2] >> 8; bin[15] = a->d[2];
    bin[16] = a->d[1] >> 56; bin[17] = a->d[1] >> 48; bin[18] = a->d[1] >> 40; bin[19] = a->d[1] >> 32; bin[20] = a->d[1] >> 24; bin[21] = a->d[1] >> 16; bin[22] = a->d[1] >> 8; bin[23] = a->d[1];
    bin[24] = a->d[0] >> 56; bin[25] = a->d[0] >> 48; bin[26] = a->d[0] >> 40; bin[27] = a->d[0] >> 32; bin[28] = a->d[0] >> 24; bin[29] = a->d[0] >> 16; bin[30] = a->d[0] >> 8; bin[31] = a->d[0];
}

__device__ inline void buffer_append(unsigned char *buf, unsigned int *offset, const void *data, unsigned int len) {
    memcpy(buf + *offset, data, len);
    *offset += len;
}

typedef struct {
    uint32_t s[8];
    unsigned char buf[64];
    uint64_t bytes;
} secp256k1_sha256;

typedef struct {
    secp256k1_sha256 inner, outer;
} secp256k1_hmac_sha256;

__device__ void secp256k1_sha256_initialize(secp256k1_sha256 *hash) {
    hash->s[0] = 0x6a09e667ul;
    hash->s[1] = 0xbb67ae85ul;
    hash->s[2] = 0x3c6ef372ul;
    hash->s[3] = 0xa54ff53aul;
    hash->s[4] = 0x510e527ful;
    hash->s[5] = 0x9b05688cul;
    hash->s[6] = 0x1f83d9abul;
    hash->s[7] = 0x5be0cd19ul;
    hash->bytes = 0;
}

/* Read a uint32_t in big endian */
__device__ uint32_t secp256k1_read_be32(const unsigned char* p) {
    return (uint32_t)p[0] << 24 |
           (uint32_t)p[1] << 16 |
           (uint32_t)p[2] << 8  |
           (uint32_t)p[3];
}

#define BE32(p) ((((p) & 0xFF) << 24) | (((p) & 0xFF00) << 8) | (((p) & 0xFF0000) >> 8) | (((p) & 0xFF000000) >> 24))
#define Ch(x,y,z) ((z) ^ ((x) & ((y) ^ (z))))
#define Maj(x,y,z) (((x) & (y)) | ((z) & ((x) | (y))))
#define Sigma0(x) (((x) >> 2 | (x) << 30) ^ ((x) >> 13 | (x) << 19) ^ ((x) >> 22 | (x) << 10))
#define Sigma1(x) (((x) >> 6 | (x) << 26) ^ ((x) >> 11 | (x) << 21) ^ ((x) >> 25 | (x) << 7))
#define sigma0(x) (((x) >> 7 | (x) << 25) ^ ((x) >> 18 | (x) << 14) ^ ((x) >> 3))
#define sigma1(x) (((x) >> 17 | (x) << 15) ^ ((x) >> 19 | (x) << 13) ^ ((x) >> 10))

#define Round(a,b,c,d,e,f,g,h,k,w) do { \
    uint32_t t1 = (h) + Sigma1(e) + Ch((e), (f), (g)) + (k) + (w); \
    uint32_t t2 = Sigma0(a) + Maj((a), (b), (c)); \
    (d) += t1; \
    (h) = t1 + t2; \
} while(0)

__device__ void secp256k1_sha256_transform(uint32_t* s, const unsigned char* buf) {
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15;

    Round(a, b, c, d, e, f, g, h, 0x428a2f98,  w0 = secp256k1_read_be32(&buf[0]));
    Round(h, a, b, c, d, e, f, g, 0x71374491,  w1 = secp256k1_read_be32(&buf[4]));
    Round(g, h, a, b, c, d, e, f, 0xb5c0fbcf,  w2 = secp256k1_read_be32(&buf[8]));
    Round(f, g, h, a, b, c, d, e, 0xe9b5dba5,  w3 = secp256k1_read_be32(&buf[12]));
    Round(e, f, g, h, a, b, c, d, 0x3956c25b,  w4 = secp256k1_read_be32(&buf[16]));
    Round(d, e, f, g, h, a, b, c, 0x59f111f1,  w5 = secp256k1_read_be32(&buf[20]));
    Round(c, d, e, f, g, h, a, b, 0x923f82a4,  w6 = secp256k1_read_be32(&buf[24]));
    Round(b, c, d, e, f, g, h, a, 0xab1c5ed5,  w7 = secp256k1_read_be32(&buf[28]));
    Round(a, b, c, d, e, f, g, h, 0xd807aa98,  w8 = secp256k1_read_be32(&buf[32]));
    Round(h, a, b, c, d, e, f, g, 0x12835b01,  w9 = secp256k1_read_be32(&buf[36]));
    Round(g, h, a, b, c, d, e, f, 0x243185be, w10 = secp256k1_read_be32(&buf[40]));
    Round(f, g, h, a, b, c, d, e, 0x550c7dc3, w11 = secp256k1_read_be32(&buf[44]));
    Round(e, f, g, h, a, b, c, d, 0x72be5d74, w12 = secp256k1_read_be32(&buf[48]));
    Round(d, e, f, g, h, a, b, c, 0x80deb1fe, w13 = secp256k1_read_be32(&buf[52]));
    Round(c, d, e, f, g, h, a, b, 0x9bdc06a7, w14 = secp256k1_read_be32(&buf[56]));
    Round(b, c, d, e, f, g, h, a, 0xc19bf174, w15 = secp256k1_read_be32(&buf[60]));

    Round(a, b, c, d, e, f, g, h, 0xe49b69c1, w0 += sigma1(w14) + w9 + sigma0(w1));
    Round(h, a, b, c, d, e, f, g, 0xefbe4786, w1 += sigma1(w15) + w10 + sigma0(w2));
    Round(g, h, a, b, c, d, e, f, 0x0fc19dc6, w2 += sigma1(w0) + w11 + sigma0(w3));
    Round(f, g, h, a, b, c, d, e, 0x240ca1cc, w3 += sigma1(w1) + w12 + sigma0(w4));
    Round(e, f, g, h, a, b, c, d, 0x2de92c6f, w4 += sigma1(w2) + w13 + sigma0(w5));
    Round(d, e, f, g, h, a, b, c, 0x4a7484aa, w5 += sigma1(w3) + w14 + sigma0(w6));
    Round(c, d, e, f, g, h, a, b, 0x5cb0a9dc, w6 += sigma1(w4) + w15 + sigma0(w7));
    Round(b, c, d, e, f, g, h, a, 0x76f988da, w7 += sigma1(w5) + w0 + sigma0(w8));
    Round(a, b, c, d, e, f, g, h, 0x983e5152, w8 += sigma1(w6) + w1 + sigma0(w9));
    Round(h, a, b, c, d, e, f, g, 0xa831c66d, w9 += sigma1(w7) + w2 + sigma0(w10));
    Round(g, h, a, b, c, d, e, f, 0xb00327c8, w10 += sigma1(w8) + w3 + sigma0(w11));
    Round(f, g, h, a, b, c, d, e, 0xbf597fc7, w11 += sigma1(w9) + w4 + sigma0(w12));
    Round(e, f, g, h, a, b, c, d, 0xc6e00bf3, w12 += sigma1(w10) + w5 + sigma0(w13));
    Round(d, e, f, g, h, a, b, c, 0xd5a79147, w13 += sigma1(w11) + w6 + sigma0(w14));
    Round(c, d, e, f, g, h, a, b, 0x06ca6351, w14 += sigma1(w12) + w7 + sigma0(w15));
    Round(b, c, d, e, f, g, h, a, 0x14292967, w15 += sigma1(w13) + w8 + sigma0(w0));

    Round(a, b, c, d, e, f, g, h, 0x27b70a85, w0 += sigma1(w14) + w9 + sigma0(w1));
    Round(h, a, b, c, d, e, f, g, 0x2e1b2138, w1 += sigma1(w15) + w10 + sigma0(w2));
    Round(g, h, a, b, c, d, e, f, 0x4d2c6dfc, w2 += sigma1(w0) + w11 + sigma0(w3));
    Round(f, g, h, a, b, c, d, e, 0x53380d13, w3 += sigma1(w1) + w12 + sigma0(w4));
    Round(e, f, g, h, a, b, c, d, 0x650a7354, w4 += sigma1(w2) + w13 + sigma0(w5));
    Round(d, e, f, g, h, a, b, c, 0x766a0abb, w5 += sigma1(w3) + w14 + sigma0(w6));
    Round(c, d, e, f, g, h, a, b, 0x81c2c92e, w6 += sigma1(w4) + w15 + sigma0(w7));
    Round(b, c, d, e, f, g, h, a, 0x92722c85, w7 += sigma1(w5) + w0 + sigma0(w8));
    Round(a, b, c, d, e, f, g, h, 0xa2bfe8a1, w8 += sigma1(w6) + w1 + sigma0(w9));
    Round(h, a, b, c, d, e, f, g, 0xa81a664b, w9 += sigma1(w7) + w2 + sigma0(w10));
    Round(g, h, a, b, c, d, e, f, 0xc24b8b70, w10 += sigma1(w8) + w3 + sigma0(w11));
    Round(f, g, h, a, b, c, d, e, 0xc76c51a3, w11 += sigma1(w9) + w4 + sigma0(w12));
    Round(e, f, g, h, a, b, c, d, 0xd192e819, w12 += sigma1(w10) + w5 + sigma0(w13));
    Round(d, e, f, g, h, a, b, c, 0xd6990624, w13 += sigma1(w11) + w6 + sigma0(w14));
    Round(c, d, e, f, g, h, a, b, 0xf40e3585, w14 += sigma1(w12) + w7 + sigma0(w15));
    Round(b, c, d, e, f, g, h, a, 0x106aa070, w15 += sigma1(w13) + w8 + sigma0(w0));

    Round(a, b, c, d, e, f, g, h, 0x19a4c116, w0 += sigma1(w14) + w9 + sigma0(w1));
    Round(h, a, b, c, d, e, f, g, 0x1e376c08, w1 += sigma1(w15) + w10 + sigma0(w2));
    Round(g, h, a, b, c, d, e, f, 0x2748774c, w2 += sigma1(w0) + w11 + sigma0(w3));
    Round(f, g, h, a, b, c, d, e, 0x34b0bcb5, w3 += sigma1(w1) + w12 + sigma0(w4));
    Round(e, f, g, h, a, b, c, d, 0x391c0cb3, w4 += sigma1(w2) + w13 + sigma0(w5));
    Round(d, e, f, g, h, a, b, c, 0x4ed8aa4a, w5 += sigma1(w3) + w14 + sigma0(w6));
    Round(c, d, e, f, g, h, a, b, 0x5b9cca4f, w6 += sigma1(w4) + w15 + sigma0(w7));
    Round(b, c, d, e, f, g, h, a, 0x682e6ff3, w7 += sigma1(w5) + w0 + sigma0(w8));
    Round(a, b, c, d, e, f, g, h, 0x748f82ee, w8 += sigma1(w6) + w1 + sigma0(w9));
    Round(h, a, b, c, d, e, f, g, 0x78a5636f, w9 += sigma1(w7) + w2 + sigma0(w10));
    Round(g, h, a, b, c, d, e, f, 0x84c87814, w10 += sigma1(w8) + w3 + sigma0(w11));
    Round(f, g, h, a, b, c, d, e, 0x8cc70208, w11 += sigma1(w9) + w4 + sigma0(w12));
    Round(e, f, g, h, a, b, c, d, 0x90befffa, w12 += sigma1(w10) + w5 + sigma0(w13));
    Round(d, e, f, g, h, a, b, c, 0xa4506ceb, w13 += sigma1(w11) + w6 + sigma0(w14));
    Round(c, d, e, f, g, h, a, b, 0xbef9a3f7, w14 + sigma1(w12) + w7 + sigma0(w15));
    Round(b, c, d, e, f, g, h, a, 0xc67178f2, w15 + sigma1(w13) + w8 + sigma0(w0));

    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
    s[5] += f;
    s[6] += g;
    s[7] += h;
}


__device__ void secp256k1_sha256_write(secp256k1_sha256 *hash, const unsigned char *data, size_t len) {
    size_t bufsize = hash->bytes & 0x3F;
    hash->bytes += len;
    while (bufsize + len >= 64) {
        /* Fill the buffer, and process it. */
        memcpy(((unsigned char*)hash->buf) + bufsize, data, 64 - bufsize);
        data += 64 - bufsize;
        len -= 64 - bufsize;
        secp256k1_sha256_transform(hash->s, hash->buf);
        bufsize = 0;
    }
    if (len) {
        /* Fill the buffer with what remains. */
        memcpy(((unsigned char*)hash->buf) + bufsize, data, len);
    }
}


__device__ void secp256k1_sha256_finalize(secp256k1_sha256 *hash, unsigned char *out32) {
    static __constant__ unsigned char pad[64] = {0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t sizedesc[2];
    uint32_t out[8];
    int i = 0;
    sizedesc[0] = BE32(hash->bytes >> 29);
    sizedesc[1] = BE32(hash->bytes << 3);
    secp256k1_sha256_write(hash, pad, 1 + ((119 - (hash->bytes % 64)) % 64));
    secp256k1_sha256_write(hash, (const unsigned char*)sizedesc, 8);
    for (i = 0; i < 8; i++) {
        out[i] = BE32(hash->s[i]);
        hash->s[i] = 0;
    }
    memcpy(out32, (const unsigned char*)out, 32);
}

__device__ void secp256k1_hmac_sha256_initialize(secp256k1_hmac_sha256 *hash, const unsigned char *key, size_t keylen) {
    int n;
    unsigned char rkey[64];
    if (keylen <= 64) {
        memcpy(rkey, key, keylen);
        memset(rkey + keylen, 0, 64 - keylen);
    } else {
        secp256k1_sha256 sha256;
        secp256k1_sha256_initialize(&sha256);
        secp256k1_sha256_write(&sha256, key, keylen);
        secp256k1_sha256_finalize(&sha256, rkey);
        memset(rkey + 32, 0, 32);
    }

    secp256k1_sha256_initialize(&hash->outer);
    for (n = 0; n < 64; n++) {
        rkey[n] ^= 0x5c;
    }
    secp256k1_sha256_write(&hash->outer, rkey, 64);

    secp256k1_sha256_initialize(&hash->inner);
    for (n = 0; n < 64; n++) {
        rkey[n] ^= 0x5c ^ 0x36;
    }
    secp256k1_sha256_write(&hash->inner, rkey, 64);
    memset(rkey, 0, 64);
}

__device__ void secp256k1_hmac_sha256_write(secp256k1_hmac_sha256 *hash, const unsigned char *data, size_t size) {
    secp256k1_sha256_write(&hash->inner, data, size);
}


__device__ void secp256k1_hmac_sha256_finalize(secp256k1_hmac_sha256 *hash, unsigned char *out32) {
    unsigned char temp[32];
    secp256k1_sha256_finalize(&hash->inner, temp);
    secp256k1_sha256_write(&hash->outer, temp, 32);
    memset(temp, 0, 32);
    secp256k1_sha256_finalize(&hash->outer, out32);
}

__device__ void secp256k1_rfc6979_hmac_sha256_initialize(secp256k1_rfc6979_hmac_sha256 *rng, const unsigned char *key, size_t keylen) {
    secp256k1_hmac_sha256 hmac;
    static const unsigned char zero[1] = {0x00};
    static const unsigned char one[1] = {0x01};

    memset(rng->v, 0x01, 32); /* RFC6979 3.2.b. */
    memset(rng->k, 0x00, 32); /* RFC6979 3.2.c. */

    /* RFC6979 3.2.d. */
    secp256k1_hmac_sha256_initialize(&hmac, rng->k, 32);
    secp256k1_hmac_sha256_write(&hmac, rng->v, 32);
    secp256k1_hmac_sha256_write(&hmac, zero, 1);
    secp256k1_hmac_sha256_write(&hmac, key, keylen);
    secp256k1_hmac_sha256_finalize(&hmac, rng->k);
    secp256k1_hmac_sha256_initialize(&hmac, rng->k, 32);
    secp256k1_hmac_sha256_write(&hmac, rng->v, 32);
    secp256k1_hmac_sha256_finalize(&hmac, rng->v);

    /* RFC6979 3.2.f. */
    secp256k1_hmac_sha256_initialize(&hmac, rng->k, 32);
    secp256k1_hmac_sha256_write(&hmac, rng->v, 32);
    secp256k1_hmac_sha256_write(&hmac, one, 1);
    secp256k1_hmac_sha256_write(&hmac, key, keylen);
    secp256k1_hmac_sha256_finalize(&hmac, rng->k);
    secp256k1_hmac_sha256_initialize(&hmac, rng->k, 32);
    secp256k1_hmac_sha256_write(&hmac, rng->v, 32);
    secp256k1_hmac_sha256_finalize(&hmac, rng->v);
    rng->retry = 0;
}


__device__ void secp256k1_rfc6979_hmac_sha256_generate(secp256k1_rfc6979_hmac_sha256 *rng, unsigned char *out, size_t outlen) {
    /* RFC6979 3.2.h. */
    static const unsigned char zero[1] = {0x00};
    if (rng->retry) {
        secp256k1_hmac_sha256 hmac;
        secp256k1_hmac_sha256_initialize(&hmac, rng->k, 32);
        secp256k1_hmac_sha256_write(&hmac, rng->v, 32);
        secp256k1_hmac_sha256_write(&hmac, zero, 1);
        secp256k1_hmac_sha256_finalize(&hmac, rng->k);
        secp256k1_hmac_sha256_initialize(&hmac, rng->k, 32);
        secp256k1_hmac_sha256_write(&hmac, rng->v, 32);
        secp256k1_hmac_sha256_finalize(&hmac, rng->v);
    }

    while (outlen > 0) {
        secp256k1_hmac_sha256 hmac;
        int now = outlen;
        secp256k1_hmac_sha256_initialize(&hmac, rng->k, 32);
        secp256k1_hmac_sha256_write(&hmac, rng->v, 32);
        secp256k1_hmac_sha256_finalize(&hmac, rng->v);
        if (now > 32) {
            now = 32;
        }
        memcpy(out, rng->v, now);
        out += now;
        outlen -= now;
    }

    rng->retry = 1;
}


__device__ void secp256k1_rfc6979_hmac_sha256_finalize(secp256k1_rfc6979_hmac_sha256 *rng) {
    memset(rng->k, 0, 32);
    memset(rng->v, 0, 32);
    rng->retry = 0;
}

__device__ int nonce_function_rfc6979(unsigned char *nonce32, const unsigned char *msg32, const unsigned char *key32, const unsigned char *algo16, void *data, unsigned int counter) {
   unsigned char keydata[112];
   unsigned int offset = 0;
   secp256k1_rfc6979_hmac_sha256 rng;
   unsigned int i;
   secp256k1_scalar msg;
   unsigned char msgmod32[32];
   secp256k1_scalar_set_b32(&msg, msg32, NULL);
   secp256k1_scalar_get_b32(msgmod32, &msg);
   /* We feed a byte array to the PRNG as input, consisting of:
    * - the private key (32 bytes) and reduced message (32 bytes), see RFC 6979 3.2d.
    * - optionally 32 extra bytes of data, see RFC 6979 3.6 Additional Data.
    * - optionally 16 extra bytes with the algorithm name.
    * Because the arguments have distinct fixed lengths it is not possible for
    *  different argument mixtures to emulate each other and result in the same
    *  nonces.
    */
   buffer_append(keydata, &offset, key32, 32);
   buffer_append(keydata, &offset, msgmod32, 32);
   if (data != NULL) {
       buffer_append(keydata, &offset, data, 32);
   }
   if (algo16 != NULL) {
       buffer_append(keydata, &offset, algo16, 16);
   }
   secp256k1_rfc6979_hmac_sha256_initialize(&rng, keydata, offset);
   memset(keydata, 0, sizeof(keydata));
   for (i = 0; i <= counter; i++) {
       secp256k1_rfc6979_hmac_sha256_generate(&rng, nonce32, 32);
   }
   secp256k1_rfc6979_hmac_sha256_finalize(&rng);
   return 1;
}










__device__ void secp256k1_ecdsa_signature_save(secp256k1_ecdsa_signature* sig, const secp256k1_scalar* r, const secp256k1_scalar* s) {
    if (sizeof(secp256k1_scalar) == 32) {
        memcpy(&sig->data[0], r, 32);
        memcpy(&sig->data[32], s, 32);
    } else {
        secp256k1_scalar_get_b32(&sig->data[0], r);
        secp256k1_scalar_get_b32(&sig->data[32], s);
    }
}





__device__ int secp256k1_ecdsa_sign(const secp256k1_ecmult_gen_context *ctx, secp256k1_ecdsa_signature *signature, const unsigned char *msg32, const unsigned char *seckey, const void* noncedata) {
    secp256k1_scalar r, s;
    secp256k1_scalar sec, non, msg;
    int ret = 0;
    int overflow = 0;

    secp256k1_scalar_set_b32(&sec, seckey, &overflow);
    /* Fail if the secret key is invalid. */
    if (!overflow && !secp256k1_scalar_is_zero(&sec)) {
        unsigned int count = 0;
        secp256k1_scalar_set_b32(&msg, msg32, NULL);
        while (1) {
            unsigned char nonce32[32];
            ret = nonce_function_rfc6979(nonce32, msg32, seckey, NULL, (void*)noncedata, count);
            if (!ret) {
                break;
            }
            secp256k1_scalar_set_b32(&non, nonce32, &overflow);
            memset(nonce32, 0, 32);
            if (!overflow && !secp256k1_scalar_is_zero(&non)) {
                if (secp256k1_ecdsa_sig_sign(ctx, &r, &s, &sec, &msg, &non, NULL)) {
                    break;
                }
            }
            count++;
        }

    }
    if (ret) {
        secp256k1_ecdsa_signature_save(signature, &r, &s);
    } else {
        memset(signature, 0, sizeof(*signature));
    }
    return ret;
}



__device__ int secp256k1_ecdsa_sig_serialize(unsigned char *sig, size_t *size, const secp256k1_scalar* ar, const secp256k1_scalar* as) {
    unsigned char r[33] = {0}, s[33] = {0};
    unsigned char *rp = r, *sp = s;
    size_t lenR = 33, lenS = 33;
    secp256k1_scalar_get_b32(&r[1], ar);
    secp256k1_scalar_get_b32(&s[1], as);
    while (lenR > 1 && rp[0] == 0 && rp[1] < 0x80) { lenR--; rp++; }
    while (lenS > 1 && sp[0] == 0 && sp[1] < 0x80) { lenS--; sp++; }
    if (*size < 6+lenS+lenR) {
        *size = 6 + lenS + lenR;
        return 0;
    }
    *size = 6 + lenS + lenR;
    sig[0] = 0x30;
    sig[1] = 4 + lenS + lenR;
    sig[2] = 0x02;
    sig[3] = lenR;
    memcpy(sig+4, rp, lenR);
    sig[4+lenR] = 0x02;
    sig[5+lenR] = lenS;
    memcpy(sig+lenR+6, sp, lenS);
    return 1;
}


__device__ void secp256k1_ecdsa_signature_load(secp256k1_scalar* r, secp256k1_scalar* s, const secp256k1_ecdsa_signature* sig) {
        memcpy(r, &sig->data[0], 32);
        memcpy(s, &sig->data[32], 32);
}


__device__ int secp256k1_ecdsa_signature_serialize_der(unsigned char *output, size_t *outputlen, const secp256k1_ecdsa_signature* sig) {
    secp256k1_scalar r, s;

    secp256k1_ecdsa_signature_load(&r, &s, sig);
    return secp256k1_ecdsa_sig_serialize(output, outputlen, &r, &s);
}





__device__ size_t Sign(const uint8_t *p_hash, uint8_t * p_vchSig, secp256k1_ecmult_gen_context *ctx, unsigned char *p_key_data) {

    size_t nSigLen = 72;

    secp256k1_ecdsa_signature sig;

    secp256k1_ecdsa_sign(ctx, &sig, p_hash, p_key_data,  nullptr);


    // printf("============OUTPUT SIGNATURE============\n");
    // for(int z=0;z<sizeof(secp256k1_ecdsa_signature);z++)
    // {
    //     printf("%02X",sig.data[z]);
    // }
    // printf("\n\n");    

    secp256k1_ecdsa_signature_serialize_der(p_vchSig, &nSigLen, &sig);

    // printf("============OUTPUT vchSig============\n");
    // for(int z=0;z<nSigLen;z++)
    // {
    //     printf("%02X",p_vchSig[z]);
    // }
    // printf("\n\n");        

    return nSigLen;
}




__device__ void sha256_hash(BYTE* signature, WORD sig_len, BYTE* hash)//, WORD* nonce)
{ 
	// WORD thread = blockIdx.x * blockDim.x + threadIdx.x;
	// if (thread >= total_threads)
	// {
	// 	return;
	// }
	// BYTE* in = indata  + thread * inlen;
	// BYTE* out = outdata  + thread * SHA256_BLOCK_SIZE;
	CUDA_SHA256_CTX ctx;

	cuda_sha256_init(&ctx);
	cuda_sha256_update(&ctx, signature, sig_len);
	cuda_sha256_final(&ctx, hash);//, nonce);

    cuda_sha256_init(&ctx);
	cuda_sha256_update(&ctx, hash, 32);
	cuda_sha256_final(&ctx, hash);//, nonce);
}






// Define a struct to represent a uint256 (256-bit integer)
struct uint256 {
    uint64_t data[4];  // Array to hold four 64-bit parts
};

// Device function to add two uint256 numbers
__device__ uint256 addUint256(const uint256& a, const uint256& b) {
    uint256 result;

    // Perform the addition (low + low) with carry propagation
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a.data[i] + b.data[i] + carry;
        result.data[i] = sum;
        carry = (sum < a.data[i]) ? 1 : 0;  // If sum overflows, set carry
    }

    return result;
}


// Device function to add uint256 with uint64_t
__device__ uint256 addUint256_64(const uint256& a, uint64_t b) {
    uint256 result;

    // Initialize carry to handle overflow from 64-bit addition
    uint64_t carry = 0;

    // Add the uint64_t b to the least significant 64 bits of uint256 (a.data[0])
    uint64_t low0 = a.data[0] + b;
    carry = (low0 < a.data[0]) ? 1 : 0;  // Check for carry overflow

    // Store the result
    result.data[0] = low0;

    // Add the carry to the next higher word and propagate
    result.data[1] = a.data[1] + carry;
    result.data[2] = a.data[2];  // No carry beyond this word
    result.data[3] = a.data[3];  // No carry beyond this word

    return result;
}

#include <unistd.h>  // For usleep()
// REGULAR EXP
//           ^(.*)$        0x$1,






__global__ void cuda_miner(BYTE* d_gpu_num, BYTE* d_is_stage1, BYTE* key_data, BYTE* ctx_data, BYTE* hash_no_sig_in, BYTE* nonce4host, BYTE* d_utxo_set_idx4host, BYTE* d_utxo_set_time4host, BYTE* d_stake_modifier, BYTE* d_utxos_block_from_time, BYTE* d_utxos_hash, BYTE* d_utxos_n, BYTE* d_start_time,
                    BYTE* d_hash_merkle_root, BYTE* d_hash_prev_block, BYTE* d_n_bits, BYTE* d_n_time, BYTE* d_prev_stake_hash, BYTE* d_prev_stake_n, BYTE* d_block_sig )
{


    secp256k1_ecmult_gen_context ctx_obj;
    secp256k1_ecmult_gen_context *ctx = &ctx_obj;

    uint256 hash_no_sig;
    uint8_t mud_array[32];
    BYTE hash_output[32];

    uint8_t ss_for_hashing[1024]; // (1024 for safety)70 or 71 vchsig bytes + 9 bytes for nonce and vchSig length

    uint256 mud;

    uint64_t thread = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t gpu_num = *d_gpu_num; 

    uint32_t throttle = 0;

    uint8_t ctx_data_prev = 11;
    uint8_t key_data_prev = 22;
    uint8_t hash_no_sig_data_prev = 33;




    //  GPU uses upper half of nonce, CPU will use lower half
    uint128_t offset = thread*0x0000100000000000ULL + gpu_num*0x0000001000000000ULL + 0x8000000000000000ULL; 

    uint128_t nonce = offset;

    // printf("THREAD: %d\n", thread);
    // printf("OFFSET: %016x\n", offset);
    for ( ; ; nonce++ )
    {



        /////////////====================================    STAGE1    ====================================/////////////
        if( d_is_stage1[0] )
        {
            // All zeros means the BTCW node is telling the miner to sleep, it is in stage1 at the moment.
            //__nanosleep(1000000000);



            uint32_t time;
            // assuming 16,000 threads, 1200000/15000 = 80 per thread.
            int const AMOUNT_PER_THREAD = 133;
            
            memcpy(&time, &d_start_time[0],4);
            uint32_t offset1 = thread*AMOUNT_PER_THREAD; // each thread can do

            // if ( thread == 0 )
            // {
            //     //printf("START\n");
            // }



            // sweep up to 10 minutes ahead, if we find a solution, we will take much time on stage2 before we find it and maybe
            // by the time stage2 solution is found, the time is valid and not too far in the future...
            for ( uint32_t xxx=0; xxx<100000; xxx++ ) 
            {
                if( !d_is_stage1[0] )
                {
                    // Need to go to stage2 now!!!>
                    printf("====GOING to STAGE2 NOW====:%d\n", thread);               
                    nonce = offset; // this might be the better place to reset the nonce for stage2
                    throttle = 0;
                    break;
                }

                // need a valid time before trying to find a solution
                if ( time < 100000 )
                {
                    // new block was found
                    break;
                }

                time++;
            

                for ( uint32_t utxo_set_idx=offset1; utxo_set_idx<(offset1+AMOUNT_PER_THREAD); utxo_set_idx++ )
                {
                    if( !d_is_stage1[0] )
                    {
                        // Need to go to stage2 now!!!>
                        break;
                    }       

                    if( utxo_set_idx >= WALLET_UTXOS_LENGTH )
                    {
                        // finished looking
                        break;
                    }                                      
                    //=======================STAGE1====================================
                    // // Calculate hash
                    // CDataStream ss(SER_GETHASH, 0);
                    // ss << nStakeModifier;
                    // ss << blockFromTime << prevout.hash << prevout.n << nTimeBlock;
                    // hashProofOfStake = Hash(ss);

                    memcpy(&ss_for_hashing[0],&d_stake_modifier[0],32);
                    memcpy(&ss_for_hashing[32],&d_utxos_block_from_time[utxo_set_idx*4],4); // time of previous block of utxo
                    memcpy(&ss_for_hashing[36],&d_utxos_hash[utxo_set_idx*32],32);
                    memcpy(&ss_for_hashing[68],&d_utxos_n[utxo_set_idx*4],4);
                    memcpy(&ss_for_hashing[72],&time,4); // This is index 'i', needs to keep incrementing

                    sha256_hash(&ss_for_hashing[0], 76, hash_output);

                    // // Now check if hash meets target protocol
                    // arith_uint256 actual = UintToArith256(hashProofOfStake);

                    // BitcoinPoW - HARDFORK - Block 23,333 and beyond - add more CPU logic work and sha256 work
                    // NOTE: Validation needs to see a solution somewhere in the 256 window. It doesn't matter which of the 256
                    //       attempts has the valid solution.
                    // int h = ChainActive().Height();
                    uint64_t data = 0;
                    uint16_t a = 0;
                    uint16_t b = 0;
                    uint16_t c = 0;
                    uint16_t d = 0;
                    uint16_t e = 0;
                    uint16_t f = 0;
                    uint16_t g = 0;
                    // auto& chain_active = gp_chainman->m_active_chainstate->m_chain;
                    for ( volatile int k=1; k<=64; k++ )
                    {
                        //     // Grab values from random previous headers
                        //     data = actual.GetLow64();
                        memcpy(&data, &hash_output[0], 8);
                        a = (20000 + (data>>0))&0xFF;
                        b = (18000 + (data>>8))&0xFF;
                        c = (16000 + (data>>16))&0xFF;
                        d = (14000 + (data>>24))&0xFF;
                        e = (12000 + (data>>32))&0xFF;
                        f = (10000 + (data>>40))&0xFF;
                        g = ( 8000 + (data>>48))&0xFF;
                        //printf("abcdefg:%02X %02X %02X %02X %02X %02X %02X\n",a,b,c,d,e,f,g);
                        //     CDataStream ss(SER_GETHASH, 0);

                        //     ss << chain_active[h-a]->GetBlockHeader_hashMerkleRoot() << 
                        //         chain_active[h-b]->GetBlockHeader_hashPrevBlock() << 
                        //         chain_active[h-c]->GetBlockHeader_nBits() << 
                        //         chain_active[h-d]->GetBlockHeader_nTime() <<
                        //         chain_active[h-e]->GetBlockHeader_prevoutStakehash() << 
                        //         chain_active[h-f]->GetBlockHeader_prevoutStaken() << 
                        //         chain_active[h-g]->GetBlockHeader_vchBlockSig();
                        memcpy(&ss_for_hashing[0], &d_hash_merkle_root[a*32],32);
                        memcpy(&ss_for_hashing[32], &d_hash_prev_block[b*32],32);
                        memcpy(&ss_for_hashing[64], &d_n_bits[c*4],4);
                        memcpy(&ss_for_hashing[68], &d_n_time[d*4],4);
                        memcpy(&ss_for_hashing[72], &d_prev_stake_hash[e*32],32);
                        memcpy(&ss_for_hashing[104], &d_prev_stake_n[f*4],4);
                        unsigned int total_len = 32 + 32 + 4 + 4 + 32 + 4;

            
                        
                        // 0x30 is start byte for signature, next byte is length of sig - 2

                        if ( (d_block_sig[g*80] == 0x30) && (d_block_sig[g*80+1] == 68) ) // 70 for sig  8 bytes for nonce appended in HDR
                        {
                            ss_for_hashing[108] = 78; // doing the serialization shit that << put in there
                            memcpy(&ss_for_hashing[109], &d_block_sig[g*80],78); // copy length field + 70 bytes -> always skip over 80 bytes at a time unused byte here, no issue
                            total_len += 79; // added for index 108 insertion
                        }
                        else if ( (d_block_sig[g*80] == 0x30) && (d_block_sig[g*80+1] == 69) ) // 71 for sig  8 for nonce appended in HDR
                        {
                            ss_for_hashing[108] = 79; // doing the serialization shit that << put in there
                            memcpy(&ss_for_hashing[109], &d_block_sig[g*80],79); // copy length field + 71 bytes -> always skip over 80 bytes at a time unused byte here, no issue
                            total_len += 80; // added for index 108 insertion
                        }
                        else
                        {   
                            total_len = 0; // why we here????
                        }
                        
                        sha256_hash(&ss_for_hashing[0], total_len, hash_output);


                        //     hashProofOfStake = Hash(ss);

                        //     actual = UintToArith256(hashProofOfStake);
                        //     if (actual <= bnTarget)
                        //         return true;   

                        // output with zeros is:  805966DDB62F91D66903FD81F6B30DCC3A351E1FCBA72679C224AE77667E0000            <------- Flipped like this look for leading zeros
                        if ( (hash_output[31] == 0) && (hash_output[30] == 0) && (hash_output[29] == 0) && (hash_output[28] == 0) && ((hash_output[27]&0xC0 ) == 0) )
                        {                       
                            // possible solution found, let the host know and go try more utxos
                            memcpy( &d_utxo_set_idx4host[0],  &utxo_set_idx, 4);
                            memcpy( &d_utxo_set_time4host[0],  &time, 4);

                            printf("========================PoS FOUND========================\n");
                            for(int z=0;z<32;z++)
                            {
                                printf("%02X",hash_output[z]);
                            }
                            printf("\n");       
                            printf("d_stake_modifier\n");  
                            for(int z=0;z<32;z++)
                            {
                                printf("%02X",d_stake_modifier[z]);
                            }
                            printf("\n");      
                            printf("d_utxos_block_from_time\n");  
                            for(int z=0;z<4;z++)
                            {
                                printf("%02X",d_utxos_block_from_time[utxo_set_idx*4+z]);
                            }
                            printf("\n");  
                            printf("d_utxos_hash\n");  
                            for(int z=0;z<32;z++)
                            {
                                printf("%02X",d_utxos_hash[utxo_set_idx*32+z]);
                            }
                            printf("\n");  
                            printf("d_utxos_n\n");  
                            for(int z=0;z<4;z++)
                            {
                                printf("%02X",d_utxos_n[utxo_set_idx*4+z]);
                            }
                            printf("\n");  
                            printf("THREAD: %016llx\n", thread);
                            printf("PoS IDX: %d\n", utxo_set_idx);
                            printf("Time: %d\n", time);
                           
                            break;
                        }
                                            
                    }

                }

            }

            // if ( thread == 0 )
            // {
            //     //printf("END\n");
            // }            

        }
        /////////////====================================    STAGE2    ====================================/////////////
        else
        {
            // if ( thread == 2000 )
            // {
            //     printf("NONCE: %016x\n", nonce);
            // }
            
            mud = addUint256_64(hash_no_sig, nonce);

            memcpy(&mud_array[0],&mud.data[0],8);
            memcpy(&mud_array[8],&mud.data[1],8);
            memcpy(&mud_array[16],&mud.data[2],8);
            memcpy(&mud_array[24],&mud.data[3],8);

            size_t sig_len = Sign(mud_array, &ss_for_hashing[9], ctx, key_data);


            // CDataStream ss(SER_GETHASH, 0);
            // ss << nonce << vchBlockSig;
            memcpy(&ss_for_hashing[0],&nonce,8);

            ss_for_hashing[8] = 70;
            if (sig_len == 71)
            {
                ss_for_hashing[8] = 71;
            }


            // need to add the other 9 bytes in front of the vch [nonce  vchlen      vchsig ..........]
            sha256_hash(&ss_for_hashing[0], sig_len+9, hash_output);


            // output with zeros is:  805966DDB62F91D66903FD81F6B30DCC3A351E1FCBA72679C224AE77667E0000            <------- Flipped like this look for leading zeros
            if ( (hash_output[31] == 0) && (hash_output[30] == 0) && (hash_output[29] == 0) && ((hash_output[28]&0xFC) == 0) )
            //if ( (hash_output[31] == 0) && (hash_output[30] == 0) && (hash_output[29] == 0) ) //&& (hash_output[28] == 0) )
            { 
                      
                memcpy( &nonce4host[0],  &nonce, 8);
                
                printf("========================PoW BLOCK FOUND========================\n");
                printf("THREAD: %016llx\n", thread);
                printf("NONCE: %016llx\n", nonce);
                printf("============OUTPUT HASH============\n");
                for(int z=0;z<32;z++)
                {
                    printf("%02X",hash_output[z]);
                }
                printf("\n\n"); 
            }
            
            // if ( thread == 1000 )
            // {
            // printf("THREAD: %016x\n", thread);
            // printf("NONCE: %016x\n", nonce);
            // }       



            

            // if ( (ctx_data_prev == ctx_data[0]) && (key_data_prev == key_data[0]) && (hash_no_sig_data_prev == hash_no_sig_in[0]) )
            // {
            //     // Data has not changed, do not send update
            // }
            // else
            if ( (throttle&0x1FFF) == 0xFF) // hackish, but is true must quicker when transition from stage1 to stage2,  0 is somehow missed so we bump it to 0xFF ....
            {
                //Host update the data, send it to the GPU
                if ( thread == 0 )
                        printf("STAGE2 BLOCK DATA UPDATED - DEVICE\n");

                    // Update the KEY DATA
                    // NOTE: key_data is actually used straight from the input buffer, no need to do anything here.


                    // just blindly copy the data from host to this structure

                    // ============INPUTS============  20*8bytes=160 bytes total
                    // secp256k1_ecmult_gen_context:
                    memcpy( &ctx->blind.d[0], &ctx_data[0], 8);
                    memcpy( &ctx->blind.d[1], &ctx_data[8], 8);
                    memcpy( &ctx->blind.d[2], &ctx_data[16], 8);
                    memcpy( &ctx->blind.d[3], &ctx_data[24], 8);
                    memcpy( &ctx->initial.x.n[0], &ctx_data[32], 8);
                    memcpy( &ctx->initial.x.n[1], &ctx_data[40], 8);
                    memcpy( &ctx->initial.x.n[2], &ctx_data[48], 8);
                    memcpy( &ctx->initial.x.n[3], &ctx_data[56], 8);
                    memcpy( &ctx->initial.x.n[4], &ctx_data[64], 8);
                    memcpy( &ctx->initial.y.n[0], &ctx_data[72], 8);
                    memcpy( &ctx->initial.y.n[1], &ctx_data[80], 8);
                    memcpy( &ctx->initial.y.n[2], &ctx_data[88], 8);
                    memcpy( &ctx->initial.y.n[3], &ctx_data[96], 8);
                    memcpy( &ctx->initial.y.n[4], &ctx_data[104], 8);
                    memcpy( &ctx->initial.z.n[0], &ctx_data[112], 8);
                    memcpy( &ctx->initial.z.n[1], &ctx_data[120], 8);
                    memcpy( &ctx->initial.z.n[2], &ctx_data[128], 8);
                    memcpy( &ctx->initial.z.n[3], &ctx_data[136], 8);
                    memcpy( &ctx->initial.z.n[4], &ctx_data[144], 8);
                    memcpy( &ctx->initial.infinity, &ctx_data[152], 8);
                    memcpy( &ctx->built, &ctx_data[156], 4);




                    // Update hash no sig
                    memcpy( &hash_no_sig.data[0],  &hash_no_sig_in[0], 8);
                    memcpy( &hash_no_sig.data[1],  &hash_no_sig_in[8], 8);
                    memcpy( &hash_no_sig.data[2], &hash_no_sig_in[16], 8);
                    memcpy( &hash_no_sig.data[3], &hash_no_sig_in[24], 8);     


                }
            throttle++;


        }

    }


}










#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>

#define SHM_NAME "/shared_mem"
#define SEM_EMPTY "/sem_empty"
#define SEM_FULL "/sem_full"

//const int NUM_HASHES_BATCH = (1<<24); // This is really the total number of threads we want on the GPU (16M   )
const int NUM_HASHES_BATCH = (1); // This is really the total number of threads we want on the GPU (16M   )


const int CTX_SIZE_BYTES = 8*20; // 160
const int KEY_SIZE_BYTES = 32;
const int HASH_NO_SIG_SIZE_BYTES = 32;
const int TOTAL_BYTES_SEND = CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES;


struct SharedData {
    volatile bool is_stage1;
    volatile uint64_t nonce;
    volatile uint8_t data[TOTAL_BYTES_SEND];      // Buffer to send data
    volatile uint32_t utxo_set_idx4host;
    volatile uint32_t utxo_set_time4host;
    bool is_data_ready;  // Flag to indicate if data is ready
    STAGE1_S stage1_data;
};



WORD nonce[1] = {0};



int main( int argc, char* argv[] ) {

    int gpu_num = 0; // default
    if ( argc == 2 )
    {
        gpu_num = (int)atoi(argv[1]);
    }


    const int CTX_SIZE_BYTES = 8*20; // 160
    const int KEY_SIZE_BYTES = 32;
    const int HASH_NO_SIG_SIZE_BYTES = 32;
    const int TOTAL_BYTES_SEND = CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES;

    const int NONCE_SIZE_BYTES = 8;


    uint8_t *d_gpu_num;
    uint8_t *h_gpu_num = new uint8_t[1];
    *h_gpu_num = gpu_num;

    uint8_t *d_is_stage1;
    uint8_t *h_is_stage1 = new uint8_t[1];
    //////////////////////STAGE2==================

    uint8_t *d_ctx_data;
    uint8_t *h_ctx_data = new uint8_t[CTX_SIZE_BYTES];


    uint8_t *d_key_data;
    uint8_t *h_key_data = new uint8_t[KEY_SIZE_BYTES];    



    uint8_t *d_hash_no_sig_data;
    uint8_t *h_hash_no_sig_data = new uint8_t[HASH_NO_SIG_SIZE_BYTES];      


    uint8_t *d_nonce_data;
    uint8_t *h_nonce_data = new uint8_t[NONCE_SIZE_BYTES];      

        //////////////////////STAGE1==================

    uint8_t *d_utxo_set_idx4host;
    uint8_t *h_utxo_set_idx4host = new uint8_t[d_utxo_set_idx4host_BYTES];   
    uint8_t *d_utxo_set_time4host;
    uint8_t *h_utxo_set_time4host = new uint8_t[d_utxo_set_idx4host_BYTES];       

    uint8_t *d_utxos_block_from_time;
    uint8_t *h_utxos_block_from_time = new uint8_t[WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH];   


    uint8_t *d_utxos_hash;
    uint8_t *h_utxos_hash = new uint8_t[WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH]; // 256 bits per hash

 
    uint8_t *d_utxos_n;
    uint8_t *h_utxos_n = new uint8_t[WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH]; 


    uint8_t *d_start_time;
    uint8_t *h_start_time = new uint8_t[START_TIME_BYTES];



    uint8_t *d_hash_merkle_root;
    uint8_t *h_hash_merkle_root = new uint8_t[HASH_MERKLE_ROOT_BYTES*HDR_DEPTH]; // 256 bits per hash


    uint8_t *d_hash_prev_block;
    uint8_t *h_hash_prev_block = new uint8_t[HASH_PREV_BLOCK_BYTES*HDR_DEPTH]; // 256 bits per hash


    uint8_t *d_n_bits;
    uint8_t *h_n_bits = new uint8_t[N_BITS_BYTES*HDR_DEPTH]; 


    uint8_t *d_n_time;
    uint8_t *h_n_time = new uint8_t[N_TIME_BYTES*HDR_DEPTH]; 


    uint8_t *d_prev_stake_hash;
    uint8_t *h_prev_stake_hash = new uint8_t[PREV_STAKE_HASH_BYTES*HDR_DEPTH];


    uint8_t *d_prev_stake_n;
    uint8_t *h_prev_stake_n = new uint8_t[PREV_STAKE_N_BYTES*HDR_DEPTH];  


    uint8_t *d_block_sig;
    uint8_t *h_block_sig = new uint8_t[BLOCK_SIG_BYTES*HDR_DEPTH]; 

    
    uint8_t *d_stake_modifier;
    uint8_t *h_stake_modifier = new uint8_t[STAKE_MODIFIER_BYTES]; 
    ///////////////////////////////////////////////////////////////////////////


    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, deviceId);

    std::cout << "BTCW GPU MINER RELEASE v26.4.71 - specify GPU number - 133 utxos per loop fast scan Dec 25 2024" << std::endl;

    std::cout << "Max threads per block: " << deviceProps.maxThreadsPerBlock << std::endl;


    // For a 1D grid:
    printf("Max grid size in X: %d\n", deviceProps.maxGridSize[0]); // x-dimension
    printf("Max grid size in Y: %d\n", deviceProps.maxGridSize[1]); // y-dimension
    printf("Max grid size in Z: %d\n", deviceProps.maxGridSize[2]); // z-dimension

    // Allocate memory on the device
    cudaMalloc(&d_gpu_num, 1);


    // Allocate memory on the device
    cudaMalloc(&d_is_stage1, 1);
    memset(h_is_stage1, 1, 1); // initizlize in stage1 on gpu side

    // Allocate memory on the device
    cudaMalloc(&d_ctx_data, CTX_SIZE_BYTES);

    // Initialize data on the host (CPU)
    for (int i = 0; i < CTX_SIZE_BYTES; ++i) {
        h_ctx_data[i] = 0;
    }

    // Allocate memory on the device
    cudaMalloc(&d_key_data, KEY_SIZE_BYTES);

    // Initialize data on the host (CPU)
    for (int i = 0; i < KEY_SIZE_BYTES; ++i) {
        h_key_data[i] = 0;
    }

    // Allocate memory on the device
    cudaMalloc(&d_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES);

    // Initialize data on the host (CPU)
    for (int i = 0; i < HASH_NO_SIG_SIZE_BYTES; ++i) {
        h_hash_no_sig_data[i] = 0;
    }    


    // Allocate memory on the device
    cudaMalloc(&d_nonce_data, NONCE_SIZE_BYTES);

    // Initialize data on the host (CPU)
    for (int i = 0; i < NONCE_SIZE_BYTES; ++i) {
        h_nonce_data[i] = 0;
    }    


    // Allocate memory on the device
    cudaMalloc(&d_utxo_set_idx4host, d_utxo_set_idx4host_BYTES);
    cudaMalloc(&d_utxo_set_time4host, d_utxo_set_idx4host_BYTES);
    // Initialize data on the host (CPU)
    for (int i = 0; i < d_utxo_set_idx4host_BYTES; ++i) {
        h_utxo_set_idx4host[i] = 0;
        h_utxo_set_time4host[i] = 0;
    }   


    // Initialize data on the host (CPU)
    for (int i = 0; i < START_TIME_BYTES; ++i) {
        h_start_time[i] = 0;
    }   

    // Initialize data on the host (CPU)
    for (int i = 0; i < STAKE_MODIFIER_BYTES; ++i) {
        h_stake_modifier[i] = 5;
    }       

    


    

    // Allocate memory on the device
    cudaMalloc(&d_stake_modifier, STAKE_MODIFIER_BYTES);
    cudaMalloc(&d_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH);
    cudaMalloc(&d_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH);
    cudaMalloc(&d_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH);
    cudaMalloc(&d_start_time, START_TIME_BYTES);
    cudaMalloc(&d_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH);
    cudaMalloc(&d_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH);
    cudaMalloc(&d_n_bits, N_BITS_BYTES*HDR_DEPTH);
    cudaMalloc(&d_n_time, N_TIME_BYTES*HDR_DEPTH);
    cudaMalloc(&d_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH);
    cudaMalloc(&d_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH);
    cudaMalloc(&d_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    cudaMalloc(&d_stake_modifier, STAKE_MODIFIER_BYTES);



    // Copy the data to the GPU
    cudaMemcpy(d_gpu_num, h_gpu_num, 1, cudaMemcpyHostToDevice);

    // Stage2 stuff  
    cudaMemcpy(d_is_stage1, h_is_stage1, 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce_data, h_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxo_set_idx4host, h_utxo_set_idx4host, d_utxo_set_idx4host_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxo_set_time4host, h_utxo_set_time4host, d_utxo_set_idx4host_BYTES, cudaMemcpyHostToDevice);

    // Stage1 stuff
    
    cudaMemcpy(d_utxos_block_from_time, h_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_hash, h_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_n, h_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_time, h_start_time, START_TIME_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_merkle_root, h_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_prev_block, h_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bits, h_n_bits, N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_time, h_n_time, N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_hash, h_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_n, h_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_sig, h_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    cudaMemcpy(d_stake_modifier, h_stake_modifier, STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 



    // Create a stream for asynchronous operations
    cudaStream_t stream, kernel_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&kernel_stream);

    // Tell the miner which GPU number it is
    cudaMemcpyAsync(d_gpu_num, h_gpu_num, 1, cudaMemcpyHostToDevice, stream);

    // Copy the modified data from the host back to the GPU asynchronously  
    cudaMemcpyAsync(d_is_stage1, h_is_stage1, 1, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nonce_data, h_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_utxo_set_idx4host, h_utxo_set_idx4host, d_utxo_set_idx4host_BYTES, cudaMemcpyHostToDevice, stream);
    

    // Stage1 stuff

    cudaMemcpyAsync(d_utxos_block_from_time, h_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_utxos_hash, h_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_utxos_n, h_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_start_time, h_start_time, START_TIME_BYTES, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hash_merkle_root, h_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hash_prev_block, h_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_n_bits, h_n_bits, N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_n_time, h_n_time, N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_prev_stake_hash, h_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_prev_stake_n, h_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_sig, h_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(d_stake_modifier, h_stake_modifier, STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice, stream); 


    // Wait for the kernel to complete
    cudaStreamSynchronize(stream);



    cudaMemcpy(d_utxos_block_from_time, h_utxos_block_from_time, WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_hash, h_utxos_hash, WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_utxos_n, h_utxos_n, WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_time, h_start_time, START_TIME_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_merkle_root, h_hash_merkle_root, HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_prev_block, h_hash_prev_block, HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bits, h_n_bits, N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_time, h_n_time, N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_hash, h_prev_stake_hash, PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_stake_n, h_prev_stake_n, PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_sig, h_block_sig, BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    cudaMemcpy(d_stake_modifier, h_stake_modifier, STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 
    //===========================================KERNEL======================================================
    // We are starting the KERNEL with NO DATA - This is intentional, data will be given to it on the fly from the BTCW node.
    cuda_miner<<<128, 256, 0, kernel_stream>>>(d_gpu_num, d_is_stage1, d_key_data, d_ctx_data, d_hash_no_sig_data, d_nonce_data, d_utxo_set_idx4host, d_utxo_set_time4host, d_stake_modifier, d_utxos_block_from_time, d_utxos_hash, d_utxos_n, d_start_time, d_hash_merkle_root, d_hash_prev_block, d_n_bits, d_n_time, d_prev_stake_hash, d_prev_stake_n, d_block_sig);

    //=================================================================================================================

  

    // Open shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Error opening shared memory" << std::endl;
        return 1;
    }

    // Map shared memory into the process's address space
    SharedData* shared_data = (SharedData*) mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_data == MAP_FAILED) {
        std::cerr << "Error mapping shared memory" << std::endl;
        return 1;
    }


    secp256k1_ecmult_gen_context ctx_obj;
    secp256k1_ecmult_gen_context *ctx = &ctx_obj;
    uint256 hash_no_sig;
    unsigned char key_data[32];

    bool is_stage1 = true;


    uint64_t *p_data = (uint64_t *)shared_data->data;


    // Cast to the volatile pointer to ensure we don't optimize reads/writes
    volatile SharedData* mapped_data = (volatile SharedData*) shared_data;


    uint32_t throttle = 0x0;
    bool is_init = false;

    while ( true )
    {

        // one time init
        if ( !is_init )
        {
            cudaMemcpyAsync(d_utxos_block_from_time, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_utxos_block_from_time[0])), WALLET_UTXOS_TIME_FROM_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_utxos_hash, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_utxos_hash[0])), WALLET_UTXOS_HASH_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_utxos_n, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_utxos_n[0])), WALLET_UTXOS_N_BYTES*WALLET_UTXOS_LENGTH, cudaMemcpyHostToDevice, stream);
            is_init = true;
        }
        
        if ( (throttle % 0x3) == 0 )
        {

        
            if ( shared_data->is_stage1 )
            {
                // we are in stage1
                printf("STAGE1 BLOCK DATA - CPU SIDE\n");

                
                cudaMemcpyAsync(d_start_time, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_start_time[0])), START_TIME_BYTES, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_stake_modifier, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_stake_modifier[0])), STAKE_MODIFIER_BYTES, cudaMemcpyHostToDevice, stream);
                
                cudaMemcpyAsync(d_hash_merkle_root, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_hash_merkle_root[0])), HASH_MERKLE_ROOT_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_hash_prev_block, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_hash_prev_block[0])), HASH_PREV_BLOCK_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_n_bits, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_n_bits[0])), N_BITS_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_n_time, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_n_time[0])), N_TIME_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_prev_stake_hash, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_prev_stake_hash[0])), PREV_STAKE_HASH_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_prev_stake_n, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_prev_stake_n[0])), PREV_STAKE_N_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_block_sig, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->stage1_data.h_block_sig[0])), BLOCK_SIG_BYTES*HDR_DEPTH, cudaMemcpyHostToDevice, stream); // 1st byte is length   either 70 or 71  unused last byte if 70 bytes 


            }
            else
            {

                // we are in stage2

                //if ( is_stage1 )
                //{
                    // we need to update our state. this is a transition. Copy over the data for stage2.
                    is_stage1 = false;

                    //Host update the data, send it to the GPU
                    printf("STAGE2 BLOCK DATA - CPU SIDE\n");


                    // Data set from BTCW node
                    memcpy( &h_key_data[0], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[0])), 32 );


                    memcpy( &h_ctx_data[0],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[32])), 8 );
                    memcpy( &h_ctx_data[8],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[40])), 8 );
                    memcpy( &h_ctx_data[16], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[48])), 8 );
                    memcpy( &h_ctx_data[24], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[56])), 8 );

                    memcpy( &h_ctx_data[32],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[64])), 8 );
                    memcpy( &h_ctx_data[40],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[72])), 8 );
                    memcpy( &h_ctx_data[48],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[80])), 8 );
                    memcpy( &h_ctx_data[56],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[88])), 8 );
                    memcpy( &h_ctx_data[64],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[96])), 8 );
                    memcpy( &h_ctx_data[72],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[104])), 8 );
                    memcpy( &h_ctx_data[80],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[112])), 8 );
                    memcpy( &h_ctx_data[88],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[120])), 8 );
                    memcpy( &h_ctx_data[96],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[128])), 8 );
                    memcpy( &h_ctx_data[104], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[136])), 8 );
                    memcpy( &h_ctx_data[112], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[144])), 8 );
                    memcpy( &h_ctx_data[120], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[152])), 8 );
                    memcpy( &h_ctx_data[128], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[160])), 8 );
                    memcpy( &h_ctx_data[136], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[168])), 8 );
                    memcpy( &h_ctx_data[144], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[176])), 8 );

                    memcpy( &h_ctx_data[152], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[184])), 4 );
                    memcpy( &h_ctx_data[156], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[188])), 4 );


                    memcpy( &h_hash_no_sig_data[0],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 8);
                    memcpy( &h_hash_no_sig_data[8],  const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[200])), 8);
                    memcpy( &h_hash_no_sig_data[16], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[208])), 8);
                    memcpy( &h_hash_no_sig_data[24], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[216])), 8);


                    // Copy the modified data from the host back to the GPU asynchronously  
                    cudaMemcpyAsync(d_ctx_data, h_ctx_data, CTX_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_key_data, h_key_data, KEY_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_hash_no_sig_data, h_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, cudaMemcpyHostToDevice, stream);
                //}

            }

        }

        throttle++;

        // convert from bool to 1 byte
        if ( shared_data->is_stage1 )
        {
            h_is_stage1[0] = 0xFF;
        }
        else
        {
            h_is_stage1[0] = 0;
        }

        cudaMemcpyAsync(d_is_stage1, h_is_stage1, 1, cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->nonce)), d_nonce_data, NONCE_SIZE_BYTES, cudaMemcpyDeviceToHost, stream); 

        cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->utxo_set_idx4host)), d_utxo_set_idx4host, d_utxo_set_idx4host_BYTES, cudaMemcpyDeviceToHost, stream); 
        cudaMemcpyAsync(const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->utxo_set_time4host)), d_utxo_set_time4host, d_utxo_set_idx4host_BYTES, cudaMemcpyDeviceToHost, stream); 


        usleep(500000);

    }


    // Wait for the kernel to complete
    cudaStreamSynchronize(kernel_stream);



    // Cleanup


    cudaFree(d_gpu_num);
    delete[] h_gpu_num;

    cudaFree(d_is_stage1);
    delete[] h_is_stage1;

    cudaFree(d_ctx_data);
    delete[] h_ctx_data;

    cudaFree(d_key_data);
    delete[] h_key_data;

    cudaFree(d_hash_no_sig_data);
    delete[] h_hash_no_sig_data;

    cudaFree(d_nonce_data);
    delete[] h_nonce_data;

    cudaFree(d_utxo_set_idx4host);
    delete[] h_utxo_set_idx4host;

    cudaFree(d_stake_modifier);
    delete[] h_stake_modifier;

    cudaFree(d_utxos_hash);
    delete[] h_utxos_hash;

    cudaFree(d_utxos_block_from_time);
    delete[] h_utxos_block_from_time;


    cudaFree(d_utxos_n);
    delete[] h_utxos_n;

    cudaFree(d_start_time);
    delete[] h_start_time;


    cudaFree(d_hash_merkle_root);
    delete[] h_hash_merkle_root;

    cudaFree(d_hash_prev_block);
    delete[] h_hash_prev_block;

    cudaFree(d_n_bits);
    delete[] h_n_bits;

    cudaFree(d_n_time);
    delete[] h_n_time;

    cudaFree(d_prev_stake_hash);
    delete[] h_prev_stake_hash;

    cudaFree(d_prev_stake_n);
    delete[] h_prev_stake_n;

    cudaFree(d_block_sig);
    delete[] h_block_sig;



    cudaStreamDestroy(stream);
    cudaStreamDestroy(kernel_stream);


    munmap(shared_data, sizeof(SharedData));
    close(shm_fd);    

    return 0;
}

